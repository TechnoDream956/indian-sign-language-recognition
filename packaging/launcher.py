#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISL Translator — macOS Application Launcher
============================================
Production entry point for the packaged .app bundle.

Responsibilities:
1. Detect frozen (PyInstaller) vs. development environment
2. Prevent duplicate processes (Single-Instance Safety)
3. Resolve all resource paths (models, templates, static, src modules)
4. Patch module-level path constants BEFORE any model loading
5. Start the Flask server and InferenceEngine
6. Poll server health dynamically and open the browser once ready
7. Handle clean shutdown (camera, MediaPipe, TTS)
"""

import os
import signal
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

# Configure unbuffered I/O so logs are immediately visible
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
#  Suppress TensorFlow / MediaPipe noisy logging before any imports
# ---------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "0"

# ---------------------------------------------------------------------------
#  Path Resolution — works for both frozen (.app) and development modes
# ---------------------------------------------------------------------------

def get_resource_dir() -> str:
    """Return the base resource directory.
    
    When frozen by PyInstaller, resources are extracted to sys._MEIPASS.
    When running from source, resources are relative to this file's parent.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    else:
        # Development mode: packaging/ is a sibling of src/ and models/
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


RESOURCE_DIR = get_resource_dir()
SRC_DIR = os.path.join(RESOURCE_DIR, "src")
MODELS_DIR = os.path.join(RESOURCE_DIR, "models")
TEMPLATE_DIR = os.path.join(RESOURCE_DIR, "src", "templates")
STATIC_DIR = os.path.join(RESOURCE_DIR, "src", "static")

# Ensure src/ is importable
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Also ensure the resource dir itself is available
if RESOURCE_DIR not in sys.path:
    sys.path.insert(0, RESOURCE_DIR)


# ---------------------------------------------------------------------------
#  Port allocation & Single-Instance Safety
# ---------------------------------------------------------------------------

def check_existing_instance(port: int = 5001) -> bool:
    """Check if an ISL Translator instance is already serving on port."""
    url = f"http://127.0.0.1:{port}/api/status"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ISL-Launcher"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass
    return False


def find_available_port(preferred: int = 5001) -> int:
    """Find an available TCP port starting from preferred."""
    for port in range(preferred, preferred + 100):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
        finally:
            sock.close()
    return preferred


# ---------------------------------------------------------------------------
#  Main Entrypoint
# ---------------------------------------------------------------------------

def main():
    from pathlib import Path

    print("\n🤟  ISL Translator — macOS Application", flush=True)

    # 1. Single-Instance Check — prevent laptop freeze from duplicate instances
    if check_existing_instance(5001):
        print("     ✓ ISL Translator is already running on http://127.0.0.1:5001", flush=True)
        print("     Activating existing application in default browser...", flush=True)
        webbrowser.open("http://127.0.0.1:5001")
        sys.exit(0)

    launcher_start_time = time.time()
    print(f"     Resource dir: {RESOURCE_DIR}", flush=True)
    print(f"     Models dir:   {MODELS_DIR}", flush=True)
    print(f"     Templates:    {TEMPLATE_DIR}", flush=True)
    print(f"     Static:       {STATIC_DIR}", flush=True)

    # 2. Verify critical assets exist (Alphabet-Only)
    critical_files = [
        os.path.join(MODELS_DIR, "alphabet_model_26class.h5"),
        os.path.join(MODELS_DIR, "alphabet_label_map.json"),
        os.path.join(TEMPLATE_DIR, "index.html"),
        os.path.join(STATIC_DIR, "style.css"),
        os.path.join(STATIC_DIR, "app.js"),
    ]
    for fpath in critical_files:
        if not os.path.exists(fpath):
            print(f"     ❌ MISSING: {fpath}", flush=True)
            sys.exit(1)
    print("     ✓ All critical assets verified.", flush=True)

    # 3. Patch module-level path constants BEFORE app.py imports
    try:
        t0 = time.time()
        import inference_engine as ie_module
        ie_module.WORKSPACE_DIR = Path(RESOURCE_DIR)
        ie_module.MODELS_DIR = Path(MODELS_DIR)

        import alphabet_engine as ae_module
        ae_module.WORKSPACE_ROOT = Path(RESOURCE_DIR)
        ae_module.DEFAULT_MODEL_PATH = Path(MODELS_DIR) / "alphabet_model_26class.h5"
        ae_module.DEFAULT_LABEL_MAP_PATH = Path(MODELS_DIR) / "alphabet_label_map.json"

        from app import app, engine
        t_patch = time.time() - t0
        print(f"     ✓ Modules initialized in {t_patch:.3f}s", flush=True)
    except Exception as e:
        import traceback
        err_msg = f"Fatal initialization error: {e}\n{traceback.format_exc()}"
        print(f"\n❌ {err_msg}", flush=True)
        try:
            with open("/tmp/isl_translator_error.log", "w") as ef:
                ef.write(err_msg)
        except Exception:
            pass
        sys.exit(1)

    # Patch Flask template/static dirs
    app.template_folder = TEMPLATE_DIR
    app.static_folder = STATIC_DIR

    # Find an available port (preferred 5001)
    port = find_available_port(5001)
    print(f"     Port:         {port}", flush=True)

    # Start inference engine
    t_eng0 = time.time()
    print("     Starting inference engine...", flush=True)
    engine.start()
    print(f"     ✓ Inference engine background loop started in {time.time() - t_eng0:.3f}s", flush=True)

    # Dynamic server health polling for browser launch
    def open_browser():
        server_url = f"http://127.0.0.1:{port}"
        health_url = f"{server_url}/api/status"
        start_time = time.time()
        ready = False
        while time.time() - start_time < 30.0:
            try:
                req = urllib.request.Request(health_url, headers={"User-Agent": "ISL-Launcher"})
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.15)
        elapsed = time.time() - launcher_start_time
        if ready:
            print(f"     ✓ Server ready in {elapsed:.2f}s. Opening browser at {server_url}", flush=True)
        else:
            print(f"     [Warning] Health check timed out after {elapsed:.2f}s. Attempting browser open at {server_url}", flush=True)
        webbrowser.open(server_url)

    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    print(f"     Server: http://127.0.0.1:{port}", flush=True)
    print("     The application will open in your default browser.", flush=True)
    print("     Use Quit App in browser, Cmd+Q, or close window to quit.\n", flush=True)

    # Idempotent Lifecycle Shutdown Coordinator
    shutdown_lock = threading.Lock()
    is_shutting_down = False

    def shutdown_coordinator(signum=None, frame=None):
        nonlocal is_shutting_down
        with shutdown_lock:
            if is_shutting_down:
                return
            is_shutting_down = True
        print("\n[Application Lifecycle] Shutdown requested. Releasing camera and stopping services...", flush=True)
        try:
            engine.stop()
        except Exception as e:
            print(f"[Warning] Engine stop error: {e}", flush=True)
        print("✓ Camera released and shutdown complete.", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_coordinator)
    signal.signal(signal.SIGINT, shutdown_coordinator)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, shutdown_coordinator)

    try:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Server error: {e}", flush=True)
    finally:
        shutdown_coordinator()


if __name__ == "__main__":
    main()
