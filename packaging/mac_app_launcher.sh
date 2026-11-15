#!/usr/bin/env bash
# =============================================================================
# ISL Translator — macOS Application Bundle Entrypoint
# =============================================================================
# This script executes inside dist/ISL Translator.app/Contents/MacOS/
# It dynamically locates the workspace root (4 levels up) and runs the live
# project using the workspace virtualenv, guaranteeing immediate access to the
# latest frontend UI, backend models, and translation engines.
# =============================================================================

set -e

# Resolve absolute path of this executable's directory
MACOS_DIR="$(cd "$(dirname "$0")" && pwd)"

# Dynamically resolve project root:
# 1. Explicit environment variable override
# 2. Inside dist/ISL Translator.app (4 levels up)
# 3. Inside root ISL Translator.app (3 levels up)
# 4. Standard user Downloads folder fallback
if [ -n "$ISL_PROJECT_ROOT" ] && [ -f "$ISL_PROJECT_ROOT/packaging/launcher.py" ]; then
    PROJECT_ROOT="$ISL_PROJECT_ROOT"
elif [ -f "$MACOS_DIR/../../../../packaging/launcher.py" ]; then
    PROJECT_ROOT="$(cd "$MACOS_DIR/../../../.." && pwd)"
elif [ -f "$MACOS_DIR/../../../packaging/launcher.py" ]; then
    PROJECT_ROOT="$(cd "$MACOS_DIR/../../.." && pwd)"
elif [ -d "$HOME/Downloads/Indian-Sign-Language-Recognition" ] && [ -f "$HOME/Downloads/Indian-Sign-Language-Recognition/packaging/launcher.py" ]; then
    PROJECT_ROOT="$HOME/Downloads/Indian-Sign-Language-Recognition"
else
    PROJECT_ROOT="$(cd "$MACOS_DIR/../../.." && pwd)"
fi

# Verify Python Virtual Environment
VENV_PYTHON="$PROJECT_ROOT/venv/bin/python"
if [ ! -f "$VENV_PYTHON" ] || [ ! -x "$VENV_PYTHON" ]; then
    MSG="Python virtual environment not found at:\n$VENV_PYTHON\n\nPlease ensure venv exists and is configured."
    echo "[ISL Translator Error] $MSG" >&2
    osascript -e 'display alert "ISL Translator Error" message "'"$MSG"'" as critical' 2>/dev/null || true
    exit 1
fi

# Verify Packaging Launcher
LAUNCHER_SCRIPT="$PROJECT_ROOT/packaging/launcher.py"
if [ ! -f "$LAUNCHER_SCRIPT" ]; then
    MSG="Launcher script not found at:\n$LAUNCHER_SCRIPT"
    echo "[ISL Translator Error] $MSG" >&2
    osascript -e 'display alert "ISL Translator Error" message "'"$MSG"'" as critical' 2>/dev/null || true
    exit 1
fi

# Switch working directory to project root
cd "$PROJECT_ROOT"

# Forward execution to Python launcher
exec "$VENV_PYTHON" "$LAUNCHER_SCRIPT" "$@"

