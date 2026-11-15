# Installation and Setup Guide

This guide details the verified steps to set up, configure, and run the **Indian Sign Language (ISL) Recognition & Translation System** on macOS and Linux.

---

## 1. System Requirements

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Operating System** | macOS 12+ (Monterey, Ventura, Sonoma, Sequoia) or Ubuntu 20.04+ | macOS 14+ (Apple Silicon M1/M2/M3/M4) |
| **Python** | Python 3.9 – 3.11 | Python 3.10 |
| **Memory (RAM)** | 4 GB | 8 GB or higher |
| **Camera** | Built-in webcam or external USB camera (720p 30fps) | 1080p 30fps webcam with good lighting |
| **Audio** | Speaker / audio output (for Text-to-Speech) | Native system audio |
| **Disk Space** | ~2 GB (including dependencies and virtual environment) | ~5 GB (if building standalone `.app` bundle) |

---

## 2. Step-by-Step Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/indian-sign-language-recognition.git
cd indian-sign-language-recognition
```

### Step 2: Set Up Python 3.10 Virtual Environment

> [!IMPORTANT]
> **Why Python 3.10 is Recommended:**
> The computer vision and neural perception pipeline requires **MediaPipe (`0.10.21`)** and **TensorFlow (`2.16.2`)**. On macOS Apple Silicon (M1/M2/M3/M4) and Linux, pre-compiled ARM64 wheels are officially verified and distributed for Python 3.9–3.11. Modern macOS installations frequently default to newer Python versions (such as Python 3.12, 3.13, or 3.14), which lack pre-compiled binary wheels for TensorFlow/MediaPipe and will fail during dependency installation.

#### macOS (Apple Silicon & Intel via Homebrew):

```bash
# 1. Install Python 3.10 if not already installed
brew install python@3.10

# 2. Verify Python 3.10 is available
python3.10 --version

# 3. Create virtual environment named 'venv'
python3.10 -m venv venv

# 4. Activate the virtual environment
source venv/bin/activate
```

#### Linux / General Setup (where `python3` is Python 3.10):

```bash
# Verify Python version (must be 3.9 - 3.11)
python3 --version

# Create virtual environment named 'venv'
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

*(To deactivate when finished later, run `deactivate`)*

### Step 3: Upgrade Pip & Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Core Dependencies Overview:
- **`mediapipe==0.10.21`**: Cross-platform ML solutions for hand landmark tracking (21 3D normalized keypoints).
- **`tensorflow==2.16.2` / `tf-keras==2.16.0`**: Neural network execution engine for Landmark MLP and visual CNN.
- **`opencv-python==4.11.0.86`**: Video capture, frame preprocessing, and HUD overlay generation.
- **`flask==3.1.3`**: High-performance local web server hosting the interactive UI, MJPEG video feed, and SSE stream.
- **`scikit-learn==1.7.2`**: Continuous learning, personal adapters, and feature evaluation.
- **`pillow==12.3.0`**: Image cropping and bounding box extraction for visual CNN.

### Step 4: Verify Model Artifacts

All necessary pre-trained model weights and label encoders are included directly within the `models/` directory (~7.1 MB total). Ensure all required files are present:

```bash
ls -lh models/
```

Expected files:
- `alphabet_model_26class.h5` (~295 KB) — Production 26-class Dense Landmark fingerspelling model
- `alphabet_label_map.json` — Mapping for characters A–Z
- `alphabet_model_26class_summary.txt` — Architecture specification
- `cnn_sign_model.h5` (~3.2 MB) — Production 128x128 RGB visual CNN model
- `cnn_label_map.json` — 26-class character mapping for CNN
- `experimental/` — Dual-hand 144D feature models and ablation checkpoints

### Step 5: Run Automated Tests

Verify that your local environment satisfies all model, inference, debounce, and translation contracts:

```bash
python -m unittest discover -s tests
```

**Expected output:**
```text
Ran 235 tests in ~12.5s
OK
```

---

## 3. Launching the Application

### Option A: Double-Click Desktop Launch (Recommended for macOS)

1. Open Finder to the repository directory.
2. Double-click **`ISL Translator.app`** (or **`Launch ISL Translator.command`**).
3. The launcher automatically:
   - Verifies models, templates, and static assets.
   - Binds port 5001 with single-instance safety.
   - Polls health and launches your default browser at **`http://127.0.0.1:5001`**.

### Option B: Terminal Command Launch

Run the Python launcher directly:

```bash
python packaging/launcher.py
```

Or start Flask directly:
```bash
PORT=5001 python src/app.py
```

Open your browser to: **`http://127.0.0.1:5001`**

---

## 4. macOS Specific Notes & Troubleshooting

### Camera Permissions
On macOS, your terminal emulator (Terminal, iTerm2, or VS Code / IDE) requires explicit permission to access the webcam:
1. Open **System Settings** > **Privacy & Security** > **Camera**.
2. Ensure your terminal application (e.g., Terminal, VS Code) is toggled **ON**.
3. If the camera feed is blank or shows error `Cannot open camera index 0`, verify that no other application (Zoom, FaceTime, Photo Booth) is using the camera.
4. If you have multiple webcams connected, you can specify camera index in `src/inference_engine.py` or `src/mediapipe_utils.py` by changing `cv2.VideoCapture(0)` to index `1`.

### Port Conflicts
If port 5000 is occupied (e.g., by macOS AirPlay Receiver on macOS Monterey or newer):
```bash
# Check what is listening on port 5000
lsof -i :5000

# Disable AirPlay Receiver: System Settings > General > AirDrop & AirPlay > AirPlay Receiver -> OFF
# Or launch on a different port:
FLASK_RUN_PORT=5001 python src/app.py
```

### Standalone macOS Application Packaging
If you wish to build a native standalone `.app` bundle for macOS:
```bash
cd packaging
chmod +x build_app.sh
./build_app.sh
```
The compiled bundle will be generated in `dist/ISL Translator.app`.
