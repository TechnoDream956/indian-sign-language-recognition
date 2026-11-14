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

### Step 2: Set Up Python Virtual Environment

It is strongly recommended to use a dedicated Python 3.10 virtual environment to avoid dependency conflicts.

```bash
# Verify Python version (must be 3.9 - 3.11)
python3 --version

# Create virtual environment named 'venv'
python3 -m venv venv

# Activate virtual environment
# On macOS / Linux:
source venv/bin/activate
```

*(To deactivate when finished later, run `deactivate`)*

### Step 3: Upgrade Pip & Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Core Dependencies Overview:
- **`mediapipe==0.10.9`**: Cross-platform ML solutions for pose, hand, and face landmark tracking.
- **`tensorflow==2.15.0`** (or `tensorflow-macos` on Apple Silicon): Neural network execution engine.
- **`opencv-python`**: Video capture, frame preprocessing, and HUD overlay generation.
- **`flask`**: Asynchronous web server hosting the interactive UI, MJPEG video feed, and SSE stream.
- **`googletrans==4.0.0-rc1`**: Neural machine translation engine for 10 regional Indian languages.
- **`pyttsx3` / `gTTS`**: Offline and cloud speech synthesis backends.

### Step 4: Verify Model Artifacts

All necessary pre-trained model weights and label encoders are included directly within the `models/` directory (~11 MB total). Ensure all required files are present:

```bash
ls -lh models/
```

Expected files:
- `alphabet_model_26class.h5` (~292 KB) — 26-class Dense fingerspelling model
- `alphabet_label_map.json` — Mapping for characters A–Z
- `alphabet_model_26class_summary.txt` — Architecture specification
- `mopgru_model.h5` (~5.2 MB) — 13-class foundational MoP-GRU word model
- `labelencoder.pkl` — 13-class label encoder
- `mopgru_model_29class.h5` (~5.2 MB) — 29-class extended MoP-GRU word model
- `labelencoder_29class.pkl` — 29-class label encoder

### Step 5: Run Automated Tests

Verify that your local environment satisfies all model, inference, debounce, and translation contracts:

```bash
python -m unittest discover -s tests
```

**Expected output:**
```text
Ran 87 tests in ~0.7s
OK
```

---

## 3. Launching the Application

### Option A: Interactive Web Application (Recommended)

Run the Flask web server:

```bash
python src/app.py
```

Output:
```text
 * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```

Open your browser (Google Chrome, Apple Safari, or Mozilla Firefox) and navigate to:
**`http://127.0.0.1:5000`**

### Option B: Solo Dual-Hand Alphabet Collection Utility

To collect new sign samples or practice fingerspelling with automatic stability gating and countdowns:

```bash
python src/collect_dual_hand_alphabet.py --solo --samples-per-label 5 --output-dir data/phase5p_dual_hand
```

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
