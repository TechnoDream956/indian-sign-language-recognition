# Cross-Platform Installation and Setup Guide

This guide details verified steps to install, configure, and run the **Indian Sign Language (ISL) Recognition & Translation Desktop System** across macOS, Linux, and Windows.

**Project Contributors & Roles:**
- **Arpit:** Recognition Model Development
- **Sachin:** Data Collection, Preprocessing & Backend Support
- **Abhu:** Multilingual Support & TTS
- **Yash:** Frontend & Deployment

---

## 1. System Requirements

| Component | macOS | Linux (Ubuntu / Debian) | Windows 10 / 11 |
| :--- | :--- | :--- | :--- |
| **Operating System** | macOS 12+ (Monterey through Sequoia) | Ubuntu 20.04+ / Debian 11+ | Windows 10 / 11 (64-bit) |
| **Python Version** | Python 3.10 (3.9–3.11 supported) | Python 3.10 (3.9–3.11 supported) | Python 3.10 (3.9–3.11 supported) |
| **Architecture** | Apple Silicon (M1–M4) / Intel x86_64 | x86_64 / ARM64 | x86_64 |
| **Memory (RAM)** | 4 GB min (8 GB recommended) | 4 GB min (8 GB recommended) | 4 GB min (8 GB recommended) |
| **Webcam** | Built-in or USB (720p 30fps) | UVC-compatible USB camera | DirectShow / Media Foundation camera |
| **Audio Output** | Native `say` / `pyttsx3` / `afplay` | `espeak` / `aplay` / `paplay` | SAPI5 via `pyttsx3` / Browser speech |
| **Disk Space** | ~2 GB free disk space | ~2 GB free disk space | ~2 GB free disk space |

> [!IMPORTANT]
> **Why Python 3.10 is Recommended:**  
> The computer vision and neural perception pipeline requires **MediaPipe (`0.10.21`)** and **TensorFlow (`2.16.2`)**. Pre-compiled binary wheels are officially distributed and verified for Python 3.9–3.11. Modern operating systems defaulting to Python 3.12+ lack pre-compiled binary wheels for these packages and will fail during dependency installation.

---

## 2. Platform-Specific Setup

### A. macOS (Apple Silicon & Intel)

#### 1. Install Python 3.10 via Homebrew
```bash
brew install python@3.10
python3.10 --version
```

#### 2. Create and Activate Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install pyttsx3 gTTS deep-translator
```

#### 4. Camera Permissions on macOS
1. Open **System Settings** > **Privacy & Security** > **Camera**.
2. Verify your terminal emulator (Terminal, iTerm2, or VS Code) is toggled **ON**.
3. Camera capture automatically selects Apple Silicon native `cv2.CAP_AVFOUNDATION`.

---

### B. Linux (Ubuntu / Debian / Raspberry Pi OS 64-bit)

#### 1. Install System Prerequisites & Audio Drivers
```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 espeak libasound2-dev \
    pulseaudio-utils v4l-utils
```

#### 2. User Camera Permissions
Ensure your user account is in the `video` group:
```bash
sudo usermod -a -G video $USER
```
*(Log out and log back in for group membership to take effect)*

#### 3. Create and Activate Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
```

#### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install pyttsx3 gTTS deep-translator
```

Camera capture automatically utilizes Linux Video4Linux2 (`cv2.CAP_V4L2`), and TTS leverages `pyttsx3` (via eSpeak/speech-dispatcher) and `gTTS` with PulseAudio (`paplay`) / ALSA (`aplay`).

---

### C. Windows (Windows 10 / 11 64-bit)

#### 1. Install Python 3.10
Download Python 3.10 from [python.org](https://www.python.org/downloads/release/python-31011/) or run via Windows Package Manager:
```powershell
winget install Python.Python.3.10
```
> [!NOTE]
> During installation, ensure the checkbox **"Add Python 3.10 to PATH"** is checked.

#### 2. Create and Activate Virtual Environment
In PowerShell or Command Prompt:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

#### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyttsx3 gTTS deep-translator
```

Camera capture utilizes Windows DirectShow (`cv2.CAP_DSHOW`) and Media Foundation (`cv2.CAP_MSMF`). Offline speech synthesis uses Microsoft SAPI5 voices via `pyttsx3` with automatic browser Web Speech API fallback.

---

## 3. Verifying Model Checkpoints

Verify that all required pre-trained weights and label maps are present in the `models/` folder:

```bash
ls -lh models/
```

Expected files:
- `alphabet_model_26class.h5` (~295 KB) — 26-class Dense Landmark fingerspelling model
- `alphabet_label_map.json` — Mapping for characters A–Z
- `cnn_sign_model.h5` (~3.2 MB) — 128x128 RGB visual CNN model
- `cnn_label_map.json` — 26-class character mapping for CNN
- `experimental/phase5b2_prompt23_class_k/config_d_ablation_144d/best_val_model.h5` (~420 KB) — 20-class dual-hand landmark model

---

## 4. Running the Automated Test Suite

Run the full cross-platform test suite:

```bash
python -m unittest discover -s tests
```

**Expected Result:**
```text
Ran 238 tests in ~30s
OK
```

---

## 5. Launching the Application

### Option A: Double-Click Desktop Launch (macOS)
1. Double-click **`ISL Translator.app`** (or **`Launch ISL Translator.command`**).
2. The launcher performs environment validation, model verification, binds port 5001, and opens your browser.

### Option B: Terminal Command Launch (Cross-Platform)
From the project root:

```bash
# macOS / Linux
PORT=5001 python src/app.py

# Windows PowerShell
$env:PORT=5001; python src/app.py
```

Open your web browser to:
**`http://127.0.0.1:5001`**

---

## 6. TTS & Translation Fallback Architecture

The system features an automatic 4-tier speech synthesis fallback chain:
1. **Tier 1:** Native macOS `say` (macOS native voices: Rishi, Lekha, Piya, Vani, Geeta, Soumya)
2. **Tier 2:** Offline cross-platform `pyttsx3` (Windows SAPI5, Linux eSpeak, macOS NSSpeechSynthesizer)
3. **Tier 3:** High-fidelity `gTTS` + local audio player (`afplay` / `paplay` / `aplay` / `powershell`)
4. **Tier 4:** Browser Web Speech API fallback (zero-configuration fallback for headless environments, Docker, and systems without local audio drivers)

Translation features online neural translation via MyMemory API with automatic exponential backoff retries and offline dictionary / transliteration fallback.
