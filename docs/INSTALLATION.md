# Cross-Platform Installation and Setup Guide

This guide details verified steps to install, configure, and run the **Indian Sign Language (ISL) Recognition & Translation Desktop System** across macOS, Linux, and Windows.

**Project Contributors & Roles:**
- **Arpit:** Recognition Model Development
- **Sachin:** Data Collection, Preprocessing & Backend Support
- **Abhu:** Multilingual Support & TTS
- **Yash:** Frontend & Deployment

---

## 1. System Requirements

| Component | macOS (Verified Local Hardware) | Linux (Ubuntu / Debian)* | Windows 10 / 11* |
| :--- | :--- | :--- | :--- |
| **Operating System** | macOS 12+ (Apple Silicon M1–M4 / Intel) | Ubuntu 20.04+ / Debian 11+ | Windows 10 / 11 (64-bit) |
| **Python Version** | Python 3.10 (3.9–3.11 supported) | Python 3.10 (3.9–3.11 supported) | Python 3.10 (3.9–3.11 supported) |
| **Architecture** | Apple Silicon ARM64 / Intel x86_64 | x86_64 / ARM64 | x86_64 |
| **Memory (RAM)** | 4 GB min (8 GB recommended) | 4 GB min (8 GB recommended) | 4 GB min (8 GB recommended) |
| **Webcam** | Built-in or USB (720p 30fps) | UVC-compatible USB camera | DirectShow / Media Foundation camera |
| **Audio Output** | Native `say` / `pyttsx3` / `afplay` | `espeak` / `aplay` / `paplay` | SAPI5 via `pyttsx3` / Browser speech |
| **Disk Space** | ~2 GB free disk space | ~2 GB free disk space | ~2 GB free disk space |

*\*Note on Platform Testing: macOS Apple Silicon is the primary development and verified execution environment. Linux and Windows paths are implemented with standard cross-platform APIs (`pathlib`, DirectShow, V4L2, SAPI5, eSpeak) but are **untested on physical Linux/Windows hardware**.*

> [!IMPORTANT]
> **Why Python 3.10 is Recommended:**  
> The computer vision and neural perception pipeline requires **MediaPipe (`0.10.21`)** and **TensorFlow (`2.16.2`)**. Pre-compiled binary wheels are officially distributed and verified for Python 3.9–3.11. Modern operating systems defaulting to Python 3.12+ lack pre-compiled binary wheels for these packages and will fail during dependency installation.

---

## 2. Platform-Specific Setup

### A. macOS (Apple Silicon & Intel) — *Verified on Local Hardware*

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
```

#### 4. Camera Permissions on macOS
1. Open **System Settings** > **Privacy & Security** > **Camera**.
2. Verify your terminal emulator (Terminal, iTerm2, or VS Code) is toggled **ON**.
3. Camera capture automatically selects Apple Silicon native `cv2.CAP_AVFOUNDATION`.

---

### B. Linux (Ubuntu / Debian / Raspberry Pi OS 64-bit) — *Untested on Real Hardware*

#### 1. Install System Prerequisites & Audio Drivers
> [!NOTE]
> Linux environments require `espeak` for offline `pyttsx3` speech synthesis and `libasound2-dev` / `pulseaudio-utils` for audio device routing.
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
```

Camera capture utilizes Linux Video4Linux2 (`cv2.CAP_V4L2`), and TTS leverages `pyttsx3` (via eSpeak/speech-dispatcher) and `gTTS` with PulseAudio (`paplay`) / ALSA (`aplay`).

---

### C. Windows (Windows 10 / 11 64-bit) — *Untested on Real Hardware*

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
```

Camera capture utilizes Windows DirectShow (`cv2.CAP_DSHOW`) and Media Foundation (`cv2.CAP_MSMF`). Offline speech synthesis uses Microsoft SAPI5 voices via `pyttsx3` with automatic browser Web Speech API fallback.

---

### D. Docker Containerization (Optional) — *Untested on Live Daemon*
A production `Dockerfile` is provided in the root repository. It bundles Python 3.10-slim with OpenCV, MediaPipe, `espeak`, and `ffmpeg` libraries.
> [!NOTE]
> Docker build is currently **untested on a live daemon** in this environment as the host Docker service is inactive.
```bash
docker build -t isl-recognition .
docker run -p 5001:5001 --device=/dev/video0 isl-recognition
```

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
Ran 245 tests in ~20-30s
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

## 6. Subsystem Notes: Translation & Speech Fallbacks

### Translation Limitations & Failure Modes:
- **Online Service Rate Limits:** Real-time online translation uses the MyMemory API via `deep-translator`. Free-tier queries are subject to daily quota constraints (~1,000 words/day per IP). The engine applies exponential backoff retries on transient errors (HTTP 429, 502, 503).
- **Offline Coverage:** Offline translation is strictly limited to the curated dictionary (common phrases, greetings, family, objects) and rule-based compound phrase connector assembly.
- **What the User Sees on Failure:** If an unknown sentence is signed and online translation fails or is rate-limited, the system executes phonetic transliteration (e.g. converting Latin character phonetics into Devanagari, Bengali, Tamil, etc.). If transliteration is unavailable, the user sees the raw English text with an indicator toast. The application never crashes or outputs empty strings.

### Speech Synthesis Fallback Tiers:
1. **Tier 1 (macOS `say`):** Verified locally on macOS for English, Hindi, Marathi, Bengali, Tamil, Telugu, and Kannada via native system synthesis calls (`say -v <voice>`).
2. **Tier 2 (`pyttsx3`):** Offline SAPI5 (Windows) and `espeak` (Linux) where a matching language voice is installed. On macOS, languages lacking an authentic named native voice (**Gujarati, Punjabi, Malayalam**) explicitly bypass pyttsx3 and route directly to Tier 3 (gTTS) to avoid incomprehensible English-accented phonetic artifacts. Note: Windows and Linux paths are **untested on real hardware**.
3. **Tier 3 (`gTTS`):** Preferred primary tier for **Gujarati, Punjabi, and Malayalam** on macOS and robust online fallback for all Indic languages, producing natural regional speech via cloud audio streaming and local playback (`afplay` on macOS; Linux `paplay`/`aplay` and Windows `powershell` are **untested on real hardware**).
4. **Tier 4 (Browser Web Speech API):** Client-side speech synthesis used automatically in headless/containerized environments, offline fallback without native voices, or when local audio output devices are missing.

> [!IMPORTANT]  
> **Audible Correctness Notice:**  
> **Audible phonetic correctness is completely unverified until a human checks it.** Automated testing only verifies that the underlying system process exits with code 0 and non-empty audio files are created without raising runtime exceptions.

### Manual Audio Listening Verification Checklist
Before deploying or demonstrating regional speech synthesis, a human evaluator should complete the following auditory verification checklist:
- [ ] **English:** Audio plays cleanly via macOS `Rishi` (clear Indian-English phonetics).
- [ ] **Hindi:** Audio plays cleanly via macOS `Lekha` (natural Devanagari pronunciation).
- [ ] **Marathi:** Audio plays cleanly via macOS `Lekha` (Devanagari pronunciation).
- [ ] **Bengali:** Audio plays cleanly via macOS `Piya` (authentic Bengali phonology).
- [ ] **Tamil:** Audio plays cleanly via macOS `Vani` (authentic Tamil phonology).
- [ ] **Telugu:** Audio plays cleanly via macOS `Geeta` (authentic Telugu phonology).
- [ ] **Kannada:** Audio plays cleanly via macOS `Soumya` (authentic Kannada phonology).
- [ ] **Gujarati:** Verify speech plays via Tier 3 `gTTS` (`gTTS_gu`) or browser Web Speech; confirm native Gujarati phonetic pronunciation (no English phonetic distortion).
- [ ] **Malayalam:** Verify speech plays via Tier 3 `gTTS` (`gTTS_ml`) or browser Web Speech; confirm native Malayalam phonetic pronunciation.
- [ ] **Punjabi:** Verify speech plays via Tier 3 `gTTS` (`gTTS_pa`) or browser Web Speech; confirm native Gurmukhi phonetic pronunciation.
- [ ] **Volume & Latency:** Audio plays within 200 ms of phrase completion without clipping, stuttering, or speaker distortion.
