# Indian Sign Language (ISL) Translator — System Architecture

## 1. System Architecture Overview

The system is engineered as an end-to-end multimodal perception and translation pipeline designed for real-time edge execution on desktop systems.

```text
               ┌───────────────────────────────┐
               │  Web Camera / Video Stream    │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │     MediaPipe Holistic        │
               │  (33 Pose + 468 Face Mesh +   │
               │   21 Left Hand + 21 Right)    │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │ 1,662-Dim Coordinate Vector   │
               └───────────────┬───────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            │ (Mode: Word)                        │ (Mode: Alphabet)
            ▼                                     ▼
 ┌──────────────────────┐              ┌───────────────────────────┐
 │ 30-Frame FIFO Buffer │              │ Single-Frame Hand Vector  │
 └──────────┬───────────┘              │  (Left/Right Normalization)
            ▼                          └─────────────┬─────────────┘
 ┌──────────────────────┐                            ▼
 │   Dual MoP-GRU       │              ┌───────────────────────────┐
 │  Inference Engine    │              │ 26-Class Dense Classifier │
 │ (13-Class / 29-Class)│              └─────────────┬─────────────┘
 └──────────┬───────────┘                            ▼
            │                          ┌───────────────────────────┐
            │                          │  WordBuffer State Machine │
            │                          │ (Debounce, Space, Del)    │
            │                          └─────────────┬─────────────┘
            └──────────────────┬─────────────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │   English Token / Text  │
                  └────────────┬────────────┘
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│ Multilingual Translator │         │   Text-to-Speech (TTS)  │
│ (10 Regional Languages) │         │ (macOS say/pyttsx3/gTTS)│
└────────────┬────────────┘         └────────────┬────────────┘
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
              ┌─────────────────────────────────┐
              │  Flask Async Application Server │
              │   • /video_feed (MJPEG Stream)  │
              │   • /prediction_stream (SSE)    │
              │   • REST API Controls           │
              └────────────────┬────────────────┘
                               ▼
              ┌─────────────────────────────────┐
              │    Responsive Web Frontend      │
              │ (HTML5 / Vanilla CSS / ES6 JS)  │
              └─────────────────────────────────┘
```

---

## 2. Core Subsystems

### A. Computer Vision & Landmark Extraction (`src/mediapipe_utils.py`)
- Employs Google MediaPipe Holistic in a lightweight real-time configuration.
- Extracts spatial coordinates across 543 distinct anatomical landmarks per frame:
  - Pose landmarks: 33 points $\times$ 4 coordinates $(x, y, z, \text{visibility}) = 132$ features.
  - Face mesh landmarks: 468 points $\times$ 3 coordinates $(x, y, z) = 1,404$ features.
  - Left hand landmarks: 21 points $\times$ 3 coordinates $(x, y, z) = 63$ features.
  - Right hand landmarks: 21 points $\times$ 3 coordinates $(x, y, z) = 63$ features.
- Total feature vector: **1,662 floating-point values per frame**.
- Zero landmarks are returned as zeros when hands or faces are out of frame.

### B. Dynamic Word Recognition (`src/inference_engine.py`)
- Maintains a 30-frame FIFO sliding window ($30 \times 1,662$) capturing ~1.0–1.2 seconds of movement.
- **Dynamic Model Switching:**
  - `mopgru_model.h5`: 13 foundational vocabulary signs with `labelencoder.pkl`.
  - `mopgru_model_29class.h5`: 29 conversational vocabulary signs with `labelencoder_29class.pkl`.
- Incorporates a confidence threshold (default 0.70) and consecutive-frame stability counter to eliminate prediction flicker.

### C. Static Fingerspelling Recognition (`src/alphabet_engine.py`)
- Classifies single-frame static hand shapes into letters A–Z using a 26-class Dense neural network (`alphabet_model_26class.h5`).
- **Dual-Hand Support & Hand Gating:**
  - Supports tracking both hands simultaneously.
  - Classifies letters according to ISL standard reference:
    - Single-handed letters: `C`, `L`, `O`, `U`, `V`
    - Two-handed letters: `A`, `B`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `M`, `N`, `P`, `Q`, `R`, `S`, `T`, `W`, `X`, `Y`, `Z`
  - Validates hand count and suppresses erroneous single-handed predictions when a two-handed gesture is required.
- **WordBuffer State Machine:**
  - Implements debounce timing and character cooldowns (default 1.2s between duplicate letters).
  - Handles sentence composition with space insertion, character backspacing, and buffer clearing.

### D. Translation & Speech Synthesis (`src/translator.py` & `src/tts_engine.py`)
- **Multilingual Translator:** Translates recognized English words/sentences into 9 regional Indian languages: Hindi (`hi`), Bengali (`bn`), Tamil (`ta`), Telugu (`te`), Marathi (`mr`), Gujarati (`gu`), Kannada (`kn`), Malayalam (`ml`), and Punjabi (`pa`), as well as English (`en`).
- **Text-to-Speech Engine:** Multi-backend synthesizer prioritizing native low-latency macOS `say` subprocesses, falling back to `pyttsx3` and `gTTS`.

### E. Web Server & Transport (`src/app.py`)
- Multi-threaded Flask application serving:
  - `/video_feed`: Real-time annotated MJPEG video stream with bounding boxes and landmark HUD.
  - `/prediction_stream`: Server-Sent Events (SSE) streaming JSON tokens, probabilities, translations, and buffer states with sub-50ms latency.
  - REST endpoints for switching modes, hot-swapping models, selecting target languages, and mutating the alphabet WordBuffer.

### F. Standalone Packaging (`packaging/`)
- Packaged using PyInstaller for standalone macOS deployment without requiring external terminal or Python environments.
- Self-contained launcher (`packaging/launcher.py`) coordinates background Flask server initialization and native browser opening.
