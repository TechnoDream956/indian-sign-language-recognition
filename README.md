<h1 align="center">   
  Indian Sign Language (ISL) Recognition & Translation System
</h1>

<p align="center">
  <b>Real-Time Neural Sign Language Perception, Stateful WordBuffer Composition, Indic Translation, and Voice Synthesis</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-Hands-00C7B7?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.0%20Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Docker-lightgrey?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Vercel-Frontend%20Ready-000000?style=for-the-badge&logo=vercel&logoColor=white" />
  <img src="https://img.shields.io/badge/Tests-106%20Passing-brightgreen?style=for-the-badge" />
</p>

---

## 📌 Project Overview

This repository provides an end-to-end, edge-deployable **Indian Sign Language (ISL) Alphabet Recognition and Multilingual Translation Platform**. Designed for consumer hardware running a standard webcam, the system extracts high-fidelity 3D skeletal landmarks using **Google MediaPipe**, classifies signs through neural architectures, manages fingerspelled sentence composition via a stateful debounce machine, translates sentences into **10 regional Indian languages**, and vocalizes translations with native low-latency speech synthesis.

The platform includes both an asynchronous **Flask web application** featuring a high-contrast **Midnight Teal accessibility UI** and a native **macOS application bundle (`ISL Translator.app`)**.

---

## 🏗️ System Architecture & Model Tiers

To preserve scientific rigor, this repository strictly separates the **active live production system** from **controlled experimental research models**:

```
                                  [ Live Webcam Input ]
                                            │
                                            ▼
                               [ Google MediaPipe Hands ]
                           (3D Skeletal Landmark Tracking)
                                            │
              ┌─────────────────────────────┴─────────────────────────────┐
              ▼                                                           ▼
    ┌───────────────────────────────────┐               ┌───────────────────────────────────┐
    │     LIVE PRODUCTION SYSTEM        │               │   EXPERIMENTAL RESEARCH MODELS    │
    │  AlphabetEngine (Single-Hand)     │               │   (Spatial & Aperture Modeling)   │
    │  Model: alphabet_model_26class.h5 │               │  • Phase 5B.1 Config C (132D)     │
    │  Input: 63D (Wrist-Normalized)    │               │  • Phase 5A Two-Stream (129D)     │
    │  Classes: 26 (A through Z)        │               │  • Proposed Temporal BiGRU (42D)  │
    └─────────────────┬─────────────────┘               └───────────────────────────────────┘
                      │ (Live Predictions)
                      ▼
         [ WordBuffer State Machine ]
         (1.2s Debounce · Cooldown · Gating)
                      │
                      ▼
            [ Fingerspelled Token ]
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
 [ Neural Translation ]    [ Audio Speech Synthesis ]
 (10 Indic Languages)      (macOS say / pyttsx3)
         │                         │
         └────────────┬────────────┘
                      ▼
        [ Asynchronous Flask 3.0 Server ]
        (Port 5001 · MJPEG Stream · SSE Stream)
                      │
                      ▼
      [ Midnight Teal Responsive Dashboard ]
      (Live Telemetry HUD · Accessible Badges)
```

### Model Classification Matrix

| Model Tier | Model Checkpoint Path | Architecture | Input Feature Vector | Classes | Status | Performance Metrics |
|---|---|---|---|:---:|:---:|---|
| **Production Baseline (Live)** | `models/alphabet_model_26class.h5` | Feedforward Sequential Dense MLP | **63D** ($21 \times 3$ wrist-normalized coords) | **26** (A–Z) | **Active in Live App** | **73.72%** Top-1 Acc, **85.89%** Top-3 Acc, **89.46%** Accepted Acc ($\tau \ge 0.75$) on 411 benchmark samples |
| **Experimental: Latest Trained** | `models/experimental/phase5b1_features/Exp_C_Aperture132D_seed42.h5` | Two-Stream MLP with Thumb-Index Apertures | **132D** (126D shape + 3D relative wrist $\Delta \mathbf{P}_{\text{wrist}}$ + 3D apertures) | **21** (Bimanual) | **Validated Research** | **90.09%** Val Acc, **82.04%** Val Macro F1, **78.50%** Holdout Acc; resolves $Q \leftrightarrow B$ collision (Letter Q F1: **90.9%**, Letter B Prec: **100.0%**) |
| **Experimental: Previous Trained** | `models/experimental/phase5a_two_stream/Exp_C_TwoStream_NoSpatialAug_seed42.h5` | Two-Stream Spatial-Aware MLP | **129D** (126D shape + 3D relative wrist $\Delta \mathbf{P}_{\text{wrist}}$) | **21** (Bimanual) | **Validated Research** | **89.21%** Val Acc, **87.38%** Holdout Acc; restored 100% RealSign Q recall; McNemar test $p < 0.001$ |
| **Proposed Architecture** | Proposed in Phase 5C.1–5C.3 | Bidirectional GRU (BiGRU) Motion Sequence Model | **$30 \times 42\text{D}$** (30 frames motion trajectory) | **15** (Dynamic) | **Untrained Proposal** | Data acquisition protocol designed (Signer 03 partitioned, Signer 04 planned). Model **not yet trained**. |
| **Historical Retired Models** | `models/retired_word_models/` | MoP-GRU (13-Class & 29-Class) | $30 \times 1662\text{D}$ Holistic Vector | 13 / 29 | **Safely Retired** | Decommissioned from live app to isolate Alphabet & Fingerspelling Mode |

> [!NOTE]
> The live production application exclusively runs the **26-class single-hand model** (`models/alphabet_model_26class.h5`). The 132D Config C and 129D Two-Stream checkpoints are experimental research prototypes and are **not** integrated into the live webcam inference path.

---

## ✨ Core Features

- **Real-Time 3D Landmark Perception:**
  - High-performance tracking via `mediapipe.solutions.hands` operating at 20–30+ FPS on consumer CPUs.
  - Wrist-relative translation and coordinate normalization ensuring scale and position invariance.
- **Stateful WordBuffer State Machine:**
  - Configurable debounce cooldown (1.2s) preventing duplicate character insertions while holding a pose.
  - Interactive composition controls: space insertion (`Space`), backspace (`⌫`), word clear (`C`), sentence clear (`X`), and speech triggering.
- **Multilingual Indic Translation (10 Languages):**
  - Instant machine translation into **Hindi (`hi`)**, **Tamil (`ta`)**, **Telugu (`te`)**, **Marathi (`mr`)**, **Bengali (`bn`)**, **Gujarati (`gu`)**, **Kannada (`kn`)**, **Malayalam (`ml`)**, **Punjabi (`pa`)**, and **Urdu (`ur`)** alongside **English (`en`)**.
- **Low-Latency Speech Synthesis (TTS):**
  - Native macOS `say` subprocess routing with designated voices (e.g., `Lekha` for Hindi/Marathi, `Rishi` for Indian English) and fallback mechanisms to `pyttsx3` and `gTTS`.
  - Non-blocking audio synthesis with overlap prevention and busy gating.
- **Midnight Teal Professional UI:**
  - Dark-mode accessibility workstation aesthetic (`#101820` background, `#35c6b0` teal accent signals).
  - Annotated MJPEG video stream (`/video_feed`) with landmark skeleton overlays.
  - Sub-50ms Server-Sent Events (SSE) telemetry (`/events`) powering live confidence bars and stability indicators.
- **Native macOS Application (`ISL Translator.app`):**
  - Double-clickable desktop bundle configured with a dynamic relative-path launcher that runs the workspace virtual environment on port 5001 with zero asset caching delays.
- **Extensive Test Coverage:**
  - **153 automated unit, integration, and regression tests** passing with a 100% success rate (`unittest`).

---

## 🚀 Installation & Setup

### Prerequisites
- **Operating System:** macOS (macOS 12.0+ Monterey, Ventura, Sonoma, Sequoia on Apple Silicon or Intel) or Linux (Ubuntu 20.04+).
- **Python Version:** Python 3.10.x (Recommended: Python 3.10.12 to 3.10.21).
- **Webcam:** Integrated FaceTime HD camera or external USB/UVC webcam.

### 1. Clone the Repository
```bash
git clone https://github.com/TechnoDream956/indian-sign-language-recognition.git
cd indian-sign-language-recognition
```

### 2. Configure Virtual Environment
```bash
# Create Python 3.10 virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and packaging tools
pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Launching the Application

### Method 1: Native macOS Application Bundle (Recommended for Mac)
Double-click the packaged application in Finder:
```bash
open "dist/ISL Translator.app"
```
*The launcher automatically detects the workspace environment, binds port 5001, opens your default browser at `http://127.0.0.1:5001`, and initiates clean camera and process shutdown when closed.*

### Method 2: Python Command Line
```bash
# Activate virtual environment
source venv/bin/activate

# Launch via packaging runner
python packaging/launcher.py

# Alternatively, launch Flask directly
python src/app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5001`**

---

## 🧪 Automated Testing

Execute the comprehensive regression and unit test suite:
```bash
# Run all 153 tests
./venv/bin/python -m unittest discover -s tests

# Run specific test modules
./venv/bin/python -m unittest tests/test_alphabet_engine.py
./venv/bin/python -m unittest tests/test_alphabet_translation.py
./venv/bin/python -m unittest tests/test_phase5b1_features.py
./venv/bin/python -m unittest tests/test_phase5c_temporal.py
```

---

## 📁 Repository Structure

```text
Indian-Sign-Language-Recognition/
├── assets/                          # Architecture diagrams and UI previews
│   └── mopgru_pipeline.png
├── dist/                            # macOS Application Bundles
│   └── ISL Translator.app           # Native double-clickable macOS bundle
├── docs/                            # Comprehensive engineering and academic docs
│   ├── ARCHITECTURE.md              # System design and pipeline data flows
│   ├── CODEBASE_GUIDE.md            # Beginner-friendly guide to all modules
│   ├── DATASETS.md                  # Dataset provenance, schemas, and manifests
│   ├── INSTALLATION.md              # Detailed environment setup instructions
│   ├── PRESENTATION_GUIDE.md        # University presentation and viva prep guide
│   ├── USER_GUIDE.md                # Gesture reference and shortcut manual
│   └── reports/                     # Historical engineering reports (Phases 1–5C.3)
├── models/                          # Trained model checkpoints and mappings
│   ├── alphabet_label_map.json      # 26-class alphabet character mapping
│   ├── alphabet_model_26class.h5    # Active production 63D single-hand model
│   ├── experimental/                # Controlled research checkpoints (Phases 4, 5A, 5B.1)
│   │   ├── phase5a_two_stream/      # 129D Two-Stream spatial models
│   │   └── phase5b1_features/       # 132D Aperture-augmented models (Exp C)
│   └── retired_word_models/         # Quarantined MoP-GRU word sequence models
├── packaging/                       # Native macOS desktop packaging harness
│   ├── ISL_Translator.spec          # PyInstaller build specification
│   ├── build_app.sh                 # PyInstaller standalone build script
│   ├── launcher.py                  # Single-instance lifecycle launcher
│   └── mac_app_launcher.sh          # Dynamic relative-path app bundle wrapper
├── src/                             # Core application source code
│   ├── alphabet_engine.py           # 26-class inference and WordBuffer state machine
│   ├── app.py                       # Asynchronous Flask web server and SSE streams
│   ├── inference_engine.py          # Background camera loop and MediaPipe pipeline
│   ├── mediapipe_utils.py           # Landmark extraction and normalization helpers
│   ├── translator.py                # 10 Indic languages neural machine translation
│   ├── tts_engine.py                # Native macOS say / pyttsx3 speech synthesizer
│   ├── static/                      # Frontend assets (style.css, app.js)
│   └── templates/                   # Frontend HTML templates (index.html)
├── tests/                           # Automated test suite (153 tests passing)
│   ├── test_alphabet_engine.py      # Unit tests for AlphabetEngine and debounce
│   ├── test_alphabet_translation.py # Translation and Indic voice tests
│   ├── test_phase5b1_features.py    # Unit tests for aperture feature extractors
│   └── test_phase5c_temporal.py     # Unit tests for temporal collector and quotas
├── .gitignore                       # Exclusion rules (caches, virtualenvs, builds)
├── LICENSE                          # MIT License
├── README.md                        # Master repository documentation
├── requirements.txt                 # Pinned Python package dependencies
└── THIRD_PARTY_NOTICES.md           # Open-source attributions and licenses
```

---

## ⚠️ Technical Considerations & Limitations

1. **Ambient Lighting:** MediaPipe landmark detection requires uniform, diffuse lighting. Harsh backlighting or dim environments may cause tracking loss or jitter.
2. **Camera Positioning:** Optimal recognition occurs when seated 0.8m to 1.5m from the camera with hands clearly visible against a plain background.
3. **Single-Hand Production Scope:** The live application is optimized for single-hand gestures (letters C, L, O, U, V and standard single-hand fingerspelling). Bimanual gestures (A, B, D, etc.) are being researched in experimental tracks (Phase 5B.1 / Phase 5C) and are not yet in production.
4. **Internet Connectivity for Translation:** Regional Indic translation uses the neural translation API and requires an active internet connection; if offline, the system safely falls back to English tokens.

---

## 📚 Scientific Acknowledgements

This project builds upon the foundational research in MediaPipe-based sign language recognition:

```bibtex
@article{subramanian2022mopgru,
  title   = {An integrated mediapipe-optimized GRU model for Indian sign language recognition},
  author  = {Subramanian, Barathi and Olimov, Bekhzod and Naik, Shraddha M. and Kim, Sangchul and Park, Kil-Houm and Kim, Jeonghong},
  journal = {Scientific Reports},
  volume  = {12},
  number  = {11964},
  year    = {2022},
  publisher = {Nature Portfolio},
  doi     = {10.1038/s41598-022-15998-7}
}
```

We acknowledge:
- **Google MediaPipe** for robust real-time skeletal hand perception.
- **Indian Sign Language Research and Training Centre (ISLRTC)** for official sign definitions.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Third-party dependencies and notices are detailed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
