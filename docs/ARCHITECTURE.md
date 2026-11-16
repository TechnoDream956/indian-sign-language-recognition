# Indian Sign Language (ISL) Translator — System Architecture

## 1. System Architecture Overview

The system is engineered as an end-to-end multimodal perception and translation pipeline designed for real-time edge execution on desktop systems.

```text
               ┌───────────────────────────────┐
               │  FaceTime HD / USB Camera     │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │     MediaPipe Hands 3D        │
               │   (21 3D Left + 21 3D Right)  │
               └───────────────┬───────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 ┌──────────────────────┐              ┌───────────────────────────┐
 │ Geometric Coordinates│              │ Visual Appearance Crop    │
 │ (63D/144D Landmark)  │              │ (128x128 RGB Image)       │
 └──────────┬───────────┘              └─────────────┬─────────────┘
            ▼                                        ▼
 ┌──────────────────────┐              ┌───────────────────────────┐
 │ Landmark MLP Model   │              │ RGB Visual CNN Model      │
 │ (26-Class / 20-Class)│              │ (26-Class Sign Classifier)│
 └──────────┬───────────┘              └─────────────┬─────────────┘
            │                                        │
            └──────────────────┬─────────────────────┘
                               ▼
               ┌───────────────────────────────┐
               │ Confidence-Aware Fusion Engine│
               │ (Geometric Prior + Agreement) │
               └───────────────┬───────────────┘
                               ▼
               ┌───────────────────────────────┐
               │   WordBuffer State Machine    │
               │   (Debounce 0.8s, Space, Edit)│
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │ Continuous Learning Feedback  │
               │ (Quality Filter & SQLite DB)  │
               └───────────────┬───────────────┘
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│ Multilingual Translator │         │   Text-to-Speech (TTS)  │
│ (10 Regional Languages) │         │ (Web Speech API / say)  │
└────────────┬────────────┘         └────────────┬────────────┘
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
              ┌─────────────────────────────────┐
              │  Flask Local Desktop Backend    │
              │   • /video_feed (MJPEG Stream)  │
              │   • /events (SSE Telemetry)     │
              │   • Single-Instance Port 5001   │
              └────────────────┬────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │    Responsive Web Frontend      │
              │ (HTML5 / Vanilla CSS / ES6 JS)  │
              └─────────────────────────────────┘
```

---

## 2. Core Subsystems

### A. Computer Vision & Landmark Perception (`src/inference_engine.py` & `src/mediapipe_utils.py`)
- Employs Google MediaPipe Hands in a dedicated low-latency capture thread.
- Extracts spatial coordinates across 21 anatomical landmarks per hand (42 total 3D points).
- Computes normalized Euclidean coordinates relative to wrist and middle MCP hand scale.

### B. Multi-Modal Decision Fusion (`src/fusion_engine.py`)
- **Geometric Landmark Prior:** Fast evaluation on 63D single-hand and 144D dual-hand coordinate features.
- **Visual RGB Appearance:** 128x128 cropped bounding box evaluated by a dedicated CNN classifier.
- **Confidence-Aware Arbitration:** Dynamically arbitrates predictions between geometric topology and visual appearance, boosting confidence when both modalities agree.

### C. Alphabet Recognition & WordBuffer (`src/alphabet_engine.py`)
- Classifies hand shapes into letters A–Z using production neural checkpoints (`alphabet_model_26class.h5` and `cnn_sign_model.h5`).
- **Dual-Hand Support & Hand Gating:** Dispatches bimanual signs to the 144D dual-hand classifier (`best_val_model.h5`).
- **WordBuffer State Machine:** Manages debounce cooldowns (0.8s), character latching, word commitments, backspaces, and sentence assembly.

### D. Multilingual Translation & Speech Synthesis (`src/translator.py` & `src/tts_engine.py`)
- **Indic Translation:** Translates assembled English text into 10 regional Indian languages (Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, and English).
- **Text-to-Speech Engine:** Leverages browser Web Speech API for native Indic voices, with offline macOS `say` execution fallback.

### E. Continuous Learning & Privacy Subsystem (`src/feedback_learning/`)
- User-driven explicit sign correction interface.
- Multi-tier quality filtering and outlier rejection.
- Secure pseudonymized session tokens (HMAC-SHA256) and atomic SQLite persistence (`data/feedback/feedback.db`).

### F. Local Desktop Packaging (`packaging/`)
- Native double-clickable macOS bundle (`ISL Translator.app`) and terminal launcher (`Launch ISL Translator.command`).
- Single-instance safety mutex on port 5001 with health-check polling and automated browser launch.
