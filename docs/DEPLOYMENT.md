# Deployment & Application Guide — Indian Sign Language Recognition & Translation

This document explains the runtime deployment and packaging architecture for the **Indian Sign Language Recognition & Multilingual Voice Translator**.

---

## 1. Local Desktop Application (macOS Primary)

The primary target platform is a self-contained local desktop application on macOS (Apple Silicon M1/M2/M3/M4 and Intel):

```
┌────────────────────────────────────────────────────────┐
│               ISL Translator.app                       │
│     (macOS Native Application Bundle / .command)       │
│  - Double-click launch with zero Terminal commands     │
│  - Single-instance safety mutex (Port 5001 auto-bind)  │
│  - Health-check polling with automatic browser launch  │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ Starts background process
                            ▼
┌────────────────────────────────────────────────────────┐
│              Local Flask & Neural Backend              │
│  - MediaPipe Hands 3D Coordinate Perception            │
│  - Multi-Modal Decision Fusion (MLP 63D + RGB CNN)     │
│  - WordBuffer Debounce & Sentence Composition          │
│  - Multilingual Translation & Native TTS Voice         │
└────────────────────────────────────────────────────────┘
```

### Launch Methods:
1. **`ISL Translator.app`**: Double-click directly in Finder.
2. **`Launch ISL Translator.command`**: Double-click Terminal script with live logging.
3. **`python packaging/launcher.py`**: Direct Python execution.

---

## 2. Docker Container Deployment

For containerized server environments:

```bash
# Build the container image
docker build -t isl-translator:latest .

# Run the container mapping port 5001
docker run -d -p 5001:5001 --name isl-app isl-translator:latest
```

Health check endpoint: `http://localhost:5001/api/status`

---

## 3. Connecting Frontend to Custom Backend

1. In the top-right header of the web application, click the **⚙️ Settings** button.
2. Enter your backend server's URL (e.g. `http://127.0.0.1:5001`).
3. The application will store the URL in local browser storage and connect all vision, speech, and translation streams automatically!
