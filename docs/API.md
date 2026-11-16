# Indian Sign Language Recognition — API Reference

The Flask backend provides a clean RESTful API and Server-Sent Events (SSE) telemetry stream for real-time sign language recognition, buffer manipulation, translation, and speech synthesis.

## Base URL
- Local Default: `http://127.0.0.1:5001`
- Docker Default: `http://0.0.0.0:5001`

---

## Core Endpoints

### 1. System Health Check
```http
GET /api/healthz
```
**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "ready": true,
  "camera_mode": "cloud_browser",
  "cloud_mode": true,
  "model_ok": true
}
```

---

### 2. Frame Ingestion & Prediction (Cloud / Web Mode)
```http
POST /api/camera/frame
Content-Type: image/jpeg
```
*Body: Raw binary JPEG bytes from client webcam.*

**Response (`200 OK`):**
```json
{
  "ok": true,
  "frame_id": 412,
  "processing_time_ms": 35.6,
  "hand_detected": true,
  "hand_count": 1,
  "prediction": "A",
  "confidence": 0.94,
  "fused_prediction": "A",
  "fused_confidence": 0.94,
  "fusion_mode": "LANDMARK_DOMINANT",
  "last_stable_letter": "A",
  "stability_progress": 1.0,
  "landmarks": [
    {
      "handedness": "Right",
      "score": 0.98,
      "points": [
        [0.542, 0.781, -0.012],
        [0.489, 0.742, -0.025]
      ]
    }
  ],
  "word": "HELLO",
  "sentence": "HELLO ",
  "translation": "नमस्ते"
}
```

---

### 3. Multilingual Translation
```http
POST /api/translate
Content-Type: application/json
```
**Request Body:**
```json
{
  "text": "THANK YOU",
  "language": "Tamil"
}
```
**Response (`200 OK`):**
```json
{
  "ok": true,
  "source": "THANK YOU",
  "language": "Tamil",
  "translation": "நன்றி"
}
```

---

### 4. Manual Sign Commitment
```http
POST /api/alphabet/commit-letter
Content-Type: application/json
```
**Request Body:**
```json
{
  "letter": "A"
}
```
**Response (`200 OK`):**
```json
{
  "ok": true,
  "letter": "A",
  "current_word": "BA",
  "sentence": ""
}
```

---

### 5. WordBuffer Controls & SSE Stream

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/alphabet/space` | `POST` | Commits active word, inserts space, and triggers translation. |
| `/api/alphabet/backspace` | `POST` | Deletes the last character in the active word buffer. |
| `/api/alphabet/commit` | `POST` | Commits current word buffer into accumulated sentence. |
| `/api/alphabet/clear-word` | `POST` | Clears active word buffer without modifying sentence. |
| `/api/alphabet/clear-all` | `POST` | Clears both active word and accumulated sentence. |
| `/api/alphabet/reset` | `POST` | Resets the hold-to-emit debounce state machine. |
| `/events` | `GET` | SSE stream emitting continuous JSON telemetry packets at ~12 Hz. |

---

### 6. Text-to-Speech (TTS) Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/tts/voices` | `GET` | Returns available voices across macOS say, pyttsx3, and gTTS. |
| `/api/action/speak` | `POST` | Synthesizes current buffer text into speech using the 4-tier fallback chain. |
