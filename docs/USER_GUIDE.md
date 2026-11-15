# User Guide — Indian Sign Language Translator

This document provides complete instructions on how to use all features of the **Indian Sign Language (ISL) Recognition & Translation Desktop Application** on macOS and Linux.

---

## 1. Quick Start

### 1-Click Launch (macOS)
1. Double-click **`ISL Translator.app`** (or **`Launch ISL Translator.command`**) in the project folder.
2. The application will start the local server, verify all models, and automatically open your default browser at:
   **`http://127.0.0.1:5001`**
3. Grant camera permissions if prompted by macOS.
4. The live video feed will appear with the neural landmark tracking overlay.

### Developer Command-Line Launch
```bash
source venv/bin/activate
python packaging/launcher.py
```

---

## 2. Dashboard Layout & Controls

The interface is structured as an interactive, low-latency desktop control center:

```text
┌───────────────────────────────────────────────┬───────────────────────────────────┐
│              LIVE VISION STREAM               │      ACTIVE PREDICTION & HUD      │
│  - Real-time MediaPipe Hand Skeleton Overlay  ├───────────────────────────────────┤
│  - 21 3D Normalized Coordinates               │ Model: Decision Fusion (MLP+CNN)  │
│  - Real-Time In-Viewport HUD (FPS, Latency)   │ Prediction: "A"  Confidence: 94%  │
│  - STOP / START Camera Toggle (State Machine) │ Target Language: Hindi (हिन्दी)  │
├───────────────────────────────────────────────┼───────────────────────────────────┤
│          WORD BUFFER COMPOSER                 │      MULTILINGUAL TRANSLATION     │
│  Assembled Text: "H E L L O"                  │  Hindi: "नमस्ते"                  │
│  Actions: [ ␣ Space ] [ ⌫ Backspace ]         │  TTS Voice: Native Indic Speech   │
│  [ 🗑️ Clear All ] [ 📢 Speak Sentence ]       │  Actions: [ 📢 Speak ] [ 📋 Copy ]│
├───────────────────────────────────────────────┴───────────────────────────────────┤
│                  CONTINUOUS FEEDBACK & CORRECTION SYSTEM                          │
│  Submit corrections to improve personal recognition adapters with SQLite storage. │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Real-Time Recognition & Decision Fusion

The system employs **Multi-Modal Decision Fusion** combining geometric coordinate analysis and visual appearance:

1. **Geometric Coordinate Prior (Landmark MLP):**
   - 21 normalized 3D hand keypoints evaluated at sub-millisecond speeds.
   - Evaluates finger angles, joint distances, and spatial orientation.
2. **Visual Feature Classifier (RGB CNN):**
   - 128x128 cropped bounding box evaluated concurrently.
   - Disambiguates visually similar signs (e.g. `M` vs `N`, `U` vs `V`).
3. **Arbitration Logic:**
   - When Landmark confidence is high (&ge; 0.70) or models agree, predictions commit with an agreement bonus.
   - Automatically handles single-handed (`C`, `L`, `O`, `U`, `V`) and dual-handed (`A`, `B`, `D`, `E`, `F`, `G`, `H`, `I`, etc.) ISL signs.

---

## 4. WordBuffer & Sentence Composition

As continuous signs are performed, the **WordBuffer state machine** manages debounce, latching, and sentence assembly:

- **Debounce & Cooldown (0.8s):** Holding a letter sign emits the letter once, preventing unwanted repetition.
- **Immediate Cross-Letter Transition:** Changing to a different letter transitions immediately without artificial delays.
- **Editing Actions:**
  - **Space (␣):** Commits the current word and appends a space to the sentence.
  - **Backspace (⌫):** Deletes the last character.
  - **Commit Sign (+):** Manually locks in the active letter candidate.
  - **Clear All:** Clears both the active word and the sentence buffer.
  - **Speak Sentence:** Synthesizes the constructed sentence in English or the selected Indic language.

---

## 5. Multilingual Translation & Text-to-Speech

The system translates English sentences into 10 regional Indian languages in real time:

1. **Hindi (`hi`)** — हिन्दी
2. **Bengali (`bn`)** — বাংলা
3. **Tamil (`ta`)** — தமிழ்
4. **Telugu (`te`)** — తెలుగు
5. **Marathi (`mr`)** — मराठी
6. **Gujarati (`gu`)** — ગુજરાતી
7. **Kannada (`kn`)** — ಕನ್ನಡ
8. **Malayalam (`ml`)** — മലയാളം
9. **Punjabi (`pa`)** — ਪੰਜਾਬੀ
10. **English (`en`)** — Source text

### Speech Synthesis:
- Uses the browser's hardware-accelerated **Web Speech API** with automatic Indic voice matching.
- Native macOS `say` subprocess acts as an offline desktop fallback.

---

## 6. Feedback Learning & Correction

If the system mispredicts a letter:
1. Click **Correct Prediction** in the interface (or press shortcut `C`).
2. Select the intended correct letter (A–Z).
3. The correction is securely validated by quality filters and stored in the local SQLite database (`data/feedback/feedback.db`) to refine personal model weights.

---

## 7. Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| **Space** | Commit word and append space |
| **Backspace** | Delete last character |
| **Enter** | Speak current sentence |
| **C** | Open Correction Modal |
| **P** | Toggle Camera Stream (Start / Stop) |
| **M** | Mute / Unmute Speech Output |
| **Esc** | Close any open modal |

---

## 8. Quitting the Application

To safely close the application:
1. Click the **🚪 QUIT APPLICATION** button in the header.
2. Confirm the dialog. The server releases the camera hardware, halts neural workers, disconnects SSE listeners, and displays the exit confirmation screen. Alternatively, press `Ctrl+C` in the terminal launcher.

### Text-to-Speech (TTS):
- On macOS, the engine utilizes the native macOS system speech synthesizer (`say`) for instant, sub-50ms offline pronunciation.
- Fallback backends include `pyttsx3` and Google Text-to-Speech (`gTTS`).
- TTS can be toggled on or muted directly from the navigation bar.

---

## 6. Solo Dual-Hand Alphabet Collection Utility

For research, dataset expansion, or solo practice, the repository includes a self-contained collector:

```bash
python src/collect_dual_hand_alphabet.py --solo --samples-per-label 5 --output-dir data/phase5p_dual_hand
```

### Features:
1. **Hands-Free Solo Mode:** Detects both hands automatically and validates hand count against the letter's expected configuration.
2. **Stability Gating:** Requires the gesture to be held stably for 5 consecutive frames before capturing.
3. **Automatic Progress Tracking:** Saves the sample, displays on-screen confirmation (e.g., `SAVED: A (5/5)`), and initiates a **5-second transition countdown** before advancing to the next letter.
4. **Release Period Enforcement:** Requires the user to release the hand pose between captures to prevent duplicate burst captures.

To verify and audit collected datasets:
```bash
python src/audit_dual_hand_dataset.py --data-dir data/phase5p_dual_hand
```
