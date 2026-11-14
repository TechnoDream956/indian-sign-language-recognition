# User Guide — Indian Sign Language Translator

This document provides complete instructions on how to use all features of the **Indian Sign Language (ISL) Recognition & Translation Web Application** and auxiliary tools.

---

## 1. Quick Start

1. Ensure the virtual environment is activated:
   ```bash
   source venv/bin/activate
   ```
2. Start the web application:
   ```bash
   python src/app.py
   ```
3. Open your browser to **`http://127.0.0.1:5000`**.
4. Allow browser camera permissions when prompted. The live webcam feed will appear with the neural inference overlay.

---

## 2. Dashboard Layout & Controls

The interface is divided into an intuitive, responsive control dashboard:

```text
┌───────────────────────────────────────────────┬───────────────────────────────────┐
│              LIVE VIDEO FEED                  │          CONTROL PANEL            │
│  - Real-time MediaPipe Skeleton Tracking      ├───────────────────────────────────┤
│  - Hand Bounding Boxes (Left / Right)         │ Mode: [ Word Mode ] [ Alphabet ]  │
│  - On-Screen HUD (Confidence, Stability, FPS) │ Model: (•) 13-Class  ( ) 29-Class  │
│                                               │ Language: [ Hindi (hi)         ▼] │
│                                               │ Audio: [🔊 TTS Enabled]           │
├───────────────────────────────────────────────┼───────────────────────────────────┤
│          REAL-TIME PREDICTION                 │      MULTILINGUAL TRANSLATION     │
│  Predicted: "Hello"   Confidence: 94.2%       │  Hindi: "नमस्ते"                  │
│  [████████████████████░░░░░░]                 │  Audio Synthesis: Auto-spoken     │
├───────────────────────────────────────────────┴───────────────────────────────────┤
│                  ALPHABET WORD BUFFER (Active in Alphabet Mode)                  │
│  Buffer Text: "I S L"                                                             │
│  Actions: [ ␣ Space ]  [ ⌫ Backspace ]  [ 🗑️ Clear Buffer ]  [ 📢 Speak Sentence ] │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Word Mode (Dynamic Sequence Recognition)

Word Mode utilizes a **30-frame sliding window buffer** with the recurrent **MoP-GRU** neural network to recognize continuous gestures.

### How to Perform a Word Gesture:
1. **Start in Neutral Position:** Keep hands resting at chest level or out of view for 0.5 seconds.
2. **Execute the Gesture:** Move hands smoothly through the signing motion. The system buffers the last 30 frames (~1.0–1.2 seconds at 25–30 FPS).
3. **Hold Finish Briefly:** Complete the sign. Once the confidence crosses the threshold (0.70) and remains stable for consecutive frames, the prediction triggers.
4. **Return to Neutral:** Drop hands briefly before starting the next sign to allow the buffer to reset cleanly.

### Model Selection:
- **13-Class Foundational Model (`mopgru_model.h5`):**
  - Vocabulary: `Hello`, `Thank You`, `Yes`, `No`, `Please`, `Help`, `Good`, `Bad`, `Food`, `Water`, `Father`, `Mother`, `Friend`.
  - Optimized for high-confidence core interactions.
- **29-Class Extended Model (`mopgru_model_29class.h5`):**
  - Expands the lexicon to 29 conversational signs, including emergency queries and daily needs.
  - Can be hot-swapped instantly in the UI with zero server restart.

---

## 4. Alphabet Mode (Static Fingerspelling & Sentence Composition)

Alphabet Mode classifies static hand shapes frame-by-frame using a 26-class Dense network, feeding into an intelligent **WordBuffer state machine**.

### Single-Hand vs. Dual-Hand Fingerspelling:
Indian Sign Language uses both single-handed and dual-handed configurations:
- **Single-Handed Letters:** `C`, `L`, `O`, `U`, `V`
- **Two-Handed Letters:** `A`, `B`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `M`, `N`, `P`, `Q`, `R`, `S`, `T`, `W`, `X`, `Y`, `Z`

### WordBuffer State Machine Features:
- **Debounce & Cooldown (1.2s):** When holding a sign (e.g., `A`), it will only be appended once. You must release or hold for the cooldown duration before the same letter can trigger again.
- **Immediate Cross-Letter Transition:** Switching directly from `A` to `B` registers immediately without artificial cooldown delays.
- **Word Composition:**
  - Click **Space** to complete a word.
  - Click **Backspace** to remove the last typed character.
  - Click **Clear** to reset the buffer.
  - Click **Speak Sentence** to synthesize and translate the entire constructed sentence.

---

## 5. Multilingual Translation & Text-to-Speech

### Supported Languages:
The system automatically translates predicted words and sentences into 10 regional Indian languages via neural machine translation:
1. **Hindi (`hi`)** — हिन्दी
2. **Bengali (`bn`)** — বাংলা
3. **Tamil (`ta`)** — தமிழ்
4. **Telugu (`te`)** — తెలుగు
5. **Marathi (`mr`)** — मराठी
6. **Gujarati (`gu`)** — ગુજરાતી
7. **Kannada (`kn`)** — ಕನ್ನಡ
8. **Malayalam (`ml`)** — മലയാളം
9. **Punjabi (`pa`)** — ਪੰਜਾਬੀ
10. **English (`en`)** — Original tokens

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
