# Repository Metadata & Demo Recording Protocol

**Repository:** `https://github.com/TechnoDream956/indian-sign-language-recognition`  
**Application Title:** Indian Sign Language (ISL) Recognition & Translation Desktop System  
**Version:** `2.6.0-prod`  
**License:** MIT License  
**Academic Institution:** School of Computer Science Engineering and Technology, Bennett University  
**Project Mentor:** Dr. Reshu Chaudhary  

---

## 1. Project Team & Responsibilities (Strict Attribution)

| Contributor | Project Team Role | Key Subsystem Responsibilities |
| :--- | :--- | :--- |
| **Arpit** | **Recognition Model Development** | 63D Single-Hand MLP, 144D Dual-Hand MLP architecture, ML diagnosis, and model training. |
| **Sachin** | **Data Collection, Preprocessing & Backend Support** | Dataset extraction, landmark coordinate normalization, pipeline profiling, and backend engineering. |
| **Abhu** | **Multilingual Support & TTS** | 10-language translation dictionary, MyMemory API integration, and multi-tier TTS fallback chain. |
| **Yash** | **Frontend & Deployment** | Local desktop overhaul, macOS application bundle (`ISL Translator.app`), and responsive dark UI. |

---

## 2. Technical Metadata Summary

- **Primary Perception:** Google MediaPipe Hands (21 3D landmarks per hand, up to 2 hands)
- **Neural Model Architectures:**
  - *Single-Hand Baseline:* 63D Landmark Dense MLP (26 classes, `alphabet_model_26class.h5`)
  - *Dual-Hand Classifier:* 144D Spatial Invariant MLP (20 classes, `best_val_model.h5`)
  - *Visual Appearance Model:* $128\times128\times3$ RGB CNN (26 classes, `cnn_sign_model.h5`)
  - *Decision Engine:* Confidence-aware rule-based decision-level fusion
- **Offline Linguistic Lexicon:** 10 Indian Regional Languages + English
- **Speech Synthesis:** 4-Tier Fallback (macOS `say` $\to$ `pyttsx3` $\to$ `gTTS` $\to$ Browser Web Speech)
- **Primary Execution Environment:** macOS (Apple Silicon M1–M4 & Intel x86_64) via `ISL Translator.app`
- **Automated Code Tests:** 245 passing tests across 19 suites

---

## 3. Honest Live Demo Recording Protocol (Zero-Fakery Protocol)

> [!IMPORTANT]
> **Strict Demonstration Ethics:** Never generate mock synthetic demos, pre-recorded video injections, or fake telemetry overlays. Record authentic, unedited webcam sessions showing real live model behavior, including known failure modes and stability thresholds.

### Equipment & Lighting Setup
1. **Camera Position:** Position webcam at eye level, roughly 0.6–1.0 meter away, framing upper chest, shoulders, and both hands within the 640×480 frame.
2. **Lighting:** Use diffuse, uniform front lighting. Avoid strong backlights (e.g. windows behind the signer) that cause hand silhouettes.
3. **Background:** Use a relatively uncluttered, neutral background for optimal MediaPipe hand segmentation.

### Step-by-Step Recording Sequence (macOS QuickTime)
1. **Start Screen Recording:**
   - Open **QuickTime Player** > **File** > **New Screen Recording**.
   - Select the display area covering the application browser window (`http://127.0.0.1:5001`).
   - In QuickTime recording options, ensure "MacBook Microphone" or system audio capture is selected so speech audio is audible.
2. **Launch Application:**
   - Double-click **`ISL Translator.app`** on your desktop or project root.
   - Show the browser automatically opening to the local dashboard with `CAMERA: OK` and `MODEL: OK` telemetry signals.
3. **Demonstrate Single-Handed Signs:**
   - Perform unimanual letters: `C`, `L`, `O`, `V`.
   - Hold each sign steady for ~1 second to demonstrate the hold-to-emit stability progress bar (green indicator reaching 100%).
   - Observe character emission into the WordBuffer without duplicate stutter.
4. **Demonstrate Two-Handed Signs:**
   - Perform bimanual letters: `A`, `B`, `D`, `K`.
   - Observe the live debugger switching to the dual-hand 144D pipeline.
   - Show natural interaction between passive and active hands.
5. **Demonstrate Sentence Composition Controls:**
   - Spell a common word (e.g. `H - E - L - L - O`).
   - Click `␣ Space` (or press spacebar) to complete the word.
   - Spell a second word (e.g. `W - O - R - L - D`).
   - Use `⌫ Backspace` to correct a mistaken letter if needed.
6. **Demonstrate Multilingual Translation:**
   - Select target languages from the language bar (e.g. Hindi, Marathi, Tamil, Bengali).
   - Click **⚡ Translate** to show immediate conversion into the native Indic script.
7. **Demonstrate Speech Synthesis:**
   - Click **🔊 Speak Translation** to demonstrate native speech output.
   - Click **🗣️ Speak English** for English audio playback.
8. **Demonstrate Clean Application Shutdown:**
   - Click **Stop Camera** in the header to demonstrate camera release.
   - Close the browser tab or terminate the application, verifying port 5001 releases cleanly.
