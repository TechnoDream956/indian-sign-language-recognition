# Testing and Quality Assurance Documentation

This document describes the automated test suite, regression verification procedures, and test execution instructions for the **Indian Sign Language (ISL) Recognition & Translation System**.

---

## 1. Test Suite Architecture

The test suite is located in the `tests/` directory and contains **87 automated test cases** covering every layer of the application:

| Test File | Focus Area | Test Count | Key Invariants Verified |
| :--- | :--- | :---: | :--- |
| `tests/test_dual_hand_collection.py` | Solo Data Collection Suite | 15 | Countdown transitions (5s), auto-advance upon sample quota, release period enforcement, expected hand configuration validation. |
| `tests/test_dual_hand_alphabet.py` | Dual-Hand Alphabet Tracking | 13 | Left/right hand landmark coordinate extraction, deterministic ordering, hand count validation, dual-hand vs single-hand letter routing. |
| `tests/test_alphabet_engine.py` | Alphabet Inference & WordBuffer | 19 | 26-class Dense network inference, debounce cooldown (1.2s), space insertion, backspace, sentence composition state machine. |
| `tests/test_dual_model_inference.py`| Sequence Inference & Hot-Swapping | 13 | 30-frame sliding FIFO buffer, hot-swapping between 13-class and 29-class models, label encoder mapping, confidence stability. |
| `tests/test_translator.py` | Multilingual Translation | 11 | Translation into 10 regional Indian languages, language code validation, fallback behavior on network disconnects. |
| `tests/test_tts_engine.py` | Speech Synthesis Engine | 8 | Native macOS `say` subprocess execution, mute/unmute state, speed and volume controls, fallback engine selection. |
| `tests/test_web_endpoints.py` | Flask Web Server & SSE Streaming | 8 | Endpoint routing (`/status`, `/set_mode`, `/switch_model`, `/set_language`, `/buffer_action`), SSE formatting, MJPEG stream headers. |
| **Total** | | **87** | **100% Pass Rate Across All Suites** |

---

## 2. Running the Test Suite

### Run All Tests

To execute the entire test suite:

```bash
# Ensure virtual environment is active
source venv/bin/activate

# Discover and run all unit and regression tests
python -m unittest discover -s tests
```

**Expected Result:**
```text
Loading AlphabetEngine model from: .../models/alphabet_model_26class.h5
AlphabetEngine initialized with 26 classes.
.......................................................................................
----------------------------------------------------------------------
Ran 87 tests in 0.670s

OK
```

### Run Verbose Output

To see each test method and its individual status:

```bash
python -m unittest discover -s tests -v
```

### Run a Specific Test Module

```bash
# Test only the dual-hand alphabet engine
python -m unittest tests/test_dual_hand_alphabet.py

# Test only the Flask web endpoints
python -m unittest tests/test_web_endpoints.py

# Test only the solo data collector
python -m unittest tests/test_dual_hand_collection.py
```

---

## 3. Continuous Integration & Pre-Commit Validation

Before submitting any pull request or pushing updates to GitHub, verify that:
1. All 87 unit tests pass without errors (`python -m unittest discover -s tests`).
2. No syntax or import errors exist across `src/*.py`.
3. Pre-trained model files in `models/` remain unmodified (hashes intact).
4. No temporary files or datasets are staged in git (`git status`).
