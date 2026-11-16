# Testing and Quality Assurance Documentation

This document describes the automated test suite, regression verification procedures, and test execution instructions for the **Indian Sign Language (ISL) Recognition & Translation System**.

---

## 1. Test Suite Architecture

The test suite is located in the `tests/` directory and contains **235 automated test cases** across 18 specialized test suites covering every layer of the application:

| Test File | Focus Area | Key Invariants Verified |
| :--- | :--- | :--- |
| `tests/test_alphabet_26class_matrix.py` | Confusion Matrix & Evaluation | Matrix generation, per-class sensitivity & precision for 26 classes. |
| `tests/test_alphabet_accuracy.py` | Model Accuracy Benchmarking | Validates minimum top-1 and top-3 accuracy thresholds on test sets. |
| `tests/test_alphabet_engine.py` | 26-Class Single-Hand Engine | Model loading, preprocessing, 63D vector normalization, confidence scoring. |
| `tests/test_alphabet_translation.py` | WordBuffer to Translation | Debounce state machine, space insertion, sentence assembly, translation trigger. |
| `tests/test_app_packaging_and_lifecycle.py` | Packaging & Port Allocation | Port auto-bind (5001), health check polling, process single-instance mutex. |
| `tests/test_cloud_camera.py` | Frame Ingestion & SSE | Frame decoding, base64 ingestion, authentication tokens, connection resilience. |
| `tests/test_dual_hand_alphabet.py` | Dual-Hand 144D Tracking | Deterministic left/right ordering, wrist distance scaling, bimanual routing. |
| `tests/test_e2e_alphabet_validation.py` | Full Pipeline E2E | Frame input -> perception -> decision fusion -> WordBuffer -> Indic translation. |
| `tests/test_feedback_continuous_learning.py`| Feedback Learning Pipeline | Prediction registration, explicit user correction, quality filtering, retraining. |
| `tests/test_feedback_security_privacy.py` | Privacy & Token Auth | HMAC session tokens, data sanitization, pseudonymization, rate limiting. |
| `tests/test_feedback_storage.py` | SQLite Persistence | Atomic transactions, schema migrations, landmark NPZ serialization. |
| `tests/test_flask_alphabet_integration.py` | Flask REST API Contracts | Status endpoints, camera controls, language switching, buffer actions. |
| `tests/test_language_services.py` | Multilingual & Indic TTS | Language registry, transliteration, cache hit/miss, audio generation. |
| `tests/test_paths_helper.py` | Path Resolution Portability | Dynamic root resolution, resource path helpers, symlink safety. |
| `tests/test_speech_adapter.py` | Speech Adapter & Fallbacks | Browser Web Speech coordination, native desktop speech fallback. |
| `tests/test_translation_adapter.py` | Translation Engine Adapter | Multi-language translation caching, dictionary lookup, error resilience. |
| `tests/test_ui_correction_and_controls.py` | Frontend UI Contracts | Button IDs, modal triggers, state machine, unmirrored canvas rendering. |
| `tests/benchmark_languages.py` | Translation Latency | Sub-millisecond dictionary lookup, cache throughput. |
| **Total** | **18 Suites** | **235 Passing Tests (100% Pass Rate)** |

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
Ran 235 tests in ~12.5s

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
