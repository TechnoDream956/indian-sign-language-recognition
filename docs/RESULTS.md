# Empirical Evaluation & Performance Results

**Repository:** `https://github.com/TechnoDream956/indian-sign-language-recognition`  
**Evaluation Date:** 2026-11-16  
**Benchmarking Environment:** Apple Silicon M2, macOS, Python 3.10.21  
**Team Contributions:**  
- **Arpit:** Recognition Model Development  
- **Sachin:** Data Collection, Preprocessing & Backend Support  
- **Abhu:** Multilingual Support & TTS  
- **Yash:** Frontend & Deployment  

---

## 1. Model Performance on Same-Signer Held-Out Test Partitions

All metrics were computed using `src/evaluate.py`. 95% Confidence Intervals were computed via 1,000 empirical bootstrap iterations.

| Model | Architecture | Input Shape | Test Partition | Test Samples | Top-1 Accuracy [95% CI] | Top-3 Accuracy | Macro F1 [95% CI] | Brier Score |
|---|---|---|---|---|---|---|---|---|
| **ISL Dual-Hand Model (Full)** | Deep MLP + BatchNorm | `(None, 144)` | Full Held-Out Test (RealSign + OpenHands) | 252 | **75.79%** [70.24%, 80.95%] | **82.94%** | **0.6683** [0.6385, 0.6974] | 0.3842 |
| **ISL Dual-Hand Model** | Deep MLP + BatchNorm | `(None, 144)` | RealSign Same-Signer Held-Out (20 Classes) | 243 | **78.19%** [72.43%, 83.13%] | **84.36%** | **0.7349** [0.7067, 0.7558] | 0.3565 |
| **ISL Dual-Hand (+ K/T Override)** | Deep MLP + Heuristic | `(None, 144)` | RealSign Same-Signer Held-Out (20 Classes) | 243 | **83.13%** [78.19%, 87.65%] | **84.36%** | **0.7834** [0.7394, 0.8190] | 0.3565 |
| **ISL Single-Hand Baseline** | Dense MLP | `(None, 63)` | Combined Alphabet Same-Signer Held-Out | 536 | **75.93%** [72.20%, 79.66%] | **86.19%** | **0.6032** [0.5530, 0.6351] | 0.3355 |
| **ISL RGB CNN Classifier** | 4-Block Conv2D + GAP | `(None, 128, 128, 3)` | RealSign Same-Signer Held-Out Test Set | 780 | **79.87%** [77.05%, 82.69%] | **80.77%** | **0.7474** [0.7361, 0.7567] | — |

> [!NOTE]  
> **Reconciliation of Test Partition Size (252 vs 243):**
> `sample_ids_test.npy` contains 252 total samples: 243 from the RealSign dataset and 9 from OpenHands. On the 243 RealSign samples (where feature scale normalization matches training), the model achieves 78.19% (83.13% with K/T override). On the 9 OpenHands samples, the model achieves 11.11% (1/9) due to the $L_\infty$ bounding box vs wrist-to-middle-MCP scale divergence documented in `docs/DIAGNOSIS.md`.
> The 20 dual-hand target classes are: `A, B, D, E, F, G, H, I, K, M, N, P, Q, R, S, T, W, X, Y, Z`. Letters `C, J, L, O, U, V` are strictly unimanual in standard Indian Sign Language and are routed to the 26-class single-hand pathway.

---

## 2. Paired Multi-Modal Fusion Ablation

Evaluated on the exact same 243 same-signer held-out samples where paired raw landmarks and cropped RGB images both exist:

| Modality / Architecture | Top-1 Accuracy | Macro F1 | Notes |
|---|---|---|---|
| **Geometry-Only Baseline (144D MLP)** | 78.19% | 0.7349 | Dual-hand landmark geometry alone |
| **Geometry-Only (+ K/T Override)** | 83.13% | 0.7834 | Disambiguates active index finger orientation |
| **RGB CNN-Only (128x128x3 Conv2D)** | 82.72% | 0.7565 | Cropped visual texture alone |
| **Multi-Modal Fused (Geometry + CNN)** | **83.13%** | **0.7834** | Combined decision fusion |

---

## 3. Comprehensive Mirror Parity Experiment Across Models

Evaluated on held-out test partitions to measure the effect of camera mirroring across all three models:

| Model | Test Set | Unmirrored Accuracy | Mirrored Accuracy | Accuracy Delta |
|---|---|---|---|---|
| **Dual-Hand 144D MLP** | 243 same-signer samples | **78.19%** | **29.22%** | **-48.97% drop** |
| **Single-Hand 63D MLP** | 536 same-signer samples | **75.93%** | **12.31%** | **-63.62% drop** |
| **RGB CNN (128x128x3)** | 780 same-signer samples | **79.87%** | **62.82%** | **-17.05% drop** |

> [!IMPORTANT]  
> All three models require unmirrored sensor inputs matching training perspective. Feeding horizontally flipped webcam images causes immediate prediction failure across landmark and visual modalities.

---

## 4. Dual-Hand Heuristics & Statistical Significance (McNemar Test)

Evaluated on the 243 same-signer held-out test samples (`best_val_model.h5`):

| Configuration | Top-1 Accuracy | Delta Acc | Macro F1 | Delta F1 | Finding & Status |
|---|---|---|---|---|---|
| **Baseline Model-Only** | 78.19% | — | 0.7349 | — | High baseline performance on held-out test set |
| **With K vs T Override** | **83.13%** | **+4.94%** | **0.7834** | **+0.0485** | **Statistically Significant**: $\chi^2 = 6.7222, p = 0.0075$ |
| **With B/H Heuristic Clamp** | 72.02% | -6.17% | 0.6281 | -0.1068 | **Disabled by Default**: Wrist distance clamp rejects valid poses |

---

## 5. Temporal Hysteresis & Flicker Reduction

Evaluated on an intermittent 50-frame dropout sequence with 4 brief hand occlusions (1–2 frames each):

| Configuration | Pipeline Switches / Flaps | Flap Reduction | User Experience Impact |
|---|---|---|---|
| **Grace = 0 Frames (Legacy)** | 7 switches | Baseline | Broken letter latching; resets stability counter |
| **Grace = 2 Frames (Hysteresis)** | 1 switch | **-85.7% flapping** | Stable letter latching; carries occluded hand position |

---

## 6. Hardware Inference Latency & End-to-End Throughput

Measured on **Apple Silicon M2 (CPU)** using `src/benchmark_latency.py`:

### Section A: Component & Inference-Only Benchmarks (Excludes Sensor Capture & Drawing)
| Component | Median Latency | Mean Latency | p95 Latency | Throughput (FPS) |
|---|---|---|---|---|
| **Dual-Hand 144D Feature Extraction** | 0.08 ms | 0.08 ms | 0.09 ms | 11,880.8 FPS |
| **Dual-Hand Engine (Features + MLP)** | 2.91 ms | 2.91 ms | 3.01 ms | 343.1 FPS |
| **Single-Hand MLP (63D) Inference** | 1.81 ms | 1.86 ms | 2.07 ms | 538.5 FPS |
| **RGB CNN Engine (Crop + 128x128)** | 8.91 ms | 9.91 ms | 14.58 ms | 100.9 FPS |
| **Multi-Modal Decision Fusion** | < 0.01 ms | < 0.01 ms | < 0.01 ms | > 600,000 FPS |
| **AlphabetEngine Single-Hand Decision** | 9.12 ms | 9.28 ms | 9.92 ms | 107.8 FPS |
| **AlphabetEngine Dual-Hand Decision** | 14.62 ms | 14.84 ms | 15.84 ms | 67.4 FPS |

### Section B: True Full End-to-End Pipeline (`InferenceEngine.process_frame`)
*(Sensor Ingestion -> Orientation -> MediaPipe Hands Tracking -> Landmarks -> Models -> Fusion -> OpenCV Drawing -> JPEG Encoding)*
| Pipeline Mode | Median Latency | Mean Latency | p95 Latency | Throughput (FPS) |
|---|---|---|---|---|
| **Empty Frame (No Hands Detected)** | 13.23 ms | 13.28 ms | 13.56 ms | 75.3 FPS |
| **Active Dual-Hand Frame** | 39.99 ms | 44.16 ms | 53.96 ms | 22.6 FPS |

> [!NOTE]  
> **Statistical Distribution & Tail Skew:**  
> Arithmetic mean latency can approach or slightly exceed p95 due to right-tail skew from infrequent OS thread scheduling, Apple Silicon frequency transitions, and Python garbage collection pauses (reflected in p99 and max). Median represents sustained frame-rate performance.

---

## 7. Artifacts & Generated Assets

- **Dual-Hand 144D Confusion Matrix:** Saved to `assets/confusion_matrix.png`
- **Model Checkpoints:**
  - Dual-Hand: `models/experimental/phase5b2_prompt23_class_k/config_d_ablation_144d/best_val_model.h5`
  - Single-Hand: `models/alphabet_model_26class.h5`
  - RGB CNN: `models/cnn_sign_model.h5`
- **Benchmarking CLI Scripts:**
  - Model Evaluation: `python src/evaluate.py`
  - Latency Benchmarking: `python src/benchmark_latency.py`
  - Data Leakage Audit: `tools/audit/leakage_audit.py`
  - Live Sample Collection & Evaluation: `tools/collection/record_live_benchmark_samples.py`
