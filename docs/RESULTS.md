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

### Clean Offline Benchmarks (Uncontaminated Estimates)
| Model | Architecture | Input Shape | Test Partition | Test Samples | Top-1 Accuracy [95% CI] | Top-3 Accuracy | Macro F1 [95% CI] | Brier Score |
|---|---|---|---|---|---|---|---|---|
| **ISL Dual-Hand Baseline Model** | Deep MLP + BatchNorm | `(None, 144)` | RealSign Same-Signer Held-Out (20 Classes) | 243 | **78.19%** [72.43%, 83.13%] | **84.36%** | **0.7349** [0.7067, 0.7558] | 0.3565 |
| **ISL RGB CNN Classifier** | 4-Block Conv2D + GAP | `(None, 128, 128, 3)` | RealSign Paired Test Subset (20 Classes) | 243 | **82.72%** [77.78%, 87.24%] | — | **0.7565** [0.7180, 0.7925] | — |
| **ISL RGB CNN Classifier (Full)** | 4-Block Conv2D + GAP | `(None, 128, 128, 3)` | RealSign Image Held-Out (26 Classes) | 780 | **79.87%** [77.05%, 82.69%] | **80.77%** | **0.7474** [0.7361, 0.7567] | — |
| **Multi-Modal Fused (WITHOUT K/T)** | Landmark Prior + CNN Agreement | `144D + 128x128` | RealSign Paired Test Subset (20 Classes) | 243 | **78.19%** [72.43%, 83.13%] | **84.36%** | **0.7349** [0.7067, 0.7558] | — |
| **ISL Single-Hand Baseline** | Dense MLP | `(None, 63)` | Combined Alphabet Same-Signer Held-Out | 536 | **75.93%** [72.20%, 79.66%] | **86.19%** | **0.6032** [0.5530, 0.6351] | 0.3355 |
| **ISL Dual-Hand Model (Full Split)** | Deep MLP + BatchNorm | `(None, 144)` | Full Held-Out Test (RealSign + OpenHands) | 252 | **75.79%** [70.24%, 80.95%] | **82.94%** | **0.6683** [0.6385, 0.6974] | 0.3842 |

### Post-Hoc Heuristic Overrides (Contaminated Estimates)
| Configuration | Architecture | Input Shape | Test Partition | Test Samples | Top-1 Accuracy | Macro F1 | Status & Validity |
|---|---|---|---|---|:---:|:---:|---|
| **Dual-Hand (+ K/T Override)** | Deep MLP + Heuristic | `(None, 144)` | RealSign Same-Signer Held-Out | 243 | **83.13%** | **0.7834** | *Possibly tuned on test data, not a clean estimate* |
| **Multi-Modal Fused (+ K/T)** | Fused + Heuristic | `144D + 128x128` | RealSign Paired Test Subset | 243 | **83.13%** | **0.7834** | *Matches Geometry (+KT) alone (0.00% gain)* |

> [!WARNING]  
> **Contaminated Estimate Warning (K/T Override):**  
> The index finger orientation heuristic (`dy > 0.08 and dy > 1.2 * dx`) was formulated by inspecting Class K $\to$ Class T misclassifications during evaluation iterations. Because these thresholds were informed by the test partition, the +4.94% accuracy gain (78.19% $\to$ 83.13%) is **possibly tuned on test data and is not a clean estimate**.

> [!NOTE]  
> **Reconciliation of Test Partition Size (252 vs 243):**
> `sample_ids_test.npy` contains 252 total samples: 243 from the RealSign dataset and 9 from OpenHands. On the 243 RealSign samples (where feature scale normalization matches training), the clean baseline achieves 78.19%. On the 9 OpenHands samples, the model achieves 11.11% (1/9) due to the $L_\infty$ bounding box vs wrist-to-middle-MCP scale divergence documented in `docs/DIAGNOSIS.md`.
> The 20 dual-hand target classes are: `A, B, D, E, F, G, H, I, K, M, N, P, Q, R, S, T, W, X, Y, Z`. Letters `C, J, L, O, U, V` are strictly unimanual in standard Indian Sign Language and are routed to the 26-class single-hand pathway.

### Near-Duplicate Sensitivity Analysis (Nearest-Neighbor Distance Check)

To ensure performance is not an artifact of temporally adjacent video frames in the held-out partition, pairwise Euclidean 1-NN distances in normalized 144D feature space to the training set were computed. Samples with 1-NN distance $< 0.10$ are classified as near-duplicates (11 samples, 4.53% of the 243 RealSign test set).

| Evaluation Set | Samples (N) | Clean Baseline 144D Acc | CNN Alone Acc | Fused (without K/T) Acc | Geometry (+ K/T)* Acc | Fused (+ K/T)* Acc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full RealSign Held-Out** | 243 | **78.19%** | **82.72%** | **78.19%** | 83.13%* | 83.13%* |
| **Clean Non-Duplicate Subset** (1-NN dist $\ge 0.10$) | 232 | **77.16%** | **81.90%** | **77.16%** | 82.33%* | 82.33%* |

*\*Note: K/T override numbers are possibly tuned on test data.*  
*Conclusion:* Removing the 4.5% near-duplicate samples results in a negligible ~1.0% change in Top-1 accuracy (78.19% $\to$ 77.16% baseline; 82.72% $\to$ 81.90% CNN alone), demonstrating that performance reflects genuine skeletal/visual patterns rather than duplicate frame memorization.

---

## 2. Paired Multi-Modal Fusion, Agreement & Disagreement Analysis

Evaluated on the exact same 243 same-signer held-out samples where paired raw landmarks and cropped RGB images both exist:

| Modality / Architecture | Top-1 Accuracy | Macro F1 | Performance vs Best Single Branch |
|---|:---:|:---:|---|
| **Geometry Baseline (144D MLP)** | **78.19%** | **0.7349** | Clean landmark geometry alone |
| **RGB CNN Alone (128x128x3 Conv2D)** | **82.72%** | **0.7565** | Clean cropped visual appearance alone |
| **Multi-Modal Fused (WITHOUT K/T)** | **78.19%** | **0.7349** | **Equals Geometry branch (78.19%), below CNN alone (82.72%)** |
| *Geometry (+ K/T Override)* | *83.13%* | *0.7834* | *Post-hoc heuristic (possibly tuned on test data, not a clean estimate)* |
| *Multi-Modal Fused (+ K/T Override)* | *83.13%* | *0.7834* | *Matches Geometry (+KT) alone (0.00% gain)* |

> [!IMPORTANT]  
> **Honest Assessment of Decision Fusion & Offline vs Live Ranking:**  
> On this same-signer held-out test set, multi-modal decision fusion without the K/T override **equals the geometry branch (78.19%) and is below the CNN alone (82.72%)**. Because the fusion engine defaults to the landmark prior when confidence is split or high, fusion does not improve accuracy.  
> **Crucial Caveat on CNN Performance:** The RGB CNN was trained on the exact same signer's cropped studio photos under fixed lighting and camera focal length. As documented in our live engineering audit (`docs/DIAGNOSIS.md`), visual CNNs generalize very poorly live to novel signers with differing skin tones, clothing, and background clutters. Consequently, **the offline ranking showing CNN higher than landmark geometry is an artifact of the single-signer dataset and says nothing about live real-world performance**.

### Agreement vs Disagreement Breakdown (Clean Baseline Geometry vs CNN)

Evaluated using the **clean baseline geometry model (strictly WITHOUT the K/T override)** against the visual CNN. When the two modalities disagree, they cannot both be correct:

| Metric / Subset | Full Held-Out Set (N=243) | Clean De-duplicated Set (N=232) | Interpretation |
|---|:---:|:---:|---|
| **Branches AGREE (Geometry == CNN)** | **186 / 243 (76.54%)** | **175 / 232 (75.43%)** | Both modalities converge on identical sign |
| **Accuracy on Agreement Subset** | **100.0% (186 / 186)** | **100.0% (175 / 175)** | **Zero errors observed when modalities agree** |
| **Branches DISAGREE (Geometry != CNN)** | **57 / 243 (23.46%)** | **57 / 232 (24.57%)** | Modalities diverge |
| • **Samples where Geometry is right** | **4 / 57 (7.02%)** | **4 / 57 (7.02%)** | Landmark skeleton correct; CNN visual texture fails |
| • **Samples where CNN is right** | **15 / 57 (26.32%)** | **15 / 57 (26.32%)** | CNN visual texture correct; landmark skeleton fails |
| • **Samples where NEITHER is right** | **38 / 57 (66.67%)** | **38 / 57 (66.67%)** | **Both models fail on ambiguous or occluded frames** |
| **Fused Accuracy on Disagreement Subset** | **7.02% (4 / 57)** | **7.02% (4 / 57)** | Fusion defaults to landmark prior (4/57 correct) |

*Reconciliation Check:*  
- Full N=243: Geometry total correct = 186 (agree) + 4 (disagree) = 190 (78.19%). CNN total correct = 186 (agree) + 15 (disagree) = 201 (82.72%).  
- De-duplicated N=232: Geometry total correct = 175 (agree) + 4 (disagree) = 179 (77.16%). CNN total correct = 175 (agree) + 15 (disagree) = 190 (81.90%).

> [!NOTE]  
> **Separate Reference: Agreement with Post-Hoc K/T Override (Possibly Tuned on Test Data):**  
> When the post-hoc K/T override is applied to geometry, 12 Class T predictions flip to Class K (the true label for those samples). This artificially shifts 12 disagreement samples into agreement:
> - Branches AGREE: 198 / 243 (81.48%), Accuracy on Agree = 100.0% (198/198).
> - Branches DISAGREE: 45 / 243 (18.52%) $\to$ Geometry right: 4 / 45 (8.89%), CNN right: 3 / 45 (6.67%), Neither right: 38 / 45 (84.4%).  
> As noted, this K/T-augmented estimate is circular and is not a clean measure of model behavior.

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

## 4. Dual-Hand Heuristics & Statistical Significance (McNemar Test Analysis)

Evaluated on the 243 same-signer held-out test samples (`best_val_model.h5`):

| Configuration | Top-1 Accuracy | Delta Acc | Macro F1 | Delta F1 | Finding & Methodological Status |
|---|---|---|---|---|---|
| **Clean Baseline Model-Only** | **78.19%** | — | **0.7349** | — | Clean held-out benchmark reference |
| **With K vs T Override** | 83.13% | +4.94% | 0.7834 | +0.0485 | *Nominal $\chi^2 = 6.7222, p = 0.0075$ — **INVALID hypothesis test** due to circular test-data tuning* |
| **With B/H Heuristic Clamp** | 72.02% | -6.17% | 0.6281 | -0.1068 | **Disabled by Default**: Wrist distance clamp rejects valid poses |

> [!CAUTION]  
> **Invalidity of McNemar Statistical Test:**  
> While the paired McNemar test yielded a nominal $\chi^2 = 6.7222$ and exact binomial $p = 0.0075$, this test is **methodologically invalid as a confirmatory hypothesis test**. The geometric override thresholds (`dy > 0.08 and dy > 1.2 * dx`) were derived after observing Class K misclassifications on these same held-out samples. Evaluating a post-hoc rule on the data that inspired it violates test-set independence.

---

## 5. Temporal Hysteresis & Flicker Reduction (Synthetic Benchmark)

Evaluated on a **synthetic 50-frame dropout sequence, not real recordings** with 4 brief hand occlusions (1–2 frames each):

| Configuration | Pipeline Switches / Flaps | Flap Reduction | User Experience Impact |
|---|---|---|---|
| **Grace = 0 Frames (Legacy)** | 7 switches | Baseline | Broken letter latching; resets stability counter |
| **Grace = 2 Frames (Hysteresis)** | 1 switch | **-85.7% flapping** | Stable letter latching; carries occluded hand position |

*\*Note: This 7-to-1 flicker reduction is measured on a synthetic 50-frame dropout sequence, not real recordings.*

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

## 7. Remaining Work & Future Hypotheses

1. **Disagreement-Gated Uncertainty Output (Hypothesis):**  
   - *Observation:* On the held-out test set under clean baseline evaluation, when landmark geometry and visual CNN branches disagree (57 samples), neither branch is correct 66.67% of the time (38/57; 84.4% or 38/45 under the K/T override).
   - *Hypothesis:* Emitting an explicit `"Uncertain / Re-sign"` status or withholding letter emission when the two modalities disagree could prevent divergence-induced errors in live usage.
   - *Status:* **Hypothesis to be tested on live multi-signer data, not an established result.** Multi-signer evaluation is required before activating this in production.
2. **Multi-Signer Re-Training:**  
   - Incorporate diverse signers with variable hand dimensions, joint elasticity, and skin tones to overcome same-signer over-reliance.
3. **Auditory Verification:**  
   - Conduct structured human listening tests across all regional Indic speech output tiers.

---

## 8. Artifacts & Generated Assets

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
