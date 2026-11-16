# Indian Sign Language (ISL) Recognition — Dataset Documentation

## Overview

The Indian Sign Language Recognition & Translation application is bundled with pre-trained neural network weights (`models/*.h5`), allowing the production runtime (both the web application and the standalone macOS `.app`) to operate **completely independently of external or local raw dataset files**. 

This document describes the provenance, schemas, collection methodology, redistribution rights, and licensing caveats for all training datasets and benchmarking subsets used across this project.

**Project Contributors & Roles:**
- **Arpit:** Recognition Model Development
- **Sachin:** Data Collection, Preprocessing & Backend Support
- **Abhu:** Multilingual Support & TTS
- **Yash:** Frontend & Deployment

---

## 1. Dataset Provenance, Counts & Licensing

| Dataset | Modality / Format | Total Samples | Split Distribution | Provenance & Origin | License Status | Redistribution Policy |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| **RealSign** | Cropped RGB ($128\times128$) & MediaPipe 3D Landmarks | 1,613 dual-hand samples | • Train: 1,189<br>• Val: 181<br>• Held-Out Test: 243 | Academic ISL research dataset recorded in controlled indoor conditions. | **TODO: confirm** *(Academic download; upstream terms unverified)* | **DO NOT REDISTRIBUTE RAW ASSETS.** Pretrained weights only in repository. |
| **OpenHands (ISL subset)** | 3D Skeletal Landmarks | 9 held-out test samples | • Test: 9 | OpenHands Sign Language open-source project repository. | **TODO: confirm** *(Permissive open-source; specific terms unverified)* | **DO NOT REDISTRIBUTE RAW ASSETS.** Precomputed features only. |
| **Ayesha (ISL Fingerspelling)** | 2D/3D Landmarks & RGB Crops | ~2,500 single-hand samples | • Test: 536 single-hand | Public ISL fingerspelling alphabet collection. | **TODO: confirm** *(Community repository; terms unverified)* | **DO NOT REDISTRIBUTE RAW ASSETS.** Frozen baseline weights only. |
| **Project Live Benchmark Recordings** | Landmark JSON & Session Telemetry | Variable (ad-hoc live sessions) | Recorded via `tools/collection/record_live_benchmark_samples.py` | Locally captured by team members for empirical testing. | **MIT License (Project Internal)** | Excluded from git tracking via `.gitignore` (`data/live_benchmark_samples/`). |

> [!CAUTION]
> **Redistribution Compliance:**  
> The repository ships **frozen model checkpoints (`models/*.h5`) and JSON label maps only**. Raw image and video archives (`Dataset.zip`, raw MP4s) are located exclusively in local staging reserves and are strictly excluded from git tracking to prevent any potential licensing or distribution violations.

---

## 2. Held-Out Evaluation Partitions (Ablation Ground Truth)

All metrics reported in `docs/RESULTS.md` and `docs/DIAGNOSIS.md` derive from strictly held-out test partitions:

1. **Dual-Hand 20-Class Held-Out Partition (`best_val_model.h5`):**
   - **Full Partition (`sample_ids_test.npy`):** **252 samples** (243 RealSign + 9 OpenHands).
   - **RealSign Same-Signer Subset:** **243 samples** (wrist-to-middle-MCP Euclidean normalization). Evaluates 20 target classes: `A, B, D, E, F, G, H, I, K, M, N, P, Q, R, S, T, W, X, Y, Z`.
   - **OpenHands Subset:** **9 samples** ($L_\infty$ span bounding-box normalization).
2. **Single-Hand 26-Class Baseline (`alphabet_model_26class.h5`):**
   - **Held-Out Test Set:** **536 samples** across characters `A`–`Z`.
3. **RGB Visual CNN Baseline (`cnn_sign_model.h5`):**
   - **Held-Out Test Set:** **780 cropped RGB frames** ($128\times 128\times 3$) from `Dataset.zip`.

---

## 3. Data Leakage & Domain Shift Audit

Empirically audited via `tools/audit/leakage_audit.py`:
- **ID Overlap:** **0.0% overlap** across train, validation, and test splits (zero duplicate sample IDs).
- **Temporal Correlation & Single-Signer Domain Shift:** In `RealSign`, training and held-out test frames were recorded with the **same signer in the same studio environment**. Nearest-neighbor analysis reveals a median Euclidean distance of **0.5492** on 144D features, with 4.5% of test samples within distance $< 0.10$.
- **Classification:** Strictly designated as **same-signer held-out** everywhere. Offline test accuracy evaluates internal consistency; generalization to novel signers requires multi-signer collection.

---

## 4. Live Data Collection Suite

The repository provides `tools/collection/record_live_benchmark_samples.py` for recording new live multi-signer test sessions:
- **Output Directory:** `data/live_benchmark_samples/` (git-ignored)
- **Captured Telemetry:** Synchronized RGB frame crops, raw unmirrored MediaPipe landmarks, mirrored display landmarks, timestamp, and operator labels.
