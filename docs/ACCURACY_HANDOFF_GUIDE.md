# Accurate Recognition Handoff Guide for Computer Vision / ML Engineer

This document provides a concise, high-value technical handoff for the next AI assistant or computer vision engineer tasked with improving sign recognition accuracy in the **Indian Sign Language (ISL) Recognition & Translation Platform**.

---

## 1. System Architecture & Model Pipeline

The application uses a **Multi-Modal Decision Fusion Pipeline** combining RGB computer vision and skeletal kinematic landmarks:

```
                          ┌───────────────────────────┐
                          │   Live Camera / WebRTC    │
                          │   (640x480 RGB @ 30 FPS)  │
                          └─────────────┬─────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
       ┌─────────────────────────┐             ┌─────────────────────────┐
       │   MediaPipe Hands    │             │   RGB Hand Region Crop  │
       │   (21 Keypoints / Hand) │             │    (Bounding Box / ROI) │
       └────────────┬────────────┘             └────────────┬────────────┘
                    │                                       │
     ┌──────────────┴──────────────┐                        │
     ▼                             ▼                        ▼
┌──────────────┐            ┌──────────────┐         ┌──────────────┐
│ Single-Hand  │            │  Dual-Hand   │         │   RGB CNN    │
│ Landmark MLP │            │ Landmark MLP │         │ (ResNet/Conv)│
│ (42D/63D)    │            │ (144D / 20C) │         │ (128x128x3/26C)│
└──────┬───────┘            └──────┬───────┘         └──────┬───────┘
       │                           │                        │
       └─────────────┬─────────────┘                        │
                     ▼                                      ▼
         ┌──────────────────────────────────────────────────────┐
         │             Decision Fusion Engine                   │
         │   Confidence weighting, posture verification & voting│
         └──────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
                         Predicted Sign Character
```

### Active Model Specifications

| Model Name | Checkpoint Path | Architecture | Input Shape | Classes |
| :--- | :--- | :--- | :--- | :--- |
| **Single-Hand Landmark MLP** | `models/alphabet_model_26class.h5` | 3-Layer Dense + BatchNorm + Dropout + Softmax | `(None, 63)` or `(None, 42)` | 26 (A–Z) |
| **Dual-Hand Landmark MLP** | `models/experimental/phase5b2_prompt23_class_k/config_d_ablation_144d/best_val_model.h5` | Deep MLP (256-128-64) + Spatial Feature Extractor | `(None, 144)` | 20 bimanual classes |
| **RGB Appearance CNN** | `models/cnn_sign_model.h5` | 4 Conv2D blocks + MaxPool + Dense + Softmax | `(None, 128, 128, 3)` | 26 (A–Z) |

---

## 2. Dispatch Logic & Fusion Strategy

The dispatch logic resides in `src/alphabet_engine.py` and `src/fusion_engine.py`:

1. **Hand Presence Dispatch:**
   - **0 Hands Detected:** Return `None` (Neutral state).
   - **1 Hand Detected:** Dispatched to **Single-Hand Landmark MLP** and cropped ROI sent to **RGB CNN**.
   - **2 Hands Detected:** Dispatched to **Dual-Hand 144D Feature Extractor** (`src/spatial_feature_extractor.py`) and evaluated against the 20 bimanual class model.
2. **Confidence-Aware Fusion (`src/fusion_engine.py`):**
   - Landmark Model Weight: $W_{\text{landmark}} = 0.60$
   - RGB CNN Weight: $W_{\text{cnn}} = 0.40$
   - Weighted score: $S(c) = W_{\text{landmark}} \cdot P_{\text{landmark}}(c) + W_{\text{cnn}} \cdot P_{\text{cnn}}(c)$
   - Agreement Bonus: $+0.15$ added when both models agree on the top class.

---

## 3. Known Problematic Letters & Root Cause Analysis

Real webcam evaluation revealed distinct failure modes across 13 letters:

| Letter | Observed Misclassification | Morphological Root Cause |
| :--- | :--- | :--- |
| **B** | False positive anchor | Palm facing camera with 4 extended fingers. Often triggers spuriously when hands enter the frame. |
| **H** | False positive anchor | Two horizontal fingers extended. Low-margin threshold causes default to H when confidence dips. |
| **K** | Often predicted as **T** | Held-out Class K samples exhibit left index extension $\approx 1.12$ vs training $1.50$, matching Class T left index extension $\approx 1.15$. |
| **M & N** | Mutual confusion | Both require tucking fingers over thumb; subtle difference in 2 vs 3 fingers is lost without finger tip contact features. |
| **D & P** | Misclassified as B / H | Single extended index with curled fingers; lighting variations flatten thumb-loop boundary in RGB CNN. |
| **F & Q** | Orientation ambiguity | Rotation around the camera Z-axis causes feature vector normalization to overlap. |
| **I & Y** | Pinky extension confusion | Pinky extension alone (I) vs Thumb + Pinky (Y); thumb occlusion in certain camera angles misleads CNN. |
| **R & U & V** | Finger crossing / separation | R requires crossed index and middle; U requires joined fingers; V requires split fingers. Keypoint normalization loses inter-finger contact distance. |
| **Z** | Static posture limitation | Z in ISL has a dynamic tracing component. In static single-frame evaluation, the starting posture is ambiguous. |

---

## 4. Key Engineering Interventions Needed

1. **Inter-Finger Contact Distance Ratios:**
   - Add explicit Euclidean distance metrics between `INDEX_FINGER_TIP` and `MIDDLE_FINGER_TIP` to distinguish **U** (distance $\to 0$) from **V** (distance $> 0.05$) and **R** (negative signed overlap).
2. **Thumb-Finger Loop Closure Metric:**
   - Compute closed-loop polygon area between `THUMB_TIP` and `INDEX_FINGER_TIP` for **D**, **O**, **P**, and **Q**.
3. **B and H Suppression Gate:**
   - Impose an entry threshold ($\ge 0.75$) specifically for Class B and Class H, preventing them from capturing transitional hand movements when entering or exiting the camera frame.
4. **Bimanual Wrist Translation Invariance:**
   - In `src/spatial_feature_extractor.py`, ensure relative wrist distance is normalized by the signer's torso/shoulder span or hand length so variations in camera distance do not shift K into T.

---

## 5. Commands for Evaluation and Retraining

### Run Active Unit & Accuracy Test Suites
```bash
# Run all unit tests hermetically
./venv/bin/python -m unittest discover -s tests -p "test_*.py"

# Run 26-class accuracy matrix evaluation
./venv/bin/python tests/test_alphabet_26class_matrix.py

# Run live application locally
./run_app.sh
```

### Model Training Tools
```bash
# Train & benchmark alphabet classifier
./venv/bin/python tools/training/train_and_benchmark_alphabet.py

# Train RGB CNN feature backbone
./venv/bin/python tools/training/train_rgb_cnn_model.py
```
