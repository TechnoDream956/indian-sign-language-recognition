# Indian Sign Language (ISL) Recognition — Dataset Documentation

## Overview

The Indian Sign Language Recognition & Translation application is bundled with pre-trained neural network weights (`models/*.h5`), allowing the production runtime (both the web application and the standalone macOS `.app`) to operate **completely independently of external or local dataset files**. 

This document describes the provenance, schemas, collection methodology, and licensing caveats for the training datasets and collection tools.

---

## 1. Dynamic Sign Dataset (Sequence Recognition)

### Target Signs
- **13-Class Foundational Model (`mopgru_model.h5`):**
  `Hello`, `Thank You`, `Yes`, `No`, `Please`, `Help`, `Good`, `Bad`, `Food`, `Water`, `Father`, `Mother`, `Friend`.
- **29-Class Extended Model (`mopgru_model_29class.h5`):**
  Expands the lexicon to 29 continuous conversational signs covering common greetings, requests, emergency queries, and familial terms.

### Feature Representation
- **Temporal Depth:** 30 consecutive video frames per sequence sampled at ~25–30 FPS.
- **Spatial Geometry:** MediaPipe Holistic landmark extraction producing **1,662 features per frame**:
  - **Pose:** 33 landmarks $\times$ 4 coordinates $(x, y, z, \text{visibility}) = 132$ features
  - **Face Mesh:** 468 landmarks $\times$ 3 coordinates $(x, y, z) = 1,404$ features
  - **Left Hand:** 21 landmarks $\times$ 3 coordinates $(x, y, z) = 63$ features
  - **Right Hand:** 21 landmarks $\times$ 3 coordinates $(x, y, z) = 63$ features
  - **Total:** $132 + 1,404 + 63 + 63 = 1,662$ coordinates per frame.
  - **Sequence Tensor Shape:** $(30, 1662)$.

### Data Augmentation & Normalization
- Coordinate jittering ($\sigma = 0.01$), temporal warping ($\pm 10\%$), and shoulder-width normalization.
- Stratified splitting: 80% Training, 10% Validation, 10% Test.

---

## 2. Static Fingerspelling Alphabet Dataset (Single-Frame Recognition)

### Target Classes
- **26 English Alphabetic Characters (`A` through `Z`):**
  Full Indian Sign Language fingerspelling representation.

### Hand Configuration
Based on the standard Indian Sign Language fingerspelling system:
- **Single-Handed Signs:** `C`, `L`, `O`, `U`, `V` (dominant hand only)
- **Two-Handed Signs:** `A`, `B`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `M`, `N`, `P`, `Q`, `R`, `S`, `T`, `W`, `X`, `Y`, `Z` (both hands interacting)

### Feature Representation
- **Spatial Geometry:** Single frame feature vector of **1,662 dimensions**, formatted identically to the dynamic sign input to ensure uniform pipeline compatibility.
- **Normalization:** Min-max spatial normalization centered around wrist landmarks to provide scale and translation invariance across different user camera distances.

---

## 3. Solo Dual-Hand Collection Suite (`src/collect_dual_hand_alphabet.py`)

To expand the dual-hand alphabet dataset, the project includes an automated collection utility.

### Storage Format:
- Data directory: `data/phase5p_dual_hand/`
- Structure:
  - `<LABEL>/<LABEL>_<INDEX>_<TIMESTAMP>.npy`: Raw numpy array containing 84 normalized coordinates (42 left hand + 42 right hand landmarks).
  - `metadata.csv`: CSV log tracking each capture event:
    - Columns: `sample_id`, `label`, `hand_count`, `has_left`, `has_right`, `timestamp`, `filepath`, `status`

---

## 4. Attribution & Licensing Caveats

| Dataset / Resource | Source / Authors | License / Usage Terms | Notes |
| :--- | :--- | :--- | :--- |
| **RealSign Dataset** | Academic research community | Creative Commons CC0 / Open Access | Primary source for hand landmark baselines. |
| **ISLRTC Guidelines** | Indian Sign Language Research and Training Centre (Govt. of India) | Educational & Research Use | Official standard for Indian Sign Language alphabet and word formations. |
| **INCLUDE Dataset** | Sridhar et al., ACM Multimedia | Non-commercial academic research | Reference benchmark for large-scale ISL gesture taxonomies. |
| **MOPGRU Reference Paper** | Subramanian et al., *Scientific Reports* (Nature Portfolio, 2022) | Open Access (CC BY 4.0) | Theoretical architecture for recurrent sequence modeling. |

> [!IMPORTANT]
> **Zero Runtime Data Footprint:** Neither `src/app.py` nor `dist/ISL Translator.app` requires raw `.npy` or video files. Pretrained model weights are frozen in `models/`, ensuring full reproducibility, zero bloat in git, and rapid cloning.
