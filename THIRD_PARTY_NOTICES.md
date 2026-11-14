# Third-Party Software & Data Notices

This document details the licensing, attributions, and copyright notices for third-party software libraries, pre-trained frameworks, and datasets utilized in the Indian Sign Language Recognition & Translation application.

---

## 1. Verified Third-Party Open Source Software

The following core software libraries are utilized at runtime by this project under standard permissive open-source licenses:

### MediaPipe
- **Copyright**: Copyright 2020–2024 The MediaPipe Authors
- **License**: Apache License, Version 2.0
- **Website**: https://github.com/google/mediapipe
- **Usage**: Used for real-time 3D landmark extraction (Holistic pipeline for Word Mode, Hands pipeline for Alphabet Mode).

### TensorFlow & tf-keras
- **Copyright**: Copyright 2015–2024 The TensorFlow Authors / Keras Team
- **License**: Apache License, Version 2.0
- **Website**: https://www.tensorflow.org
- **Usage**: Neural network runtime for MOPGRU sequence models and Alphabet MLP feedforward models.

### OpenCV (Open Source Computer Vision Library)
- **Copyright**: Copyright 2000–2024 OpenCV Foundation / Intel Corporation
- **License**: Apache License, Version 2.0
- **Website**: https://opencv.org
- **Usage**: Video capture from local webcam, image decoding, drawing, and JPEG compression for MJPEG streaming.

### NumPy
- **Copyright**: Copyright 2005–2024 NumPy Developers
- **License**: BSD 3-Clause License
- **Website**: https://numpy.org
- **Usage**: Multi-dimensional landmark array manipulations and vector calculations.

### Flask & Pallets Ecosystem (Werkzeug, Jinja2, Click, itsdangerous, MarkupSafe)
- **Copyright**: Copyright 2010–2024 Pallets Projects
- **License**: BSD 3-Clause License
- **Website**: https://palletsprojects.com
- **Usage**: Local HTTP application server, template rendering, and Server-Sent Events (SSE) dispatching.

### Scikit-learn
- **Copyright**: Copyright 2007–2024 Scikit-learn developers
- **License**: BSD 3-Clause License
- **Website**: https://scikit-learn.org
- **Usage**: Label encoding serialization (`LabelEncoder`) for sign classes.

### Pillow (PIL Fork)
- **Copyright**: Copyright 2010–2024 Alex Clark and Contributors; 1997–2011 Secret Labs AB
- **License**: HPND License (MIT-like permissive)
- **Website**: https://python-pillow.org
- **Usage**: Image loading and processing utilities.

---

## 2. Dataset Attribution & Provenance

### A. RealSign Indian Sign Language Dataset (Verified)
- **Source**: https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset
- **Authors**: RealSign Project Contributors
- **License**: Creative Commons Zero 1.0 Universal (CC0 1.0) — Public Domain Dedication
- **Verified Status**: **VERIFIED SAFE FOR ALL USES**
- **Description**: 26,000 RGB images of isolated Indian Sign Language alphabet gestures (A–Z) across multiple signers. Sanitized landmark arrays are used to train `models/alphabet_model_26class.h5`.

### B. AI4Bharat INCLUDE Dataset (Verification Pending)
- **Source**: Zenodo Record 4010759 / IIT Madras (AI4Bharat)
- **Authors**: Advaith Sridhar, Rohith G, et al.
- **Reference**: *"INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition"*
- **Verified Status**: **NEEDS UPSTREAM VERIFICATION BEFORE COMMERCIAL REDISTRIBUTION**
- **Description**: Videos corresponding to 4 classes (`family`, `hospital`, `medicine`, `school`) incorporated into the experimental 29-class model. The research paper indicates open research availability, but specific commercial redistribution rights require explicit upstream confirmation with AI4Bharat.

### C. ISLRTC Dictionary Recordings (Verification Pending)
- **Source**: Indian Sign Language Research and Training Centre (ISLRTC), Department of Empowerment of Persons with Disabilities, Ministry of Social Justice and Empowerment, Government of India.
- **Verified Status**: **NEEDS UPSTREAM VERIFICATION BEFORE COMMERCIAL REDISTRIBUTION**
- **Description**: Educational demonstration videos for 12 classes (`food`, `help`, `home`, `how`, `name`, `no`, `please`, `sorry`, `water`, `what`, `where`, `yes`). Government educational materials typically permit public/academic use, but redistributing derivative model weights commercially requires written clarification from ISLRTC.

### D. Original Baseline Repository Recordings (`MP_Data`)
- **Source**: Original base repository (`barathi-1993/Indian-Sign-Language-Recognition`)
- **Verified Status**: **NEEDS UPSTREAM VERIFICATION BEFORE COMMERCIAL REDISTRIBUTION**
- **Description**: 30 landmark sequence recordings per class across 13 baseline signs (`hello`, `thanks`, `you`, etc.). No explicit software license was attached to the original student GitHub repository.

---

## 3. Commercial & Distribution Notice

> [!WARNING]
> While the **Alphabet Mode** (`alphabet_model_26class.h5`) is trained exclusively on CC0 Public Domain data (`RealSign`) and is fully unencumbered, the **Word Mode models** (`mopgru_model.h5` and `mopgru_model_29class.h5`) incorporate data from repositories without explicit commercial grant licenses.
> 
> Therefore, this software package is currently approved and documented for **educational, academic, research, and non-commercial evaluation purposes**. Commercial redistribution should not occur until upstream licenses for `MP_Data`, `INCLUDE`, and `ISLRTC` are formally audited or replaced with custom-recorded signs.
