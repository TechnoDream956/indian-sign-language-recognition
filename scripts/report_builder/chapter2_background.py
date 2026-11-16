"""Chapter 2 builder: Background Research, Objectives, Literature Review, and Project Feasibility.
"""

from docx import Document
from .xml_helpers import (
    add_heading_1, add_heading_2, add_body_p, add_bullet_p,
    add_table_caption, format_academic_table
)


def build_chapter2(doc: Document):
    """Generate Chapter 2: Background Research, Objectives, Literature Review, and Feasibility."""
    
    # 2. BACKGROUND RESEARCH
    add_heading_1(doc, "2. BACKGROUND RESEARCH")

    add_body_p(
        doc,
        "Sign Language Recognition (SLR) is an inherently interdisciplinary computational domain spanning computer vision, "
        "biomechanical kinematics, deep neural representations, and natural language processing. In the Indian context, "
        "computational research has lagged behind international benchmarks due to a historical scarcity of standardized, "
        "publicly annotated video corpora, wide regional dialectal divergence, and the intrinsic complexity of bimanual "
        "fingerspelling. This chapter establishes the rigorous engineering objectives of our platform, delivers a critical "
        "systematic literature review of state-of-the-art vision and translation architectures, and presents a multi-dimensional "
        "feasibility assessment governing on-device edge execution."
    )

    # 2.1 Objectives
    add_heading_2(doc, "2.1. Objectives")

    add_body_p(
        doc,
        "To address the limitations of prior systems, this project establishes a concrete matrix of functional, "
        "computational, and accessibility objectives. These objectives transition the software from an isolated, experimental "
        "classifier into an integrated, production-grade assistive communication platform. The specific, quantifiable "
        "goals governing development are detailed in Table 1."
    )

    # Table 1: Goals and Objectives
    add_table_caption(doc, 1, "System Goals and Quantitative Engineering Objectives")
    table1 = doc.add_table(rows=7, cols=4)
    headers = ["Goal ID", "Functional Domain", "Engineering Objective", "Quantitative Success Metric"]
    for idx, h in enumerate(headers):
        table1.rows[0].cells[idx].text = h

    goals_data = [
        ("G-01", "Kinematic Perception", "Extract 3D anatomical skeletal landmarks from live webcam streams using MediaPipe Hands.", ">= 20 FPS on commodity CPU; 21 3D landmarks per hand extracted with < 20 ms per-frame latency."),
        ("G-02", "Alphabet Recognition", "Accurately classify static and dynamic ISL fingerspelling gestures across letters A–Z.", ">= 75% overall accuracy on held-out same-signer benchmark partition; support both 1-hand & 2-hand signs."),
        ("G-03", "Decision Fusion & Gating", "Fuse geometric skeletal landmarks with RGB appearance CNN features; filter transitional noise.", "W_geom=0.65, W_cnn=0.35; +0.10 agreement bonus; confidence threshold 0.50."),
        ("G-04", "Temporal Smoothing", "Implement an intelligent WordBuffer state machine with debouncing to assemble words hands-free.", "Cooldown debounce period of 1.2 s; 5-frame stability confirmation threshold; backspace & clear gestures."),
        ("G-05", "Multilingual Translation", "Translate assembled English phrases into 10 Indian regional languages with offline dictionary mapping.", "< 5 ms local dictionary lookup, zero cloud dependency for core phrases; optional IndicTrans2 neural module."),
        ("G-06", "Speech Output", "Synthesize audible speech corresponding to translated regional text using multi-tier fallback.", "Sub-second response using pyttsx3, Edge-TTS, gTTS, or browser Web Speech API; optional Meta MMS-TTS module.")
    ]

    for r_idx, row in enumerate(goals_data, start=1):
        for c_idx, val in enumerate(row):
            table1.rows[r_idx].cells[c_idx].text = val

    format_academic_table(table1, col_widths=[0.8, 1.4, 2.8, 2.0])

    # 2.2 Literature Review & Project Feasibility
    add_heading_2(doc, "2.2. Literature Review & Project Feasibility")

    add_body_p(
        doc,
        "A comprehensive review of modern sign language literature reveals a decisive evolution from sensor-laden hardware "
        "toward non-invasive deep learning vision pipelines. Early works relied on colored gloves or Microsoft Kinect depth sensors "
        "to resolve finger occlusions. While depth sensors simplify spatial segmentation, their commercial discontinuation "
        "and reliance on specialized infrared emitters rendered them obsolete for ubiquitous consumer deployment. The advent "
        "of high-speed deep perception frameworks—most notably Google MediaPipe by Lugaresi et al. (2019) and Zhang et al. (2020)—"
        "revolutionized the field by enabling 2.5D landmark extraction directly from RGB frames on low-power mobile and CPU edge devices."
    )

    add_body_p(
        doc,
        "Subramanian et al. (2022) demonstrated the efficacy of landmark-based sequence classification for Indian Sign Language, "
        "proposing an integrated MediaPipe-Optimized Gated Recurrent Unit (MoP-GRU) pipeline. Their architecture extracted 1,662 "
        "landmarks across pose, face, and hands over a 30-frame temporal window, achieving strong recognition on an isolated 13-sign "
        "vocabulary. However, their research evaluated whole dynamic word sequences rather than fingerspelling alphabets, lacked "
        "a mechanism to handle bimanual hand-swapping, and did not incorporate downstream language translation or acoustic speech synthesis."
    )

    add_body_p(
        doc,
        "Upstream Work and Project Contributions: This repository originated as a research fork of the MoP-GRU implementation "
        "by Barathi Subramanian et al. (2022). In accordance with the MIT License, our work preserves the original copyright notice. "
        "From the upstream repository, we inherited the dynamic sequence modeling concept and baseline landmark utilities "
        "(preserved in legacy/mopgru_cell.py, legacy/collect_keypoints.py, legacy/preprocess_keypoints.py, legacy/realtime_inference.py, "
        "and legacy/train_word_model.py, with pretrained models archived in reserve/old_models/). "
        "Our team's novel contributions encompass: (1) shifting from closed-set dynamic word gestures to the complete 26-letter "
        "combinatorial ISL alphabet, comprising a 26-class single-hand MLP (63D) and a 20-class dual-hand bimanual spatial MLP (144D); "
        "(2) training and integrating a visual appearance CNN (128x128x3) and confidence-aware decision fusion engine; "
        "(3) designing a temporal WordBuffer state machine with 1.2s debouncing and hands-free text manipulation; "
        "(4) engineering a multi-tier speech fallback engine (pyttsx3 -> Edge-TTS -> gTTS -> Web Speech API) and 10-language Indic translation; "
        "(5) diagnosing and fixing camera selfie mirroring (-48.97% drop) and 2-frame occlusion hysteresis; and "
        "(6) developing a modern desktop application (.app), Docker containerization, and an automated 245-test suite."
    )

    add_body_p(
        doc,
        "In a comprehensive comparative study, Adaloglou et al. (2022) evaluated deep learning-based SLR across "
        "RGB, optical flow, and skeletal modalities. Their findings confirmed that while 3D Convolutional Networks (such as I3D) "
        "capture fine-grained appearance, they suffer from extreme computational complexity (>100 GFLOPs), precluding real-time "
        "local CPU execution. Skeletal graph networks and multi-layer perceptrons (MLPs) operating on pre-extracted landmarks "
        "provided a superior trade-off, reducing model parameters by two orders of magnitude while preserving kinematic trajectory. "
        "Similarly, Rastgoo et al. (2021) and Koller (2020) emphasized that the primary failure mode of skeletal SLR is "
        "the loss of inter-finger contact semantics and camera perspective distortion, which necessitates multi-modal fusion "
        "with localized RGB visual crops."
    )

    add_body_p(
        doc,
        "Regarding the linguistic translation layer, sign language machine translation (SLMT) has historically treated sign tokens "
        "as raw text glosses. Camgoz et al. (2018) pioneered neural sign language translation using attentional sequence-to-sequence "
        "models. However, translating English tokens into regional Indian languages presents grammatical challenges, "
        "including Subject-Object-Verb (SOV) sentence structuring and rich agglutinative morphology. Recent state-of-the-art work "
        "by Gala et al. (2023) introduced IndicTrans2, an open-source Transformer model supporting all 22 scheduled Indian languages. "
        "By utilizing the 200M distilled int8-quantized ONNX variant of IndicTrans2 as an optional offline module alongside local dictionary "
        "mapping, our platform enables accessible regional translation. Furthermore, speech synthesis is structured with a resilient "
        "multi-tier fallback chain (pyttsx3 -> Edge-TTS -> gTTS -> Web Speech API), with Meta's Massively Multilingual Speech (MMS-TTS) "
        "project by Pratap et al. (2024), utilizing the VITS architecture (Kim et al., 2021), implemented in the codebase as an optional "
        "neural waveform generation research service."
    )

    # Table 2: Comparative Literature Review
    add_table_caption(doc, 2, "Comparative Literature Review of State-of-the-Art ISL Recognition Systems")
    table2 = doc.add_table(rows=7, cols=6)
    headers2 = ["Author & Year", "Methodology / Architecture", "Input Modality", "Target Vocabulary", "Reported Performance", "Key Limitations & Gaps"]
    for idx, h in enumerate(headers2):
        table2.rows[0].cells[idx].text = h

    lit_data = [
        ("Subramanian et al. (2022)", "MediaPipe Holistic + Gated Recurrent Unit (MoP-GRU)", "RGB Camera (1662 Skeletal Landmarks)", "13 Dynamic ISL Words", "98.5% on isolated train/test split", "Evaluated on single signer; no fingerspelling alphabet; no translation/speech."),
        ("Adaloglou et al. (2022)", "Comparative 3D-CNN (I3D, ResNet3D) vs Skeletal GCN", "RGB Video & 2D/3D Landmarks", "Continuous Sign Glosses", "Top-1 Acc: 78.4% (I3D), 72.1% (GCN)", "Severe computational overhead; requires discrete high-end GPUs; unusable on CPU."),
        ("Gala et al. (2023)", "IndicTrans2: 200M Distilled Quantized Transformer", "Text Token Sequences (En -> Indic)", "22 Scheduled Indian Languages", "BLEU: 34.2 (En->Hi), 28.5 (En->Ta)", "Text-to-text only; requires tight edge quantization (int8) to run locally on CPU."),
        ("Pratap et al. (2024)", "Meta MMS-TTS: VITS End-to-End Neural Speech", "Phonemic / Text Sequences", "1,000+ Global Languages", "MOS: 4.12; Latency < 60 ms", "Requires PyTorch/SafeTensors runtime optimization and audio buffering for low latency."),
        ("Camgoz et al. (2018)", "Attentive Encoder-Decoder Seq2Seq Gloss Translation", "Continuous Video Frames", "German Sign Language (PHOENIX)", "BLEU-4: 18.13", "Gloss-level vocabulary; struggles with out-of-vocabulary words and names; cloud-bound."),
        ("This Research (2026)", "Dual-Branch Multi-Modal (144D Invariant MLP + 128x128 RGB CNN) + WordBuffer + Dictionary / IndicTrans2 + Multi-Tier TTS", "Webcam RGB + MediaPipe Hands Landmarks (21 per hand) + 128x128 Crop", "26 ISL Letters (1-Hand & 2-Hand) -> 10 Regional Languages + Speech", "78.19% Acc (83.13% with K/T override) / 0.7349 Macro F1 (Held-Out Same-Signer)", "Live multi-signer accuracy pending; camera mirroring coordinate drop (-48.97%); 2-frame hysteresis evaluated on synthetic sequence.")
    ]

    for r_idx, row in enumerate(lit_data, start=1):
        for c_idx, val in enumerate(row):
            table2.rows[r_idx].cells[c_idx].text = val

    format_academic_table(table2, col_widths=[1.2, 1.5, 1.2, 1.1, 1.1, 1.4])

    # Project Feasibility
    add_body_p(
        doc,
        "To establish operational viability before deployment, a rigorous four-dimensional feasibility analysis was conducted:"
    )

    add_bullet_p(
        doc,
        "The MediaPipe Holistic pipeline executes landmark inference in ~18 ms per frame on Apple Silicon (M-series) "
        "and ~28 ms on Intel Core i7 CPUs. The subsequent 144D spatial feature extraction and Dense MLP inference require < 2 ms. "
        "Decoupled asynchronous video streaming via Flask MJPEG and Server-Sent Events (SSE) guarantees that video presentation "
        "and deep inference do not block each other, sustaining 22–27 FPS.",
        bold_prefix="1. Technical Feasibility: "
    )

    add_bullet_p(
        doc,
        "The system exclusively leverages permissive open-source frameworks (TensorFlow/Keras, MediaPipe, HuggingFace ONNX, "
        "Meta MMS under CC-BY-NC/MIT licenses). By utilizing quantized on-device IndicTrans2 and local VITS speech synthesis, "
        "recurring API costs (e.g., Google Cloud Translate or AWS Polly) are exactly $0.00, ensuring sustainable long-term "
        "deployment for non-profit and academic institutions.",
        bold_prefix="2. Economic Feasibility: "
    )

    add_bullet_p(
        doc,
        "The user interacts through a modern, responsive browser interface accessible on laptops and tablets without installing "
        "proprietary client drivers. Standalone PyInstaller packaging allows the software to execute as a single-click native application, "
        "shielding non-technical users from terminal commands, virtual environments, or dependency conflicts.",
        bold_prefix="3. Operational Feasibility: "
    )

    add_bullet_p(
        doc,
        "Total runtime resident memory footprint is strictly bounded under 1.2 GB RAM with all subsystems active "
        "(Inference Engine: ~180 MB, IndicTrans2 ONNX int8: ~340 MB, MMS-TTS VITS: ~450 MB, Web UI: ~80 MB). CPU utilization "
        "remains below 45% during continuous live prediction, ensuring stable thermal and acoustic operation on portable hardware.",
        bold_prefix="4. Resource & Thermal Feasibility: "
    )

    print("Chapter 2 generated successfully.")
