"""Chapter 2 builder: Background Research, Objectives, Literature Review, and Project Feasibility.
"""

from docx import Document
from .xml_helpers import (
    add_heading_1, add_heading_2, add_body_p, add_bullet_p,
    add_table_caption, format_academic_table
)


def build_chapter2(doc: Document):
    """Generate Chapter 2: Background Research, Objectives, Literature Review, and Feasibility."""
    
    # 2. BACKGROUND RESEARCH (Style GS1 provides automatic chapter numbering)
    add_heading_1(doc, "BACKGROUND RESEARCH")

    add_body_p(
        doc,
        "Sign Language Recognition (SLR) is an inherently interdisciplinary computational domain spanning computer vision, "
        "kinematics, deep neural representations, and natural language processing. In the Indian context, "
        "computational research has faced challenges due to a historical scarcity of standardized, "
        "publicly annotated video corpora, regional dialectal divergence, and the intrinsic complexity of bimanual "
        "fingerspelling. This chapter establishes the engineering objectives of our platform, delivers a comparative "
        "literature review of vision and translation architectures, and presents a multi-dimensional "
        "feasibility assessment governing on-device edge execution."
    )

    # 2.1 Objectives (Style GS2 provides automatic section numbering)
    add_heading_2(doc, "Objectives")

    add_body_p(
        doc,
        "To address the limitations of prior systems, this project establishes a concrete matrix of functional, "
        "computational, and accessibility objectives. These objectives transition the software from an isolated "
        "classifier into an integrated assistive communication platform. The specific, quantifiable "
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
        ("G-06", "Speech Output", "Synthesize audible speech corresponding to translated regional text using multi-tier fallback.", "Sub-second response using macOS say, pyttsx3, gTTS, or browser Web Speech API; optional Meta MMS-TTS module.")
    ]

    for r_idx, row in enumerate(goals_data, start=1):
        for c_idx, val in enumerate(row):
            table1.rows[r_idx].cells[c_idx].text = val

    format_academic_table(table1, col_widths=[0.8, 1.4, 2.8, 2.0])

    # 2.2 Literature Review & Project Feasibility (Style GS2 provides automatic section numbering)
    add_heading_2(doc, "Literature Review & Project Feasibility")

    add_body_p(
        doc,
        "A comprehensive review of modern sign language literature reveals an evolution from sensor-laden hardware "
        "toward non-invasive deep learning vision pipelines. Early works relied on colored gloves or Microsoft Kinect depth sensors "
        "to resolve finger occlusions. While depth sensors simplify spatial segmentation, their commercial discontinuation "
        "and reliance on specialized infrared emitters limited ubiquitous consumer deployment. The advent "
        "of high-speed perception frameworks—most notably Google MediaPipe by Lugaresi et al. (2019) and Zhang et al. (2020)—"
        "advanced the field by enabling 2.5D landmark extraction directly from RGB frames on low-power mobile and CPU edge devices."
    )

    add_body_p(
        doc,
        "Sign Language Recognition systems typically bifurcate into Word-Level Continuous Recognition and Isolated "
        "Alphabet Fingerspelling. In word-level recognition, models classify predefined full-word gestures (e.g., 'Hello', 'Help', 'Water'). "
        "The foundation of our initial exploratory phase was inspired by the MediaPipe-Optimized Gated Recurrent Unit (MoP-GRU) architecture "
        "by Barathi Subramanian et al. (2022). In accordance with the MIT License, our work preserves the original copyright notice. "
        "Subramanian et al. demonstrated that recurrent cells could classify dynamic temporal trajectories from skeletal sequences "
        "(preserved in legacy/mopgru_cell.py, legacy/collect_keypoints.py, legacy/preprocess_keypoints.py, legacy/realtime_inference.py, "
        "and legacy/train_word_model.py; the original word-level checkpoints are kept locally and not in the repository). "
        "However, whole-sign models suffer from limited vocabulary coverage (typically 10–50 signs). In real-world assistive "
        "scenarios, signers frequently communicate proper nouns, personal names, locations, technical terminology, and open-ended "
        "phrases that do not exist in fixed vocabularies. This realization motivated our technical transition to the 26-letter "
        "ISL fingerspelling alphabet."
    )

    add_body_p(
        doc,
        "In alphabet fingerspelling, literature has predominantly explored two parallel paradigms: Appearance-based 2D/3D CNNs "
        "and Geometric Landmark Multi-Layer Perceptrons (MLPs). Purely visual 2D CNNs (e.g., VGG, ResNet, MobileNet) operate directly "
        "on cropped bounding boxes of the hands. While effective at capturing subtle surface textures and palm orientations, CNNs "
        "exhibit severe sensitivity to lighting changes, skin tone variations, background clutter, and camera perspective shifts. "
        "Conversely, skeletal pipelines (extracting coordinates via MediaPipe Hands) are invariant to skin color, illumination, "
        "and clothing, but discard rich texture cues. While 3D-CNNs (e.g., I3D by Carreira & Zisserman, 2017) "
        "capture fine-grained appearance, they suffer from high computational complexity (>100 GFLOPs), precluding real-time "
        "execution on commodity CPU hardware without dedicated graphics acceleration."
    )

    add_body_p(
        doc,
        "On the downstream linguistic side, translating transcribed sign text into Indian languages presents substantial challenges "
        "due to grammatical divergences between Indian Sign Language and spoken Indic vernaculars. Spoken Indian languages feature "
        "Subject-Object-Verb (SOV) sentence structuring and rich agglutinative morphology. Recent work "
        "from AI4Bharat on IndicTrans2 (Gala et al., 2023) has established benchmark quality across all 22 scheduled Indian languages "
        "using a 200M parameter distilled Transformer architecture. For speech synthesis, Meta's Massively Multilingual Speech (MMS-TTS) "
        "project (Pratap et al., 2024) demonstrated high-fidelity Text-to-Speech synthesis across 1,107 languages using the VITS "
        "end-to-end neural architecture. Our project implements a hybrid approach: instantaneous local dictionary lookup for core everyday phrases, "
        "with online translation API fallback, while IndicTrans2 and Meta MMS-TTS are implemented in src/language_services as optional "
        "offline neural research modules."
    )

    # Table 2: Comparative Literature Review
    add_table_caption(doc, 2, "Comparative Literature Review of Existing ISL Recognition Systems")
    table2 = doc.add_table(rows=7, cols=6)
    headers2 = ["Author & Year", "Methodology / Architecture", "Input Modality", "Target Vocabulary", "Reported Performance", "Key Limitations & Gaps"]
    for idx, h in enumerate(headers2):
        table2.rows[0].cells[idx].text = h

    lit_data = [
        ("Subramanian et al. (2022)", "MediaPipe Holistic + Gated Recurrent Unit (MoP-GRU)", "RGB Camera (Skeletal Landmarks)", "13 Dynamic ISL Words", "R^2: 0.88; class F1 up to 0.95", "Evaluated on single signer; no fingerspelling alphabet; no translation or speech output."),
        ("Adaloglou et al. (2022)", "Comparative 3D-CNN (I3D, ResNet3D) vs Skeletal GCN", "RGB Video & 2D/3D Landmarks", "Continuous Sign Glosses", "WER: 22.1%–38.6% across continuous benchmarks", "High computational overhead; requires discrete high-end GPUs; unusable on commodity CPU."),
        ("Gala et al. (2023)", "IndicTrans2: 200M Distilled Quantized Transformer", "Text Token Sequences (En -> Indic)", "22 Scheduled Indian Languages", "chrF++: 58.2, BLEU: 31.8 (En->Hi on IN22-Gen)", "Text-to-text only; requires tight edge quantization (int8) to run locally on CPU."),
        ("Pratap et al. (2024)", "Meta MMS-TTS: VITS End-to-End Neural Speech", "Phonemic / Text Sequences", "1,000+ Global Languages", "MOS: ~3.9–4.0 naturalness", "Requires PyTorch/SafeTensors runtime optimization and audio buffering for low latency."),
        ("Camgoz et al. (2018)", "Attentive Encoder-Decoder Seq2Seq Gloss Translation", "Continuous Video Frames", "German Sign Language (PHOENIX-2014T)", "BLEU-4: 18.13 (gloss-to-text)", "Gloss-level vocabulary; struggles with out-of-vocabulary words and names; cloud-bound."),
        ("This Research (2026)", "Dual-Branch Multi-Modal (144D Invariant MLP + 128x128 RGB CNN) + WordBuffer + Dictionary / IndicTrans2 + Multi-Tier TTS", "Webcam RGB + MediaPipe Hands Landmarks (21 per hand) + 128x128 Crop", "26 ISL Letters (1-Hand & 2-Hand) -> 10 Regional Languages + Speech", "78.19% Top-1 Acc (83.13% with K/T override*) / 0.7349 Macro F1 (Held-Out Same-Signer)", "Live multi-signer accuracy pending; camera mirroring coordinate drop (-48.97%); 2-frame hysteresis evaluated on synthetic sequence. *K/T override possibly tuned on test data.")
    ]

    for r_idx, row in enumerate(lit_data, start=1):
        for c_idx, val in enumerate(row):
            table2.rows[r_idx].cells[c_idx].text = val

    format_academic_table(table2, col_widths=[1.2, 1.5, 1.2, 1.1, 1.1, 1.4])

    # Project Feasibility
    add_body_p(
        doc,
        "To establish operational viability before deployment, a four-dimensional feasibility analysis was conducted:"
    )

    add_bullet_p(
        doc,
        "The MediaPipe Hands pipeline executes landmark tracking in 13.23 ms (empty frame) to 14.84 ms on Apple Silicon M2 (CPU) "
        "as measured by src/benchmark_latency.py. The subsequent 144D spatial feature extraction and Dense MLP inference require < 3 ms. "
        "Decoupled asynchronous video streaming via Flask MJPEG and Server-Sent Events (SSE) guarantees that video presentation "
        "and model inference do not block each other, sustaining 22.6 FPS on active dual-hand frames.",
        bold_prefix="Technical Feasibility: "
    )

    add_bullet_p(
        doc,
        "The system exclusively leverages permissive open-source frameworks (TensorFlow/Keras, MediaPipe, HuggingFace ONNX, "
        "Meta MMS under CC-BY-NC/MIT licenses). By utilizing local dictionary translation and offline multi-tier speech synthesis, "
        "recurring API costs (e.g., Google Cloud Translate or AWS Polly) are exactly $0.00, ensuring sustainable long-term "
        "deployment for non-profit and academic institutions.",
        bold_prefix="Economic Feasibility: "
    )

    add_bullet_p(
        doc,
        "The user interacts through a modern, responsive browser interface accessible on laptops and tablets without installing "
        "proprietary client drivers. Standalone packaging allows the software to execute as a single-click native application, "
        "shielding non-technical users from terminal commands, virtual environments, or dependency conflicts.",
        bold_prefix="Operational Feasibility: "
    )

    add_bullet_p(
        doc,
        "The core vision pipeline is engineered to operate within commodity laptop hardware resources without requiring discrete GPUs. "
        "Independent benchmarking using src/benchmark_latency.py confirms that the entire processing loop sustains real-time interactive "
        "speeds on CPU while maintaining responsiveness across active camera sessions.",
        bold_prefix="Resource & Thermal Feasibility: "
    )

    print("Chapter 2 generated successfully.")
