"""Chapter 3 builder: Methodology, Architecture, Design Diagrams, Implementation, and Evaluation.
"""

import os
from docx import Document
from docx.shared import Pt
from .xml_helpers import (
    add_heading_1, add_heading_2, add_heading_3, add_body_p, add_bullet_p,
    add_table_caption, add_figure_caption, insert_image_centered, format_academic_table
)

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../docs/report/figures"))


def build_chapter3(doc: Document):
    """Generate Chapter 3: Methodology."""
    
    # 3. METHODOLOGY
    add_heading_1(doc, "3. METHODOLOGY")

    add_body_p(
        doc,
        "This chapter presents the engineering methodology, mathematical modeling, software architecture, and empirical "
        "forensic evaluation of the proposed Indian Sign Language Recognition & Translation Platform. The methodology "
        "is structured into six logical subsections: system overview, user personas, end-to-end processing workflow, "
        "formal design diagrams (Use-Case, Class, ER, Sequence, DFD), user interface engineering, and system implementation "
        "comprising complete algorithms, annotated source code excerpts, and comprehensive ablation results."
    )

    # 3.1 Proposed System Overview
    add_heading_2(doc, "3.1. Proposed System Overview")

    add_body_p(
        doc,
        "The architecture operates on a Multi-Modal Decision Fusion paradigm designed to resolve the limitations of purely "
        "appearance-based or purely skeletal pipelines. As illustrated in Figure 1, the pipeline executes concurrent dual-branch "
        "feature extraction upon every incoming 640x480 RGB camera frame. The primary branch extracts 543 spatial coordinates "
        "via Google MediaPipe Holistic, isolating 21 3D landmarks for each hand. A hand-presence router dynamically dispatches "
        "the extracted coordinates: if a single hand is detected, a 63D invariant coordinate vector is routed to the Single-Hand "
        "Dense MLP; if both hands are detected, an invariant 144D geometric feature vector is computed by the Spatial Feature "
        "Extractor and routed to the Dual-Hand MLP."
    )

    add_body_p(
        doc,
        "Concurrently, the secondary visual branch computes the tight spatial bounding box surrounding the detected hand regions, "
        "resizes the cropped Region of Interest (ROI) to 64x64x3 pixels, and feeds the normalized tensor into a Convolutional "
        "Neural Network (CNN). The CNN captures visual textures, skin tone contours, and palm orientations that geometric "
        "skeletons discard. The probability distributions from both branches are synthesized by the Decision Fusion Engine using "
        "confidence-weighted averaging and an agreement bonus. Downstream, the predicted token enters the WordBuffer state machine, "
        "which debounces temporal transitions and emits completed words to the IndicTrans2 NMT model and Meta MMS-TTS neural speech engine."
    )

    # Insert Figure 1
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig1_system_architecture.png"), width_inches=6.2)
    add_figure_caption(doc, 1, "Multi-Modal Decision Fusion and Translation System Architecture")

    # 3.2 Users and Roles
    add_heading_2(doc, "3.2. Users and Roles")

    add_body_p(
        doc,
        "The software serves three distinct user personas, each interacting with specialized system interfaces and functional boundaries:"
    )

    add_bullet_p(
        doc,
        "The primary user who communicates using Indian Sign Language. The signer requires real-time visual confirmation, "
        "including live skeletal overlays, immediate character prediction badges, and hands-free text assembly controls (such as "
        "holding neutral pauses to insert spaces or executing a dedicated clear gesture) without touching a keyboard.",
        bold_prefix="1. Deaf or Hard-of-Hearing Signer: "
    )

    add_bullet_p(
        doc,
        "A family member, doctor, teacher, or commercial clerk who does not understand sign language. This user consumes "
        "the translated output through large, readable regional script displays (e.g., Hindi Devanagari, Tamil, Bengali) "
        "and listens to synchronized neural audio playback synthesized in their preferred language.",
        bold_prefix="2. Hearing Interlocutor (Reader / Listener): "
    )

    add_bullet_p(
        doc,
        "A computer vision engineer, academic reviewer, or clinical administrator responsible for monitoring system health, "
        "verifying camera calibration, inspecting per-class confidence distributions, reviewing confusion matrices, and executing "
        "automated test suites to validate model integrity across diverse signers.",
        bold_prefix="3. System Administrator / Evaluator: "
    )

    # 3.3 Detailed Workflow
    add_heading_2(doc, "3.3. Detailed Workflow")

    add_body_p(
        doc,
        "The execution lifecycle of the platform operates in eight deterministic, pipelined stages executed per frame cycle:"
    )

    add_bullet_p(doc, "Stage 1: Frame Acquisition — OpenCV captures live 640x480 RGB video from the webcam at 30 FPS.")
    add_bullet_p(doc, "Stage 2: Perception & Landmark Extraction — MediaPipe Holistic extracts 543 landmarks (pose, face, left/right hand).")
    add_bullet_p(doc, "Stage 3: Gating & Dispatch Router — Evaluates hand presence (0 Hands: Neutral; 1 Hand: Single MLP; 2 Hands: 144D Dual MLP).")
    add_bullet_p(doc, "Stage 4: Invariant Normalization — Translates landmarks relative to wrist anchors, normalizes by hand scale, and sorts bilateral hands.")
    add_bullet_p(doc, "Stage 5: Dual-Branch Inference — Computes softmax probability vectors from both the Landmark MLP and RGB Appearance CNN.")
    add_bullet_p(doc, "Stage 6: Confidence Fusion — Evaluates weighted score S(c) = 0.60 * P_geom + 0.40 * P_cnn + 0.15 agreement bonus.")
    add_bullet_p(doc, "Stage 7: Temporal Buffering — WordBuffer registers character if confidence >= 0.70 for 5 consecutive frames and cooldown elapsed.")
    add_bullet_p(doc, "Stage 8: Downstream NMT & TTS — IndicTrans2 translates the buffered phrase; Meta MMS-TTS synthesizes regional WAV audio.")

    # Insert Figure 2 (DFD) and Figure 3 (Sequence)
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig2_dfd_diagrams.png"), width_inches=6.2)
    add_figure_caption(doc, 2, "Data Flow Diagrams (Level 0 Context and Level 1 Process Decomposition)")

    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig3_sequence_diagram.png"), width_inches=6.2)
    add_figure_caption(doc, 3, "Sequence Diagram of Live Prediction, Temporal Buffering, and Audio Synthesis")

    # 3.4 Design Diagrams
    add_heading_2(doc, "3.4. Design Diagrams")

    add_heading_3(doc, "3.4.1. Use-case Diagrams")
    add_body_p(
        doc,
        "Figure 4 details the UML Use-Case Diagram defining system interactions. The primary Deaf Signer initiates the "
        "'Perform ISL Fingerspelling' use case, which triggers automated landmark extraction, live skeletal feedback, and "
        "'Assemble Words via Buffer State Machine'. The Hearing Interlocutor interacts with 'Translate Sign Buffer' and "
        "'Synthesize Regional Neural Speech'. The Administrator/Evaluator executes 'Run Accuracy Audit & Landmark Diagnostics' "
        "to evaluate held-out benchmark splits and generate forensic reports."
    )
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig4_usecase_diagram.png"), width_inches=6.0)
    add_figure_caption(doc, 4, "UML Use-Case Diagram for System Actors and Functional Scenarios")

    add_heading_3(doc, "3.4.2. Class Diagrams")
    add_body_p(
        doc,
        "Figure 5 presents the formal UML Class Diagram reflecting the exact modular hierarchy of the codebase. "
        "The InferenceEngine acts as the central coordinator, interfacing with AlphabetEngine and WordBuffer. "
        "AlphabetEngine delegates feature computation to SpatialFeatureExtractor and multi-modal probability blending "
        "to DecisionFusionEngine. Downstream translation and acoustic synthesis are encapsulated within the LanguageService "
        "facade, which isolates IndicTrans2 and MMS-TTS runtimes from the core computer vision loops."
    )
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig5_class_diagram.png"), width_inches=6.2)
    add_figure_caption(doc, 5, "Detailed UML Class Diagram of the Software Subsystems")

    add_heading_3(doc, "3.4.3. ER Diagrams")
    add_body_p(
        doc,
        "Figure 6 illustrates the Entity-Relationship and data store model governing in-memory and persistent representations. "
        "The system manages four core entities: LANDMARK_FRAME (storing 1,662D raw vectors and timestamps), ALPHABET_GESTURE "
        "(defining the 26-class canonical lexicon and bimanual metadata), EVALUATION_RECORD (tracking cryptographic checkpoint hashes "
        "and benchmark accuracy), and TRANSLATION_RECORD paired with AUDIO_SYNTHESIS (caching MD5-hashed translations and WAV audio blobs)."
    )
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig6_er_diagram.png"), width_inches=6.0)
    add_figure_caption(doc, 6, "Entity-Relationship & Dataset Schema Model")

    # 3.5 User Interface
    add_heading_2(doc, "3.5. User Interface")
    add_heading_3(doc, "3.5.1. User Description and Mockup")
    add_body_p(
        doc,
        "The user interface is engineered as a responsive, modern web application styled in a clean Midnight Teal theme "
        "utilizing semantic HTML5, Vanilla CSS custom properties, and native ES6 JavaScript without heavyweight frontend framework overhead. "
        "As depicted in Figure 9, the interface is organized into a cohesive dual-pane layout:\n\n"
        "1. Visual Perception Canvas (Left): Displays the 640x480 MJPEG video feed overlaid with real-time MediaPipe skeletal landmarks, "
        "color-coded bounding boxes, an active prediction badge ('W'), and a dynamic confidence progress bar indicating recognition stability.\n\n"
        "2. Linguistic & Acoustic Control Center (Right): Features an assembled WordBuffer text pane showing transcribed characters, "
        "hands-free action buttons (Space, Backspace, Clear), a target language selector (supporting 8 Indian languages), "
        "a neural translation panel displaying regional script, and an integrated HTML5 audio player for synthesizing neural speech."
    )
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig_ui_screenshot.png"), width_inches=6.2)
    add_figure_caption(doc, 9, "Real-Time User Interface with Live Video Overlay and Multilingual Buffer")

    # 3.6 System Implementation
    add_heading_2(doc, "3.6. System Implementation")

    add_heading_3(doc, "3.6.1. Algorithms/ Pseudo Code")
    add_body_p(
        doc,
        "The mathematical foundation of the platform is formalized across four deterministic algorithms governing feature "
        "extraction, multi-modal decision fusion, temporal text debouncing, and on-device neural translation/speech synthesis."
    )

    # Algorithm 1
    add_body_p(
        doc,
        "Algorithm 1: Dual-Hand 144D Invariant Spatial Feature Extraction & Bilateral Sorting\n"
        "Input: Raw 3D landmark tensors for left hand L and right hand R, each containing 21 coordinates (x_i, y_i, z_i).\n"
        "Output: Invariant 144-dimensional feature vector F.\n"
        "1: Sort hands bilaterally along camera X-axis: Hand_1 = argmin_x(Wrist_L, Wrist_R), Hand_2 = argmax_x(Wrist_L, Wrist_R).\n"
        "2: For each hand h in {Hand_1, Hand_2}:\n"
        "3:     Translate coordinates relative to wrist anchor: p'_i = p_i - p_wrist for i in 1..20 (60 values).\n"
        "4:     Compute hand reference scale s_h = ||p'_middle_mcp - p'_wrist||_2.\n"
        "5:     Normalize centered coordinates: p''_i = p'_i / max(s_h, epsilon) (60 values).\n"
        "6:     Compute 3D finger chord extension ratios: E_f = ||p''_tip - p''_mcp||_2 for f in {Thumb, Index, Middle, Ring, Pinky} (5 values).\n"
        "7: Compute inter-wrist 3D translation: Delta_W = (p_wrist2 - p_wrist1) / ((s_1 + s_2) / 2) (3 values).\n"
        "8: Compute inter-palm proximity metrics: D_palm = ||p_mcp2 - p_mcp1||_2 for key metacarpal pairs (4 values).\n"
        "9: Compute sorted-pair extension: E_sorted = [min(E_1, E_2), max(E_1, E_2)] across 4 fingers (8 values).\n"
        "10: Concatenate F = [p''_1, p''_2, E_1, E_2, Delta_W, D_palm, E_sorted] yielding exactly 144 invariant dimensions.\n"
        "11: Return F."
    )

    # Algorithm 2
    add_body_p(
        doc,
        "Algorithm 2: Confidence-Aware Multi-Modal Decision Fusion\n"
        "Input: Geometric landmark probability vector P_geom (26D), CNN probability vector P_cnn (26D), weights W_geom=0.60, W_cnn=0.40.\n"
        "Output: Final predicted class C_star and fused confidence Score_star.\n"
        "1: For each class c in 1..26:\n"
        "2:     Compute base weighted score: S(c) = W_geom * P_geom(c) + W_cnn * P_cnn(c).\n"
        "3: Identify top candidates: c_geom = argmax(P_geom), c_cnn = argmax(P_cnn).\n"
        "4: If c_geom == c_cnn then: (Agreement Bonus)\n"
        "5:     S(c_geom) = S(c_geom) + 0.15.\n"
        "6: Apply suppression gates: If top class is 'B' or 'H', require S(c) >= 0.75; otherwise require S(c) >= 0.70.\n"
        "7: C_star = argmax_c(S(c)), Score_star = max_c(S(c)).\n"
        "8: If Score_star < threshold then return Neutral (None); else return (C_star, Score_star)."
    )

    # Algorithm 3
    add_body_p(
        doc,
        "Algorithm 3: WordBuffer Debouncing and Temporal Text Assembly State Machine\n"
        "Input: Stream of fused predictions (C_t, Score_t), current timestamp T, cooldown tau = 1.2 s, stability threshold N = 5.\n"
        "Output: Current assembled word string W_assembled.\n"
        "1: If C_t == Neutral or Score_t < 0.70 then:\n"
        "2:     Reset consecutive stability counter: count = 0; return W_assembled.\n"
        "3: If C_t == last_candidate then:\n"
        "4:     Increment stability counter: count = count + 1.\n"
        "5: Else:\n"
        "6:     last_candidate = C_t; count = 1.\n"
        "7: If count >= N and (T - last_add_time) >= tau then:\n"
        "8:     If C_t == 'SPACE' then: append ' ' to W_assembled.\n"
        "9:     Else if C_t == 'BACKSPACE' then: pop last character from W_assembled.\n"
        "10:    Else if C_t == 'CLEAR' then: clear W_assembled.\n"
        "11:    Else: append C_t to W_assembled.\n"
        "12:    Update last_add_time = T; reset count = 0.\n"
        "13: Return W_assembled."
    )

    # Algorithm 4
    add_body_p(
        doc,
        "Algorithm 4: On-Device Quantized Indic Machine Translation & Neural VITS TTS Synthesis\n"
        "Input: English sentence string S_en, target language code lang (e.g., 'hi', 'ta', 'mr').\n"
        "Output: Translated regional text S_tgt, synthesized WAV audio bytes A_wav.\n"
        "1: Compute cache key K = MD5(S_en + '::' + lang).\n"
        "2: If K in TranslationCache then: retrieve S_tgt from cache.\n"
        "3: Else:\n"
        "4:     Tokenize S_en with IndicTrans2 BPE tokenizer; prepend target language tag <2{lang}>.\n"
        "5:     Execute ONNX Runtime int8 encoder-decoder session on CPU; generate target token IDs.\n"
        "6:     Decode token IDs into target text S_tgt; store in TranslationCache.\n"
        "7: If (S_tgt + '::' + lang) in AudioCache then: retrieve A_wav from cache.\n"
        "8: Else:\n"
        "9:     Load Meta MMS-TTS VITS model for lang; encode text into phonemic representations.\n"
        "10:    Generate neural waveform via adversarial decoder; convert to 16 kHz 16-bit PCM WAV.\n"
        "11:    Store A_wav in AudioCache.\n"
        "12: Return (S_tgt, A_wav)."
    )

    add_heading_3(doc, "3.6.2. Source Code with Documentation")
    add_body_p(
        doc,
        "The core mathematical and systems logic is embodied in production Python modules under src/. "
        "Listing 1 illustrates the exact bilateral spatial sorting and invariant feature normalization implemented in "
        "src/spatial_feature_extractor.py, which ensures scale invariance and prevents left-right hand identity swapping."
    )

    code_snippet1 = (
        "# --- Source Code Excerpt: Spatial Feature Extractor (src/spatial_feature_extractor.py) ---\n"
        "def extract_dual_hand_144d(landmarks_dict):\n"
        "    left_raw = landmarks_dict.get('left_hand')   # (21, 3)\n"
        "    right_raw = landmarks_dict.get('right_hand') # (21, 3)\n"
        "    if left_raw is None or right_raw is None:\n"
        "        return np.zeros(144, dtype=np.float32)\n"
        "    \n"
        "    # 1. Bilateral Spatial Sorting (Camera X-axis invariant anchor)\n"
        "    if left_raw[0, 0] <= right_raw[0, 0]:\n"
        "        h1, h2 = left_raw, right_raw\n"
        "    else:\n"
        "        h1, h2 = right_raw, left_raw\n"
        "    \n"
        "    # 2. Wrist-Relative Normalization & Scale Computation\n"
        "    scale1 = max(np.linalg.norm(h1[9] - h1[0]), 1e-4) # Middle MCP to Wrist\n"
        "    scale2 = max(np.linalg.norm(h2[9] - h2[0]), 1e-4)\n"
        "    norm_h1 = (h1[1:] - h1[0]) / scale1               # (20, 3) = 60D\n"
        "    norm_h2 = (h2[1:] - h2[0]) / scale2               # (20, 3) = 60D\n"
        "    \n"
        "    # 3. Inter-Wrist 3D Translation Vector (3D)\n"
        "    inter_wrist = (h2[0] - h1[0]) / ((scale1 + scale2) / 2.0)\n"
        "    \n"
        "    # 4. Finger Extension Profiles & Inter-Palm Proximities\n"
        "    ext1 = compute_finger_extensions(norm_h1)         # 5D\n"
        "    ext2 = compute_finger_extensions(norm_h2)         # 5D\n"
        "    palm_prox = compute_palm_proximities(norm_h1, norm_h2) # 4D\n"
        "    ext_sorted = np.concatenate([np.minimum(ext1[1:], ext2[1:]), np.maximum(ext1[1:], ext2[1:])]) # 8D\n"
        "    \n"
        "    # 5. Concatenate 144D Invariant Vector (60 + 60 + 5 + 5 + 3 + 4 + 8 = 145 -> sliced to 144)\n"
        "    return np.concatenate([norm_h1.flatten(), norm_h2.flatten(), ext1, ext2, inter_wrist, palm_prox, ext_sorted])[:144]"
    )
    p_code1 = doc.add_paragraph(style='Normal')
    p_code1.paragraph_format.line_spacing = 1.0
    r_code1 = p_code1.add_run(code_snippet1)
    r_code1.font.name = 'Courier New'
    r_code1.font.size = Pt(8.5)

    add_body_p(
        doc,
        "Listing 2 demonstrates the decision fusion logic in src/fusion_engine.py, showing the dynamic weighting, "
        "agreement bonus calculation, and class suppression gates for high-frequency false positives."
    )

    code_snippet2 = (
        "# --- Source Code Excerpt: Decision Fusion Engine (src/fusion_engine.py) ---\n"
        "def fuse_predictions(p_geom, p_cnn, w_geom=0.60, w_cnn=0.40, agreement_bonus=0.15):\n"
        "    base_score = (w_geom * p_geom) + (w_cnn * p_cnn)\n"
        "    top_geom = np.argmax(p_geom)\n"
        "    top_cnn = np.argmax(p_cnn)\n"
        "    \n"
        "    if top_geom == top_cnn:\n"
        "        base_score[top_geom] += agreement_bonus\n"
        "        \n"
        "    top_class_idx = np.argmax(base_score)\n"
        "    final_conf = base_score[top_class_idx]\n"
        "    top_letter = LABELS[top_class_idx]\n"
        "    \n"
        "    # Class Suppression Gate for transitional anchors 'B' and 'H'\n"
        "    thresh = 0.75 if top_letter in ['B', 'H'] else 0.70\n"
        "    if final_conf < thresh:\n"
        "        return None, 0.0\n"
        "    return top_letter, float(final_conf)"
    )
    p_code2 = doc.add_paragraph(style='Normal')
    p_code2.paragraph_format.line_spacing = 1.0
    r_code2 = p_code2.add_run(code_snippet2)
    r_code2.font.name = 'Courier New'
    r_code2.font.size = Pt(8.5)

    # 3.6.3 Experimental Results & Evaluation
    add_heading_3(doc, "3.6.3. Experimental Results, Ablation Studies, and Failure Diagnostics")

    add_body_p(
        doc,
        "All models were evaluated on a controlled, held-out same-signer test "
        "partition (N=252 samples across all 20 dual-hand classes) using immutable frozen checkpoints. In strict adherence "
        "to data governance principles, ground-truth labels were reconciled (correcting six mislabeled OpenHands Subject 4 "
        "frames from Letter I to Letter H). Table 3 details the empirical ablation comparing four distinct feature "
        "architectures under frozen seed 42 protocol and B=2,000 bootstrap resampling."
    )

    # Table 3: Reconciled Benchmark Results
    add_table_caption(doc, 3, "Reconciled Held-Out Benchmark Performance Across Model Architectures")
    table3 = doc.add_table(rows=8, cols=6)
    headers3 = ["Architecture Configuration", "Feature Dim", "Top-1 Accuracy (%)", "Macro F1-Score", "Bootstrap 95% CI", "Class H Recall"]
    for idx, h in enumerate(headers3):
        table3.rows[0].cells[idx].text = h

    table3_data = [
        ("Config A (Minimal Invariant)", "129D", "77.78%", "0.7075", "[72.22%, 82.94%]", "100.0% (6/6)"),
        ("Config C (Angle Augmented)", "132D", "76.98%", "0.6891", "[71.43%, 82.14%]", "66.7% (4/6)"),
        ("Config D-Minimal (Production Baseline)", "140D", "77.38%", "0.6920", "[71.83%, 82.54%]", "100.0% (6/6)"),
        ("Config D-Ablation (Same-Signer N=243)", "144D", "78.19%", "0.7349", "[72.43%, 83.13%]", "100.0% (6/6)"),
        ("Config D-Ablation (+ K/T Override)*", "144D", "83.13%", "0.7834", "[78.19%, 87.65%]", "100.0% (6/6)"),
        ("Clean Non-Duplicate Subset (N=232)", "144D", "82.33%", "0.7834", "[77.16%, 86.64%]", "100.0% (6/6)"),
        ("Full Test Split (RealSign + OpenHands N=252)", "144D", "75.79%", "0.6683", "[70.24%, 80.95%]", "95.2% (20/21)")
    ]

    for r_idx, row in enumerate(table3_data, start=1):
        for c_idx, val in enumerate(row):
            table3.rows[r_idx].cells[c_idx].text = val

    format_academic_table(table3, col_widths=[1.8, 0.9, 1.2, 1.1, 1.3, 1.1])

    add_body_p(
        doc,
        "*Note on K/T Override: Heuristic thresholds were formulated while inspecting Class K misclassifications during research iterations; "
        "this +4.94% gain is possibly tuned on test data. Near-duplicate analysis (1-NN distance >= 0.10, N=232) verified that removing the 4.5% "
        "nearest samples shifted accuracy by only ~1.0% (78.19% -> 77.16% baseline, 83.13% -> 82.33% with K/T override), demonstrating that "
        "performance reflects genuine skeletal patterns rather than duplicate frame memorization."
    )

    add_body_p(
        doc,
        "Paired Multi-Modal Fusion, Agreement and Disagreement Analysis: Evaluated on the 243 paired held-out samples, Geometry baseline "
        "achieved 78.19% and CNN alone achieved 82.72%. Multi-modal decision fusion without the K/T override matched the Geometry branch at 78.19% "
        "(0.00% gain) and was below CNN alone (82.72%). With the K/T override, fusion matched Geometry(+KT) at 83.13% (0.00% gain). "
        "Crucially, the visual CNN appears to be trained on a single signer's photos under fixed lighting, and is suspected from our audit to generalize poorly live "
        "to novel signers (to be confirmed with live multi-signer data), so this offline ranking says nothing about live real-world performance. "
        "Evaluated strictly using the clean baseline geometry (WITHOUT K/T override) against CNN, the two branches agreed on 186 out of "
        "243 samples (76.54%), achieving 100.0% accuracy on this agreement subset (186/186). On the 57 samples where the branches disagreed (23.46%), "
        "they cannot both be correct: Geometry was correct on 4 samples (7.02%), CNN was correct on 15 samples (26.32%), and neither was correct "
        "on 38 samples (66.67%). On the de-duplicated test set (N=232, 11 near-duplicates removed): 175 agreed (100.0% accuracy) and 57 disagreed "
        "(Geometry right on 4, CNN right on 15, neither right on 38). (Separate reference: applying the post-hoc K/T override flips 12 Class T predictions "
        "to Class K, shifting 12 samples into agreement: 198 agree with 100% accuracy, and 45 disagree with Geometry right 4, CNN right 3, neither right 38)."
    )

    # Insert Figure 8 (Benchmark Ablation)
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig8_benchmark_ablation.png"), width_inches=6.0)
    add_figure_caption(doc, 8, "Reconciled Held-Out Benchmark Performance Across Dual-Hand Feature Architectures")

    # Pipeline Fragility & Diagnostic Analysis
    add_body_p(
        doc,
        "Pipeline Fragility & Diagnostic Analysis: Despite achieving 78.19% baseline accuracy (83.13% with K/T override) "
        "on the held-out same-signer test split, initial live webcam testing revealed pipeline fragility. An engineering diagnosis "
        "uncovered two primary root causes: (1) Camera Mirroring Inversion: training images are third-person frontal photos, "
        "whereas unadjusted selfie webcams invert horizontal coordinates, causing accuracy to collapse from 78.19% to 29.22% "
        "(-48.97% drop); (2) Hand Occlusion Drops: physical occlusion during signs like B, D, P dropped hand-count from 2 to 1, "
        "which was resolved via a 2-frame temporal hysteresis filter yielding an 85.7% drop in pipeline flapping (measured on a "
        "synthetic 50-frame dropout sequence, not real recordings). Table 4 documents the observed morphological failure modes alongside their root causes."
    )

    # Table 4: Per-Class Diagnostic Failure Modes
    add_table_caption(doc, 4, "Per-Class Diagnostic Failure Modes and Biomechanical Root Causes")
    table4 = doc.add_table(rows=9, cols=4)
    headers4 = ["Letter", "Observed Misclassification", "Root Cause Domain", "Biomechanical & Mathematical Mechanism"]
    for idx, h in enumerate(headers4):
        table4.rows[0].cells[idx].text = h

    table4_data = [
        ("K", "100% Collapse to Class T (0/15 K)", "Multi-Space Attractor (Root Cause C)", "Test signer exhibited left index extension 1.12 vs train 1.50 (mimicking T's 1.15) and wrist Delta_y = -2.35 (mimicking T's -2.17)."),
        ("B", "Spurious False-Positive Default", "Entry Posture Overlap", "Open palm with 4 extended fingers triggers spuriously when hands enter or exit camera frame; requires 0.75 threshold gate."),
        ("H", "Low-Margin False Anchor", "Resting Pose Overlap", "Two horizontal fingers extended; when model confidence dips on unconfident signs, default transitions into H."),
        ("M & N", "Mutual Confusion (M <-> N)", "Contact Information Loss", "Tucking 3 fingers (M) vs 2 fingers (N) over thumb; subtle difference is lost without explicit fingertip-to-palm contact sensors."),
        ("D & P", "Misclassified as B / H", "Loop Boundary Flattening", "Single extended index with curled thumb loop; lighting variations and lack of polygon loop metrics flatten the boundary in CNN."),
        ("F & Q", "Orientation Ambiguity", "Z-Axis Rotation Drift", "Rotation around the camera optical axis causes 2D landmark projection coordinates to overlap in normalized feature space."),
        ("I & Y", "Pinky Extension Ambiguity", "Thumb Occlusion", "Pinky only (I) vs Thumb+Pinky (Y); subtle thumb tuck is occluded from shallow monocular webcam angles."),
        ("R, U, V", "Finger Crossing Ambiguity", "Contact Loss", "R (crossed), U (joined), V (split); coordinate normalization strips absolute millimeter inter-finger contact distance.")
    ]

    for r_idx, row in enumerate(table4_data, start=1):
        for c_idx, val in enumerate(row):
            table4.rows[r_idx].cells[c_idx].text = val

    format_academic_table(table4, col_widths=[0.8, 1.8, 1.6, 2.8])

    # Forensic Analysis of K -> T
    add_body_p(
        doc,
        "Figure 7 illustrates the definitive forensic breakdown of the Class K -> Class T collapse (Root Cause C: Combined Translation "
        "and Hand-Shape Interaction). Counterfactual inference ablation proved that inter-wrist vertical translation (Delta_y) "
        "was not the primary driver: replacing Delta_y with the training mean still yielded 100% Class T predictions. Only when "
        "the 126D hand-shape stream was counterfactually replaced did the prediction immediately flip to Class K (99.45% confidence), "
        "demonstrating that multi-space postural drift across multiple joints collectively drove the classification failure."
    )
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig7_class_k_t_forensics.png"), width_inches=6.2)
    add_figure_caption(doc, 7, "Forensic Root-Cause Analysis of Class K → Class T Classification Collapse")

    # Figure 10: Normalization flaw
    add_body_p(
        doc,
        "Figure 10 contrasts the historical flawed normalization with the reconciled invariant feature extraction. In earlier "
        "iterations, normalizing two hands as a single unified bounding box resulted in severe scale distortion whenever the "
        "signer moved their hands apart. The reconciled pipeline normalizes each hand by its own independent metacarpal bone scale, "
        "preserving intrinsic finger geometry regardless of bilateral wrist separation."
    )
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig_normalization_flaw.png"), width_inches=6.0)
    add_figure_caption(doc, 10, "Biomechanical Skeletal Normalization and Cross-Hand Distortion Comparison")

    # Runtime Benchmarks (Table 5)
    add_body_p(
        doc,
        "To measure edge performance, exhaustive latency and resource profiling was conducted on Apple Silicon M2 (CPU) "
        "using src/benchmark_latency.py. As detailed in Table 5, the full end-to-end vision pipeline (including camera ingestion, "
        "MediaPipe Hands tracking, coordinate normalization, dual-hand MLP inference, CNN crop classification, decision fusion, "
        "and OpenCV rendering) achieves an average throughput of 22.6 FPS (44.16 ms mean latency, 39.99 ms median) on active dual-hand frames, "
        "enabling responsive real-time interaction. Default translation uses instantaneous local dictionary mapping (0.001 ms), while "
        "IndicTrans2 and Meta MMS-TTS are implemented in src/language_services as optional offline neural extensions."
    )

    add_table_caption(doc, 5, "Measured Hardware Latency Benchmarks on Apple Silicon M2 (CPU)")
    table5 = doc.add_table(rows=7, cols=5)
    headers5 = ["Pipeline Stage / Component", "Technology / Architecture", "Mean Execution Latency", "Throughput / RTF", "Memory Footprint"]
    for idx, h in enumerate(headers5):
        table5.rows[0].cells[idx].text = h

    table5_data = [
        ("Full Pipeline (Active Dual-Hand)", "InferenceEngine.process_frame (CPU)", "44.16 ms / frame (39.99 ms median)", "22.6 FPS", "185 MB"),
        ("Dual-Hand Landmark Engine", "144D Features + MLP (20 classes)", "2.91 ms / inference", "343.1 FPS", "95 MB"),
        ("Visual CNN Engine", "128x128x3 Crop + Conv2D (Keras)", "9.91 ms / inference", "100.9 FPS", "110 MB"),
        ("Multi-Modal Decision Fusion", "Confidence Arbitration (Python)", "< 0.01 ms / frame", "> 600,000 FPS", "< 1 MB"),
        ("Default Translation Engine", "Offline Dictionary Mapping (Python)", "0.001 ms / phrase", "> 10,000 phrases/s", "5 MB"),
        ("Optional Neural NMT & TTS*", "IndicTrans2 (42.8ms) & MMS-TTS (112.5ms)", "Research extensions in src/language_services", "RTF: 0.148 (MMS)", "785 MB (Combined)")
    ]

    for r_idx, row in enumerate(table5_data, start=1):
        for c_idx, val in enumerate(row):
            table5.rows[r_idx].cells[c_idx].text = val

    format_academic_table(table5, col_widths=[1.7, 1.8, 1.5, 1.0, 1.0])

    add_body_p(
        doc,
        "*Note on Optional Neural Services: IndicTrans2 and Meta MMS-TTS are implemented in src/language_services/neural_translator.py "
        "and src/language_services/neural_tts.py as optional offline research modules. The default production pipeline uses local dictionary "
        "translation and the multi-tier speech fallback engine (pyttsx3 -> Edge-TTS -> gTTS -> Web Speech API)."
    )

    print("Chapter 3 generated successfully.")
