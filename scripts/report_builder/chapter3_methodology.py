"""Chapter 3 builder: Methodology, Architecture, Design Diagrams, Implementation, and Evaluation.
"""

import os
from docx import Document
from docx.shared import Pt
from .xml_helpers import (
    add_heading_1, add_heading_2, add_heading_3, add_body_p, add_bullet_p,
    add_table_caption, add_figure_caption, insert_image_centered, format_academic_table,
    add_todo_placeholder
)

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../docs/report/figures"))


def build_chapter3(doc: Document):
    """Generate Chapter 3: Methodology."""
    
    # 3. METHODOLOGY (Style GS1 provides automatic chapter numbering)
    add_heading_1(doc, "METHODOLOGY")

    add_body_p(
        doc,
        "This chapter presents the engineering methodology, mathematical modeling, software architecture, and empirical "
        "evaluation of the proposed Indian Sign Language Recognition & Translation Platform. The methodology "
        "is structured into six logical subsections: system overview, user personas, end-to-end processing workflow, "
        "formal design diagrams (Use-Case, Class, Logical Data Model, Sequence, DFD), user interface engineering, and system implementation "
        "comprising complete algorithms, annotated source code excerpts, and comprehensive ablation results."
    )

    # 3.1 Proposed System Overview (Style GS2 provides automatic section numbering)
    add_heading_2(doc, "Proposed System Overview")

    add_body_p(
        doc,
        "The architecture operates on a Multi-Modal Decision Fusion paradigm designed to combine the strengths of "
        "appearance-based and skeletal pipelines. As illustrated in Figure 1, the pipeline executes concurrent dual-branch "
        "feature extraction upon incoming 640x480 RGB camera frames. The primary branch extracts 3D spatial coordinates "
        "via Google MediaPipe Hands, tracking 21 3D landmarks for each detected hand. A hand-presence router dynamically dispatches "
        "the extracted coordinates: if a single hand is detected, a 63D invariant coordinate vector is routed to the Single-Hand "
        "Dense MLP; if both hands are detected, an invariant 144D geometric feature vector is computed by the spatial feature "
        "extraction pipeline and routed to the Dual-Hand MLP."
    )

    add_body_p(
        doc,
        "Concurrently, the secondary visual branch computes the tight spatial bounding box surrounding the detected hand regions, "
        "resizes the cropped Region of Interest (ROI) to 128x128x3 pixels, and feeds the normalized tensor into a Convolutional "
        "Neural Network (CNN). The CNN captures visual textures, skin tone contours, and palm orientations that geometric "
        "skeletons discard. The predictions from both branches are synthesized by the Decision Fusion Engine using "
        "confidence-weighted arbitration (W_geom=0.65, W_cnn=0.35, +0.10 agreement bonus). Downstream, the predicted character enters the WordBuffer state machine, "
        "which debounces temporal transitions (1.2 s cooldown, 5-frame stability confirmation) and emits completed words to the translation "
        "engine (local dictionary mapping across 10 regional Indian languages with online fallback) and multi-tier speech engine "
        "(macOS say, pyttsx3, gTTS, and browser Web Speech API). Optional neural translation (IndicTrans2) and speech synthesis (MMS-TTS) "
        "modules are implemented in src/language_services for offline research evaluation."
    )

    # Insert Figure 1
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig1_system_architecture.png"), width_inches=6.2)
    add_figure_caption(doc, 1, "Multi-Modal Decision Fusion and Translation System Architecture")

    # 3.2 Users and Roles (Style GS2 provides automatic section numbering)
    add_heading_2(doc, "Users and Roles")

    add_body_p(
        doc,
        "The software serves three distinct user personas, each interacting with specialized system interfaces and functional boundaries:"
    )

    add_bullet_p(
        doc,
        "The primary user who communicates using Indian Sign Language. The signer requires real-time visual confirmation, "
        "including live skeletal overlays, immediate character prediction badges, and hands-free text assembly controls (such as "
        "holding neutral pauses to insert spaces or executing a dedicated clear gesture) without touching a keyboard.",
        bold_prefix="Deaf or Hard-of-Hearing Signer: "
    )

    add_bullet_p(
        doc,
        "A family member, doctor, teacher, or commercial clerk who does not understand sign language. This user consumes "
        "the translated output through large, readable regional script displays (supporting 10 Indian regional languages) "
        "and listens to synchronized audio playback synthesized across multi-tier speech backends.",
        bold_prefix="Hearing Interlocutor (Reader / Listener): "
    )

    add_bullet_p(
        doc,
        "A computer vision engineer, academic reviewer, or clinical administrator responsible for monitoring system health, "
        "verifying camera calibration, inspecting per-class confidence distributions, reviewing confusion matrices, and executing "
        "automated test suites to validate model integrity across diverse signers.",
        bold_prefix="System Administrator / Evaluator: "
    )

    # 3.3 Detailed Workflow (Style GS2 provides automatic section numbering)
    add_heading_2(doc, "Detailed Workflow")

    add_body_p(
        doc,
        "The execution lifecycle of the platform operates in eight deterministic, pipelined stages executed per frame cycle:"
    )

    add_bullet_p(doc, "OpenCV captures live 640x480 RGB video from the webcam at interactive frame rates.", bold_prefix="Stage 1: Frame Acquisition: ")
    add_bullet_p(doc, "MediaPipe Hands extracts 21 3D landmarks (x, y, z) per hand for detected hands.", bold_prefix="Stage 2: Perception & Landmark Tracking: ")
    add_bullet_p(doc, "Evaluates hand presence (0 Hands: Neutral; 1 Hand: 63D Single MLP; 2 Hands: 144D Dual MLP).", bold_prefix="Stage 3: Gating & Dispatch Router: ")
    add_bullet_p(doc, "Translates landmarks relative to wrist anchors, normalizes by metacarpal bone scale, and sorts bilateral hands.", bold_prefix="Stage 4: Invariant Normalization: ")
    add_bullet_p(doc, "Computes softmax probability vectors from both the Landmark MLP and RGB Appearance CNN (128x128 crop).", bold_prefix="Stage 5: Dual-Branch Inference: ")
    add_bullet_p(doc, "Synthesizes modalities (W_geom=0.65, W_cnn=0.35, +0.10 agreement bonus); preserves landmark physical authority during disagreements.", bold_prefix="Stage 6: Confidence Fusion: ")
    add_bullet_p(doc, "WordBuffer registers character if confidence >= 0.50 for 5 consecutive frames and 1.2 s cooldown elapsed.", bold_prefix="Stage 7: Temporal Buffering: ")
    add_bullet_p(doc, "Curated dictionary translates buffered words into 10 regional Indian languages; multi-tier speech engine vocalizes text.", bold_prefix="Stage 8: Downstream Translation & Speech: ")

    # Insert Figure 2 (DFD) and Figure 3 (Sequence)
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig2_dfd_diagrams.png"), width_inches=6.2)
    add_figure_caption(doc, 2, "Data Flow Diagrams (Level 0 Context and Level 1 Process Decomposition)")

    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig3_sequence_diagram.png"), width_inches=6.2)
    add_figure_caption(doc, 3, "Sequence Diagram of Live Prediction, Temporal Buffering, and Audio Synthesis")

    # 3.4 Design Diagrams (Style GS2 provides automatic section numbering)
    add_heading_2(doc, "Design Diagrams")

    add_heading_3(doc, "Use-case Diagrams")
    add_body_p(
        doc,
        "Figure 4 details the UML Use-Case Diagram defining system interactions. The primary Deaf Signer initiates the "
        "'Perform ISL Fingerspelling' use case, which triggers automated landmark extraction, live skeletal feedback, and "
        "'Assemble Words via Buffer State Machine'. The Hearing Interlocutor interacts with 'Translate Sign Buffer' and "
        "'Synthesize Regional Speech'. The Administrator/Evaluator executes 'Run Accuracy Audit & Landmark Diagnostics' "
        "to evaluate held-out benchmark splits and generate diagnostic reports."
    )
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig4_usecase_diagram.png"), width_inches=6.0)
    add_figure_caption(doc, 4, "UML Use-Case Diagram for System Actors and Functional Scenarios")

    add_heading_3(doc, "Class Diagrams")
    add_body_p(
        doc,
        "Figure 5 presents the formal UML Class Diagram reflecting the modular hierarchy of the codebase. "
        "The InferenceEngine acts as the central coordinator, interfacing with AlphabetEngine and WordBuffer. "
        "AlphabetEngine delegates feature computation to DualHandEngine / spatial feature extractors and multi-modal probability blending "
        "to FusionEngine. Downstream translation and speech synthesis are encapsulated within Translator and TTSEngine, "
        "with optional neural services (IndicTrans2 and MMS-TTS) isolated in src/language_services."
    )
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig5_class_diagram.png"), width_inches=6.2)
    add_figure_caption(doc, 5, "Detailed UML Class Diagram of the Software Subsystems")

    add_heading_3(doc, "ER Diagrams")
    add_body_p(
        doc,
        "Figure 6 illustrates the logical data model governing in-memory and file-based representations across the pipeline. "
        "Because the system runs completely locally without requiring an external relational database, data structures are "
        "managed through optimized in-memory tensors and file-backed caches. The model captures: FRAME_LANDMARKS "
        "(storing runtime 63D/144D feature tensors and tracking flags), ALPHABET_GESTURE (defining the 26-class canonical lexicon, "
        "bimanual classifications, and target class pools), EVALUATION_SAMPLE (storing ground-truth labels and evaluation predictions "
        "for offline test splits), TRANSLATION_RECORD (persisting source/target text in data/translation_cache.json), and "
        "TTS_AUDIO_ENTRY (managing cached audio output paths across speech synthesis tiers)."
    )
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig6_er_diagram.png"), width_inches=6.0)
    add_figure_caption(doc, 6, "Logical Data Model (In-Memory and File-Based Data Structures)")

    # 3.5 User Interface (Style GS2 provides automatic section numbering)
    add_heading_2(doc, "User Interface")
    add_heading_3(doc, "User Description and Interface Mockup")
    add_body_p(
        doc,
        "The user interface is engineered as a responsive, modern web application styled in a clean Midnight Teal theme "
        "utilizing semantic HTML5, Vanilla CSS custom properties, and native ES6 JavaScript without heavyweight frontend framework overhead. "
        "As depicted in Figure 7, the interface is organized into a cohesive dual-pane telemetry layout:\n\n"
        "1. Visual Perception Canvas (Left): Displays the 640x480 video feed overlaid with real-time MediaPipe skeletal landmarks, "
        "color-coded bounding boxes, an active prediction badge ('W'), and a dynamic confidence progress bar indicating recognition stability.\n\n"
        "2. Linguistic & Acoustic Control Center (Right): Features an assembled WordBuffer text pane showing transcribed characters, "
        "hands-free action buttons (Space, Backspace, Clear), a target language selector (supporting 10 regional Indian languages), "
        "a translation panel displaying regional script, and an integrated audio control for vocalizing speech."
    )
    ui_live_path = os.path.join(FIGURES_DIR, "ui_live_running_desktop.png")
    ui_mockup_path = os.path.join(FIGURES_DIR, "fig_ui_screenshot.png")
    if os.path.exists(ui_live_path):
        insert_image_centered(doc, ui_live_path, width_inches=6.2)
        add_figure_caption(doc, 7, "Real-Time User Interface with Live Video Overlay and Multilingual Buffer (Desktop Application v2.5 Local Desktop)")
    else:
        p_mockup_note = add_body_p(
            doc,
            "Note on Interface Representation: The graphic below depicts the simulated UI layout mockup. "
        )
        add_todo_placeholder(p_mockup_note, "Capture and insert live webcam screenshot with novel signer on physical test machine")
        insert_image_centered(doc, ui_mockup_path, width_inches=6.2)
        p_caption7 = add_figure_caption(doc, 7, "User Interface Architecture Layout Mockup (Live Video Overlay and Multilingual Buffer)")
        add_todo_placeholder(p_caption7, "Replace interface layout mockup with verified live hardware webcam screenshot")

    # 3.6 System Implementation (Style GS2 provides automatic section numbering)
    add_heading_2(doc, "System Implementation")

    add_heading_3(doc, "Algorithms/ Pseudo Code")
    add_body_p(
        doc,
        "The mathematical foundation of the platform is formalized across four deterministic algorithms governing feature "
        "extraction, multi-modal decision fusion, temporal text debouncing, and downstream translation and speech synthesis."
    )

    # Algorithm 1: Real 144D Layout
    add_body_p(
        doc,
        "Algorithm 1: Dual-Hand 144D Invariant Spatial Feature Extraction & Bilateral Sorting\n"
        "Input: Raw 3D landmark coordinates for Screen-Left hand c1 and Screen-Right hand c2, each containing 21 points (x_i, y_i, z_i).\n"
        "Output: Invariant 144-dimensional geometric feature vector F.\n"
        "1: For each hand h in {c1, c2}:\n"
        "2:     Translate coordinates relative to wrist anchor: p'_i = p_i - p_wrist for i in 0..20 (63 values).\n"
        "3:     Compute hand reference scale: s_h = ||p'_middle_mcp - p'_wrist||_2 (Euclidean distance to landmark 9).\n"
        "4:     Normalize centered coordinates: p''_i = p'_i / max(s_h, 1e-4) (63 values per hand -> 126D shape).\n"
        "5: Compute bilateral reference scale: s_ref = max((s_1 + s_2) / 2, 1e-4).\n"
        "6: Compute relative 3D wrist vector: Delta_W = (wrist_2 - wrist_1) / s_ref (3 values -> 129D).\n"
        "7: Compute 3D aperture features: a1 = ||c1_thumb - c1_index||_2 / s1; a2 = ||c2_thumb - c2_index||_2 / s2;\n"
        "   a_cross = min(||c1_thumb - c2_index||_2, ||c2_thumb - c1_index||_2) / s_ref (3 values -> 132D).\n"
        "8: Compute 4D symmetric palm proximity: D_palm[k] = min(||c1_tip[k] - c2_palm||_2, ||c2_tip[k] - c1_palm||_2) / s_ref\n"
        "   for fingertips k in {Index(8), Middle(12), Ring(16), Pinky(20)} and contralateral palm centers (4 values -> 136D).\n"
        "9: Compute 8D sorted finger extensions: for each finger k in {8, 12, 16, 20}, compute extensions ext1_k = ||c1_tip - c1_middle_mcp|| / s1\n"
        "   and ext2_k = ||c2_tip - c2_middle_mcp|| / s2; store sorted pairs [min(ext1_k, ext2_k), max(ext1_k, ext2_k)] (8 values -> 144D).\n"
        "10: Concatenate F = [c1_63, c2_63, Delta_W, a1, a2, a_cross, D_palm_4, ext_sorted_8] yielding exactly 144 invariant dimensions.\n"
        "11: Return F."
    )

    # Algorithm 2: Real Fusion Engine
    add_body_p(
        doc,
        "Algorithm 2: Confidence-Aware Multi-Modal Decision Fusion\n"
        "Input: Landmark result (pred_label, conf_lm, status), CNN result (pred_label, conf_cnn, is_available), hand_detected flag.\n"
        "Parameters: landmark_weight = 0.65, cnn_weight = 0.35, min_confidence_threshold = 0.50, agreement_bonus = 0.10.\n"
        "Output: Fused decision dictionary (fused_label, confidence, status, is_accepted).\n"
        "1: If not hand_detected then return (fused_label=None, confidence=0.0, status='NO_HAND', is_accepted=False).\n"
        "2: If CNN is offline / unavailable then:\n"
        "3:     accepted = (lm_label != None and conf_lm >= min_confidence_threshold).\n"
        "4:     Return (fused_label=lm_label if accepted else None, confidence=conf_lm, status='ACCEPTED' if accepted else 'REJECTED').\n"
        "5: If lm_label == cnn_label then: (Multi-Modal Agreement)\n"
        "6:     fused_conf = min(1.0, 0.65 * conf_lm + 0.35 * conf_cnn + 0.10).\n"
        "7:     accepted = (fused_conf >= min_confidence_threshold).\n"
        "8:     Return (fused_label=lm_label if accepted else None, confidence=fused_conf, status='AGREEMENT' if accepted else 'REJECTED').\n"
        "9: If lm_label != cnn_label then: (Disagreement Arbitration)\n"
        "10:    If conf_lm >= min_confidence_threshold (0.50) then:\n"
        "11:        Return (fused_label=lm_label, confidence=conf_lm, status='ACCEPTED', reason='Landmark Geometric Authority').\n"
        "12:    Else if conf_cnn >= 0.85 then:\n"
        "13:        Return (fused_label=cnn_label, confidence=conf_cnn, status='ACCEPTED', reason='CNN Strongly Confident').\n"
        "14:    Else: return (fused_label=None, confidence=max(conf_lm, conf_cnn), status='REJECTED_UNCERTAIN', is_accepted=False)."
    )

    # Algorithm 3: WordBuffer
    add_body_p(
        doc,
        "Algorithm 3: WordBuffer Debouncing and Temporal Text Assembly State Machine\n"
        "Input: Stream of fused predictions (C_t, Score_t), current timestamp T, cooldown tau = 1.2 s, stability threshold N = 5.\n"
        "Output: Current assembled word string W_assembled.\n"
        "1: If C_t == None or Score_t < 0.50 then:\n"
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

    # Algorithm 4: Real Downstream Translation & Speech
    add_body_p(
        doc,
        "Algorithm 4: Production Multilingual Translation & Multi-Tier Speech Synthesis\n"
        "Input: Assembled English text string S_en, target language code lang (10 regional Indian languages).\n"
        "Output: Translated regional text S_tgt, audible speech synthesis.\n"
        "1: Check local in-memory translation cache and persistent disk cache (data/translation_cache.json) for S_en.\n"
        "2: If match found in curated 10-language phrase lexicon then:\n"
        "3:     Retrieve S_tgt instantaneously (< 0.01 ms local lookup).\n"
        "4: Else: (Online Fallback)\n"
        "5:     Query MyMemory translation API with 1.5 s timeout; if successful, store S_tgt in persistent cache.\n"
        "6:     If network fails or timeout occurs, apply rule-based phonetic transliteration fallback for proper nouns.\n"
        "7: Execute Speech Synthesis via Multi-Tier Fallback Chain:\n"
        "8:     Tier 1 (macOS say): invoke native operating system voice if running on macOS.\n"
        "9:     Tier 2 (Offline pyttsx3): synthesize speech via local SAPI5/eSpeak/NSSpeechSynthesizer drivers.\n"
        "10:    Tier 3 (Online gTTS): synthesize MP3/WAV via Google Text-to-Speech and play through local audio player.\n"
        "11:    Tier 4 (Web Speech API): emit browser synthesis event as client-side fallback if server audio output is unavailable.\n"
        "12: Return S_tgt."
    )

    add_heading_3(doc, "Source Code with Documentation")
    add_body_p(
        doc,
        "The core mathematical and systems logic is embodied in production Python modules under src/. "
        "Listing 1 illustrates the exact dual-hand 144D invariant feature extraction implemented in "
        "src/dual_hand_engine.py (Lines 75–142), which ensures scale invariance and extracts relative wrist vectors, "
        "apertures, palm proximities, and sorted finger extensions."
    )

    code_snippet1 = (
        "# --- Source Code Excerpt: Dual-Hand 144D Feature Extraction (src/dual_hand_engine.py, Lines 75–142) ---\n"
        "def compute_dual_hand_features(c1: np.ndarray, c2: np.ndarray, config_key: str = \"config_d_ablation_144d\") -> np.ndarray:\n"
        "    c1, c2 = np.array(c1, dtype=np.float32), np.array(c2, dtype=np.float32)\n"
        "    has_c1, has_c2 = not np.allclose(c1, 0.0), not np.allclose(c2, 0.0)\n"
        "    has_both = has_c1 and has_c2\n"
        "\n"
        "    # 1. Base 126D Shape (63D per hand, wrist-normalized & middle-MCP scaled) + 3D Relative Wrist\n"
        "    c1_63, s1, wrist_1 = normalize_single_hand_coords(c1)\n"
        "    c2_63, s2, wrist_2 = normalize_single_hand_coords(c2)\n"
        "    s_ref = max((s1 + s2) / 2.0, 1e-4) if has_both else (s1 if has_c1 else s2)\n"
        "    rel_wrist_3 = ((wrist_2 - wrist_1) / s_ref).astype(np.float32) if has_both else np.zeros(3, dtype=np.float32)\n"
        "    feat_129 = np.concatenate([c1_63, c2_63, rel_wrist_3]).astype(np.float32)\n"
        "\n"
        "    # 2. 3D Aperture Features: Thumb(4) to Index(8) distance in c1, c2, and min cross-hand\n"
        "    a1 = float(np.linalg.norm(c1[4] - c1[8])) / s1 if has_c1 else 0.0\n"
        "    a2 = float(np.linalg.norm(c2[4] - c2[8])) / s2 if has_c2 else 0.0\n"
        "    a_cross = min(float(np.linalg.norm(c1[4, :2] - c2[8, :2])), float(np.linalg.norm(c2[4, :2] - c1[8, :2]))) / s_ref if has_both else 0.0\n"
        "    feat_132 = np.concatenate([feat_129, [a1, a2, a_cross]]).astype(np.float32)\n"
        "\n"
        "    # 3. Supplemental 144D Features: 4D Palm Proximity + 8D Sorted Finger Extensions\n"
        "    tips, palm_landmarks = [8, 12, 16, 20], [0, 5, 9, 17]\n"
        "    c1_palm = np.mean(c1[palm_landmarks, :2], axis=0) if has_c1 else np.zeros(2, dtype=np.float32)\n"
        "    c2_palm = np.mean(c2[palm_landmarks, :2], axis=0) if has_c2 else np.zeros(2, dtype=np.float32)\n"
        "    palm_prox_4d = np.array([min(np.linalg.norm(c1[t, :2] - c2_palm), np.linalg.norm(c2[t, :2] - c1_palm)) / s_ref for t in tips], dtype=np.float32) if has_both else np.zeros(4, dtype=np.float32)\n"
        "\n"
        "    ext1 = (np.linalg.norm(c1[tips] - c1[9], axis=1) / max(s1, 1e-4)).astype(np.float32) if has_c1 else np.zeros(4, dtype=np.float32)\n"
        "    ext2 = (np.linalg.norm(c2[tips] - c2[9], axis=1) / max(s2, 1e-4)).astype(np.float32) if has_c2 else np.zeros(4, dtype=np.float32)\n"
        "    sorted_ext_8d = np.zeros(8, dtype=np.float32)\n"
        "    for t_idx in range(4):\n"
        "        sorted_ext_8d[2 * t_idx], sorted_ext_8d[2 * t_idx + 1] = min(ext1[t_idx], ext2[t_idx]), max(ext1[t_idx], ext2[t_idx])\n"
        "\n"
        "    return np.concatenate([feat_132, palm_prox_4d, sorted_ext_8d]).astype(np.float32)"
    )
    p_code1 = doc.add_paragraph(style='Normal')
    p_code1.paragraph_format.line_spacing = 1.0
    r_code1 = p_code1.add_run(code_snippet1)
    r_code1.font.name = 'Courier New'
    r_code1.font.size = Pt(8.5)

    add_body_p(
        doc,
        "Listing 2 demonstrates the decision fusion logic in src/fusion_engine.py (Lines 40–120), showing the calibrated weighting "
        "(W_geom=0.65, W_cnn=0.35), agreement bonus calculation (+0.10), and landmark physical authority preservation."
    )

    code_snippet2 = (
        "# --- Source Code Excerpt: Confidence Decision Fusion (src/fusion_engine.py, Lines 40–120) ---\n"
        "class FusionEngine:\n"
        "    def __init__(self, landmark_weight: float = 0.65, cnn_weight: float = 0.35, min_confidence_threshold: float = 0.50, agreement_bonus: float = 0.10):\n"
        "        self.landmark_weight = landmark_weight\n"
        "        self.cnn_weight = cnn_weight\n"
        "        self.min_confidence_threshold = min_confidence_threshold\n"
        "        self.agreement_bonus = agreement_bonus\n"
        "\n"
        "    def fuse(self, landmark_result: Dict[str, Any], cnn_result: Optional[Dict[str, Any]] = None, hand_detected: bool = False) -> Dict[str, Any]:\n"
        "        if not hand_detected:\n"
        "            return {\"fused_label\": None, \"confidence\": 0.0, \"status\": \"NO_HAND\", \"is_accepted\": False}\n"
        "\n"
        "        lm_label, lm_conf = landmark_result.get(\"pred_label\"), float(landmark_result.get(\"confidence\", 0.0))\n"
        "        cnn_available = cnn_result is not None and cnn_result.get(\"is_available\", False)\n"
        "        cnn_label = cnn_result.get(\"pred_label\") if cnn_available else None\n"
        "        cnn_conf = float(cnn_result.get(\"confidence\", 0.0)) if cnn_available else 0.0\n"
        "\n"
        "        if not cnn_available:\n"
        "            accepted = lm_label is not None and lm_conf >= self.min_confidence_threshold\n"
        "            return {\"fused_label\": lm_label if accepted else None, \"confidence\": lm_conf, \"status\": \"ACCEPTED\" if accepted else \"REJECTED\"}\n"
        "\n"
        "        # Multi-Modal Agreement (+0.10 Bonus)\n"
        "        if lm_label is not None and lm_label == cnn_label:\n"
        "            fused_conf = min(1.0, self.landmark_weight * lm_conf + self.cnn_weight * cnn_conf + self.agreement_bonus)\n"
        "            accepted = fused_conf >= self.min_confidence_threshold\n"
        "            return {\"fused_label\": lm_label if accepted else None, \"confidence\": round(fused_conf, 4), \"status\": \"AGREEMENT\" if accepted else \"REJECTED\", \"is_accepted\": accepted}\n"
        "\n"
        "        # Disagreement Arbitration: Prioritize Geometric Authority when confident\n"
        "        if lm_label is not None and lm_conf >= self.min_confidence_threshold:\n"
        "            return {\"fused_label\": lm_label, \"confidence\": lm_conf, \"status\": \"ACCEPTED\", \"is_accepted\": True}\n"
        "        if cnn_label is not None and cnn_conf >= 0.85:\n"
        "            return {\"fused_label\": cnn_label, \"confidence\": cnn_conf, \"status\": \"ACCEPTED\", \"is_accepted\": True}\n"
        "\n"
        "        return {\"fused_label\": None, \"confidence\": max(lm_conf, cnn_conf), \"status\": \"REJECTED\", \"is_accepted\": False}"
    )
    p_code2 = doc.add_paragraph(style='Normal')
    p_code2.paragraph_format.line_spacing = 1.0
    r_code2 = p_code2.add_run(code_snippet2)
    r_code2.font.name = 'Courier New'
    r_code2.font.size = Pt(8.5)

    # 3.6.3 Experimental Results & Evaluation
    add_heading_3(doc, "Experimental Results, Ablation Studies, and Failure Diagnostics")

    add_body_p(
        doc,
        "All models were evaluated on controlled, held-out same-signer test "
        "partitions using immutable frozen checkpoints. The test set comprises N=243 samples from the RealSign held-out partition "
        "(and N=252 total samples when including 9 cross-dataset OpenHands samples). In accordance "
        "with data governance principles, ground-truth labels were reconciled (correcting six mislabeled OpenHands Subject 4 "
        "frames from Letter I to Letter H). Table 3 details the empirical ablation comparing four distinct feature "
        "architectures evaluated under frozen seed 42 protocol and B=1,000 bootstrap iterations as implemented in src/evaluate.py."
    )

    # Table 3: Reconciled Benchmark Results
    add_table_caption(doc, 3, "Reconciled Held-Out Benchmark Performance Across Model Architectures")
    table3 = doc.add_table(rows=9, cols=6)
    headers3 = ["Architecture Configuration", "Feature Dim", "Top-1 Accuracy (%)", "Macro F1-Score", "Bootstrap 95% CI", "Class H Recall"]
    for idx, h in enumerate(headers3):
        table3.rows[0].cells[idx].text = h

    table3_data = [
        ("Config A (Minimal Invariant)", "129D", "77.78%", "0.7075", "[72.22%, 82.94%]", "100.0% (6/6)"),
        ("Config C (Angle Augmented)", "132D", "76.98%", "0.6891", "[71.43%, 82.14%]", "66.7% (4/6)"),
        ("Config D-Minimal (Production Baseline)", "140D", "77.38%", "0.6920", "[71.83%, 82.54%]", "100.0% (6/6)"),
        ("Config D-Ablation (Clean RealSign N=243)", "144D", "78.19%", "0.7349", "[72.43%, 83.13%]", "100.0% (6/6)"),
        ("Config D-Ablation (+ K/T Override)*", "144D", "83.13%", "0.7834", "[78.19%, 87.65%]", "100.0% (6/6)"),
        ("Clean Non-Duplicate Subset (N=232)", "144D", "77.16%", "0.7349", "[71.12%, 81.90%]", "100.0% (6/6)"),
        ("Non-Duplicate Subset (+ K/T Override)*", "144D", "82.33%", "0.7834", "[76.72%, 87.07%]", "100.0% (6/6)"),
        ("Full Test Split (RealSign + OpenHands N=252)", "144D", "75.79%", "0.6683", "[70.24%, 80.95%]", "95.2% (20/21)")
    ]

    for r_idx, row in enumerate(table3_data, start=1):
        for c_idx, val in enumerate(row):
            table3.rows[r_idx].cells[c_idx].text = val

    format_academic_table(table3, col_widths=[1.8, 0.9, 1.2, 1.1, 1.3, 1.1])

    add_body_p(
        doc,
        "*Note on K/T Override: Heuristic thresholds were formulated while inspecting Class K misclassifications during research iterations; "
        "this +4.94% gain is possibly tuned on test data and is not a clean test estimate. Near-duplicate analysis (1-NN distance >= 0.10, N=232) verified that removing the 4.5% "
        "nearest samples changed accuracy by about 1 point (78.19% -> 77.16% clean baseline, 83.13% -> 82.33% with K/T override)."
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
        "(Geometry right on 4, CNN right on 15, neither right on 38). (Separate reference: applying the post-hoc K/T override shifts 12 samples into agreement: "
        "198 agree with 100% accuracy, and 45 disagree with Geometry right 4, CNN right 3, neither right 38)."
    )

    # Insert Figure 8 (Benchmark Ablation)
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig8_benchmark_ablation.png"), width_inches=6.0)
    add_figure_caption(doc, 8, "Reconciled Held-Out Benchmark Performance Across Dual-Hand Feature Architectures")
    add_body_p(
        doc,
        "Note on Figure 8: As illustrated, the bar chart displays accuracy and macro F1 scores across the evaluated dual-hand feature configurations. "
        "Note that the vertical y-axis is truncated (starting at 60%) to visually distinguish fine performance differentials across model ablations."
    )

    # Pipeline Fragility & Diagnostic Analysis
    add_body_p(
        doc,
        "Pipeline Fragility & Diagnostic Analysis: Despite achieving 78.19% baseline accuracy (83.13% with K/T override) "
        "on the held-out same-signer test split, offline analysis and informal testing revealed pipeline fragility (formal live validation pending). "
        "An engineering diagnosis uncovered two primary root causes: (1) Camera Mirroring Inversion: training images are third-person frontal photos, "
        "whereas unadjusted selfie webcams invert horizontal coordinates, causing accuracy to drop from 78.19% to 29.22% "
        "(-48.97% drop on held-out landmark data; live validation pending); (2) Hand Occlusion Drops: physical occlusion during signs like B, D, P dropped hand-count from 2 to 1, "
        "which was reduced via a 2-frame temporal hysteresis filter yielding an 85.7% drop in flapping on a synthetic 50-frame dropout sequence, not real recordings. "
        "Table 4 documents the observed failure modes alongside their skeletal root causes."
    )

    # Table 4: Per-Class Diagnostic Failure Modes
    add_table_caption(doc, 4, "Per-Class Diagnostic Failure Modes and Skeletal Root Causes")
    table4 = doc.add_table(rows=9, cols=4)
    headers4 = ["Letter", "Observed Misclassification", "Root Cause Domain", "Morphological & Mathematical Mechanism"]
    for idx, h in enumerate(headers4):
        table4.rows[0].cells[idx].text = h

    table4_data = [
        ("K", "Collapse to Class T (0/15 K)", "Multi-Space Attractor (Root Cause C)", "Test signer exhibited left index extension 1.12 vs train 1.50 (mimicking T's 1.15) and wrist Delta_y = -2.35 (mimicking T's -2.17)."),
        ("B", "Spurious False-Positive Default", "Entry Posture Overlap", "Open palm with 4 extended fingers triggers when hands enter or exit camera frame; requires 0.75 threshold gate."),
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

    # Analysis of K -> T
    add_body_p(
        doc,
        "Figure 9 illustrates the diagnostic breakdown of the Class K -> Class T collapse (Root Cause C: Combined Translation "
        "and Hand-Shape Interaction). Counterfactual inference ablation showed that inter-wrist vertical translation (Delta_y) "
        "was not the primary driver: replacing Delta_y with the training mean still yielded Class T predictions. Only when "
        "the 126D hand-shape stream was counterfactually replaced did the prediction immediately flip to Class K (99.45% confidence), "
        "showing that multi-space postural drift across multiple joints collectively drove the classification failure."
    )
    forensic_img = os.path.join(FIGURES_DIR, "fig9_class_k_t_forensics.png")
    if not os.path.exists(forensic_img):
        forensic_img = os.path.join(FIGURES_DIR, "fig7_class_k_t_forensics.png")
    insert_image_centered(doc, forensic_img, width_inches=6.2)
    add_figure_caption(doc, 9, "Analysis of Class K to Class T Misclassifications")

    # Figure 10: Normalization comparison
    add_body_p(
        doc,
        "Figure 10 contrasts the historical flawed normalization with the reconciled invariant feature extraction. In earlier "
        "iterations, normalizing two hands as a single unified bounding box resulted in scale distortion whenever the "
        "signer moved their hands apart. The reconciled pipeline normalizes each hand by its own independent metacarpal bone scale, "
        "preserving intrinsic finger geometry regardless of bilateral wrist separation."
    )
    insert_image_centered(doc, os.path.join(FIGURES_DIR, "fig_normalization_flaw.png"), width_inches=5.5)
    add_figure_caption(doc, 10, "Skeletal Normalization and Cross-Hand Distortion Comparison")

    # Runtime Benchmarks (Table 5)
    add_body_p(
        doc,
        "To measure edge performance, latency and resource profiling was conducted on Apple Silicon M2 (CPU) "
        "using src/benchmark_latency.py. As detailed in Table 5, the full end-to-end vision pipeline (including camera ingestion, "
        "MediaPipe Hands tracking, coordinate normalization, dual-hand MLP inference, CNN crop classification, decision fusion, "
        "and OpenCV rendering) achieves an average throughput of 22.6 FPS (44.16 ms mean latency, 39.99 ms median) on active dual-hand frames, "
        "enabling responsive real-time interaction. Default translation uses instantaneous local dictionary mapping (0.001 ms), while "
        "IndicTrans2 and Meta MMS-TTS are implemented in src/language_services as optional offline neural extensions."
    )

    add_table_caption(doc, 5, "Measured Hardware Latency Benchmarks on Apple Silicon M2 (CPU)")
    table5 = doc.add_table(rows=8, cols=5)
    headers5 = ["Pipeline Stage / Component", "Technology / Architecture", "Median Latency", "Mean Latency", "Throughput / FPS"]
    for idx, h in enumerate(headers5):
        table5.rows[0].cells[idx].text = h

    table5_data = [
        ("Full Pipeline (Active Dual-Hand)", "InferenceEngine.process_frame (CPU)", "39.99 ms", "44.16 ms", "22.6 FPS"),
        ("Full Pipeline (Empty Frame)", "InferenceEngine.process_frame (CPU)", "13.23 ms", "13.28 ms", "75.3 FPS"),
        ("AlphabetEngine Dual-Hand Decision", "MediaPipe Hands + 144D MLP + Fusion", "14.62 ms", "14.84 ms", "67.4 FPS"),
        ("Visual CNN Engine (128x128x3)", "128x128x3 Crop + Conv2D (Keras)", "8.91 ms", "9.91 ms", "100.9 FPS"),
        ("Dual-Hand Landmark Engine", "144D Features + MLP (20 classes)", "2.91 ms", "2.91 ms", "343.1 FPS"),
        ("Single-Hand Baseline MLP", "63D Invariant Features + MLP (26 classes)", "1.81 ms", "1.86 ms", "538.5 FPS"),
        ("Multi-Modal Decision Fusion", "Confidence Arbitration (Python)", "< 0.01 ms", "< 0.01 ms", "> 600,000 FPS")
    ]

    for r_idx, row in enumerate(table5_data, start=1):
        for c_idx, val in enumerate(row):
            table5.rows[r_idx].cells[c_idx].text = val

    format_academic_table(table5, col_widths=[1.8, 2.0, 1.2, 1.2, 1.0])

    add_body_p(
        doc,
        "Note on Pipeline Benchmarking: Latency metrics were recorded using src/benchmark_latency.py over 50 iterations on an Apple Silicon M2 processor. "
        "Default translation uses instantaneous offline dictionary mapping (0.001 ms mean latency, >10,000 phrases/s). Optional neural translation "
        "(IndicTrans2) and speech synthesis (Meta MMS-TTS) modules are implemented in src/language_services for offline research evaluation."
    )

    print("Chapter 3 generated successfully.")
