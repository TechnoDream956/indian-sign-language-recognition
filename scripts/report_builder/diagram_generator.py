"""Diagram generator for ISL Project Report.
Generates publication-quality technical diagrams using matplotlib.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../docs/report/figures"))


def ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def copy_existing_figures():
    """Copy high-value existing figures into figures directory."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    # UI screenshot
    ui_src = os.path.join(repo_root, "reserve/historical_reports/prompt_audits/ui_live_running_desktop.png")
    if os.path.exists(ui_src):
        shutil.copy2(ui_src, os.path.join(FIGURES_DIR, "fig_ui_screenshot.png"))
        print("Copied UI live running desktop screenshot.")

    # Normalization comparison
    norm_src = os.path.join(repo_root, "reserve/historical_reports/reports/dual_hand_investigation/images/normalization_flaw_comparison.png")
    if os.path.exists(norm_src):
        shutil.copy2(norm_src, os.path.join(FIGURES_DIR, "fig_normalization_flaw.png"))
        print("Copied normalization flaw comparison.")

    # Confusion matrix or MoP-GRU
    mopgru_src = os.path.join(repo_root, "assets/mopgru_pipeline.png")
    if os.path.exists(mopgru_src):
        shutil.copy2(mopgru_src, os.path.join(FIGURES_DIR, "fig_mopgru_pipeline.png"))
        print("Copied mopgru pipeline.")


def generate_architecture_diagram():
    """Figure 1: Multi-Modal Decision Fusion System Architecture."""
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Palette
    c_blue = '#1f77b4'
    c_teal = '#17becf'
    c_orange = '#ff7f0e'
    c_green = '#2ca02c'
    c_purple = '#9467bd'
    c_gray = '#e0e0e0'
    c_dark = '#333333'

    def draw_box(x, y, w, h, title, subtitle="", color='#ffffff', edgecolor=c_blue, lw=1.5):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.5",
                                     linewidth=lw, edgecolor=edgecolor, facecolor=color, zorder=2)
        ax.add_patch(rect)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_dark, zorder=3)
            ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=7.5, color='#555555', zorder=3)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=c_dark, zorder=3)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color='#444444', lw=1.5, shrinkA=3, shrinkB=3), zorder=4)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 1.2, label, ha='center', va='bottom', fontsize=7.5, color='#222222', zorder=5)

    # 1. Video Source
    draw_box(37.5, 90, 25, 7, "Camera / WebRTC Stream", "640x480 RGB @ 30 FPS", color='#e8f4f8', edgecolor=c_blue)

    # 2. Preprocessing / Perception Layer
    draw_box(12, 75, 34, 8, "MediaPipe Holistic Perception", "543 Anatomical Landmarks (21 Keypoints / Hand)", color='#eafaf1', edgecolor=c_green)
    draw_box(54, 75, 34, 8, "Spatial ROI Extraction", "Hand Bounding Box & 64x64 Crop", color='#fef5e7', edgecolor=c_orange)

    draw_arrow(43, 90, 29, 83)
    draw_arrow(57, 90, 71, 83)

    # 3. Model Backbones
    draw_box(4, 56, 26, 11, "Single-Hand MLP", "63D Invariant Coordinates\nDense(128-64) + Softmax (26C)", color='#ffffff', edgecolor=c_green)
    draw_box(32, 56, 30, 11, "Dual-Hand Spatial MLP", "144D Invariant Features\nDeep MLP(256-128-64) (20C)", color='#ffffff', edgecolor=c_teal)
    draw_box(64, 56, 32, 11, "RGB Appearance CNN", "4x Conv2D + MaxPool + Dense\nColor / Texture Backbone (26C)", color='#ffffff', edgecolor=c_orange)

    draw_arrow(25, 75, 17, 67, "1 Hand")
    draw_arrow(33, 75, 47, 67, "2 Hands")
    draw_arrow(71, 75, 80, 67, "ROI Tensor")

    # 4. Multi-Modal Decision Fusion
    draw_box(20, 38, 60, 11, "Confidence-Aware Decision Fusion Engine", 
             "Weighted Fusion (W_geom=0.60, W_cnn=0.40) + Agreement Bonus (+0.15)\nDynamic Posture Gating & Spatial Validation", 
             color='#f5eef8', edgecolor=c_purple, lw=2.0)

    draw_arrow(17, 56, 35, 49)
    draw_arrow(47, 56, 50, 49)
    draw_arrow(80, 56, 65, 49)

    # 5. Temporal Buffer State Machine
    draw_box(22, 22, 56, 9, "WordBuffer Temporal State Machine", 
             "Temporal Cooldown Debounce (1.2s) • Stability Threshold (5 Frames)\nSpace Insertion • Backspace • Spell Synthesis", 
             color='#fbfcfc', edgecolor=c_blue)

    draw_arrow(50, 38, 50, 31, "Verified Token")

    # 6. Downstream Subsystems (NMT & Neural TTS)
    draw_box(5, 5, 42, 10, "IndicTrans2 Multilingual NMT", "ONNX int8 (200M Distilled)\n8 Regional Indian Languages", color='#e8f8f5', edgecolor='#16a085')
    draw_box(53, 5, 42, 10, "Meta MMS-TTS Neural Speech", "VITS Multilingual Synthesis\nSub-50ms Speech Generation", color='#f4ecf7', edgecolor='#8e44ad')

    draw_arrow(40, 22, 26, 15, "English Sign Buffer")
    draw_arrow(60, 22, 74, 15, "Synthesized Text")

    plt.title("Figure 1: Multi-Modal Decision Fusion and Translation Architecture", fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig1_system_architecture.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def generate_dfd_diagram():
    """Figure 2: Data Flow Diagram (DFD Level 0 & Level 1)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300)
    for ax in (ax1, ax2):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')

    c_blue = '#1f77b4'
    c_orange = '#ff7f0e'
    c_green = '#2ca02c'
    c_gray = '#eaecee'

    # --- DFD Level 0 (Context Level) ---
    ax1.text(50, 94, "Data Flow Diagram — Level 0 (Context Level)", ha='center', fontsize=10, fontweight='bold')
    
    # External Entities
    rect_signer = patches.Rectangle((4, 35), 20, 25, facecolor='#ebf5fb', edgecolor=c_blue, lw=1.5)
    ax1.add_patch(rect_signer)
    ax1.text(14, 47.5, "Deaf Signer\n(User)", ha='center', va='center', fontsize=9, fontweight='bold')

    circle_system = patches.Circle((50, 47.5), 18, facecolor='#eafaf1', edgecolor=c_green, lw=2)
    ax1.add_patch(circle_system)
    ax1.text(50, 47.5, "0.0\nISL Recognition &\nTranslation System", ha='center', va='center', fontsize=9.5, fontweight='bold')

    rect_listener = patches.Rectangle((76, 35), 20, 25, facecolor='#fef5e7', edgecolor=c_orange, lw=1.5)
    ax1.add_patch(rect_listener)
    ax1.text(86, 47.5, "Hearing\nInterlocutor", ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrows Level 0
    ax1.annotate('Raw Video Stream\n(ISL Gestures)', xy=(32, 53), xytext=(24, 53),
                arrowprops=dict(arrowstyle="->", color='#333333', lw=1.5), fontsize=8, ha='center', va='bottom')
    ax1.annotate('Visual Feedback\n(Video + Transcripts)', xy=(24, 42), xytext=(32, 42),
                arrowprops=dict(arrowstyle="->", color='#333333', lw=1.5), fontsize=8, ha='center', va='top')
    ax1.annotate('Translated Text &\nNeural Speech Audio', xy=(76, 47.5), xytext=(68, 47.5),
                arrowprops=dict(arrowstyle="->", color='#333333', lw=1.5), fontsize=8, ha='center', va='bottom')

    # --- DFD Level 1 (Decomposition) ---
    ax2.text(50, 96, "Data Flow Diagram — Level 1 (Process Decomposition)", ha='center', fontsize=10, fontweight='bold')

    def pbox(ax, x, y, num, text, color='#f4f6f6', border=c_blue):
        rect = patches.FancyBboxPatch((x, y), 18, 16, boxstyle="round,pad=0.5", facecolor=color, edgecolor=border, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 9, y + 11.5, num, ha='center', va='center', fontsize=8, fontweight='bold', color=border)
        ax.text(x + 9, y + 5.5, text, ha='center', va='center', fontsize=7.5, color='#222222')

    pbox(ax2, 3, 40, "1.0", "Frame Capture &\nLandmark Detection", color='#ebf5fb', border=c_blue)
    pbox(ax2, 28, 62, "2.0", "Feature Extraction\n& Invariant Norm", color='#eafaf1', border=c_green)
    pbox(ax2, 28, 18, "3.0", "Appearance ROI\nCrop (RGB CNN)", color='#fef5e7', border=c_orange)
    pbox(ax2, 53, 40, "4.0", "Multi-Modal Fusion\n& Temporal Buffer", color='#f5eef8', border='#8e44ad')
    pbox(ax2, 78, 40, "5.0", "Indic NMT &\nNeural TTS Engine", color='#e8f8f5', border='#16a085')

    # Data Store
    ax2.plot([40, 60], [8, 8], color='#555555', lw=1.5)
    ax2.plot([40, 60], [14, 14], color='#555555', lw=1.5)
    ax2.text(50, 11, "D1: Sign Lexicon & Cache", ha='center', va='center', fontsize=7.5, fontweight='bold')

    # Arrows Level 1
    ax2.annotate('', xy=(28, 68), xytext=(21, 52), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax2.text(23, 62, "Landmarks", fontsize=7)

    ax2.annotate('', xy=(28, 28), xytext=(21, 44), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax2.text(23, 34, "RGB Frames", fontsize=7)

    ax2.annotate('', xy=(53, 50), xytext=(46, 68), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax2.text(48, 61, "Geom Probs", fontsize=7)

    ax2.annotate('', xy=(53, 46), xytext=(46, 28), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax2.text(48, 35, "CNN Probs", fontsize=7)

    ax2.annotate('', xy=(78, 48), xytext=(71, 48), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax2.text(74.5, 51, "Text Tokens", fontsize=7, ha='center')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig2_dfd_diagrams.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def generate_sequence_diagram():
    """Figure 3: Sequence Diagram for Live Prediction & SSE Cycle."""
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    lifelines = [
        (12, "User / Camera"),
        (32, "Flask MJPEG / SSE"),
        (52, "Inference / Fusion"),
        (72, "WordBuffer Engine"),
        (90, "Language Services")
    ]

    # Draw lifelines
    for x, label in lifelines:
        rect = patches.Rectangle((x-9, 90), 18, 7, facecolor='#ebf5fb', edgecolor='#2980b9', lw=1.5)
        ax.add_patch(rect)
        ax.text(x, 93.5, label, ha='center', va='center', fontsize=8, fontweight='bold')
        ax.plot([x, x], [10, 90], color='#bdc3c7', linestyle='--', lw=1.2, zorder=1)

    steps = [
        (85, 12, 32, "1: capture_frame() [30 FPS RGB]", True),
        (78, 32, 52, "2: extract_features() & predict()", True),
        (71, 52, 52, "3: DecisionFusion (W_geom=0.6, W_cnn=0.4)", False),
        (63, 52, 72, "4: update_buffer(char, conf >= 0.70)", True),
        (55, 72, 72, "5: debounce_check(cooldown=1.2s)", False),
        (47, 72, 32, "6: push_event('letter', 'W') via SSE", True),
        (39, 32, 12, "7: render_overlay() & update_dom()", True),
        (30, 72, 90, "8: translate_sentence('WATER') [Optional]", True),
        (22, 90, 90, "9: IndicTrans2 ONNX int8 + MMS-TTS", False),
        (14, 90, 12, "10: audio_playback(WAV audio / hi speech)", True)
    ]

    for y, x1, x2, msg, is_arrow in steps:
        if is_arrow:
            ax.annotate('', xy=(x2, y), xytext=(x1, y),
                        arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=1.2, shrinkA=0, shrinkB=0), zorder=3)
            ax.text((x1+x2)/2, y + 1.2, msg, ha='center', va='bottom', fontsize=7.2, color='#2c3e50', zorder=4)
        else:
            # Self activation
            rect_act = patches.Rectangle((x1-1, y-3), 2, 6, facecolor='#f9e79f', edgecolor='#d4ac0d', lw=1.0, zorder=2)
            ax.add_patch(rect_act)
            ax.annotate('', xy=(x1+1, y-2), xytext=(x1+6, y+2),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.5", color='#b7950b', lw=1.0))
            ax.text(x1+7, y, msg, ha='left', va='center', fontsize=7.0, color='#7d6608')

    plt.title("Figure 3: Sequence Diagram of Live Prediction, Temporal Buffering, and Audio Synthesis", fontsize=10.5, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig3_sequence_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def generate_usecase_diagram():
    """Figure 4: UML Use-Case Diagram."""
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # System boundary
    rect_sys = patches.Rectangle((28, 5), 58, 90, facecolor='#fafafa', edgecolor='#7f8c8d', lw=1.5, linestyle='--')
    ax.add_patch(rect_sys)
    ax.text(57, 92, "ISL Recognition & Translation Platform Boundary", ha='center', fontsize=9.5, fontweight='bold', color='#2c3e50')

    # Actors
    def draw_actor(ax, x, y, name):
        # Head
        circle = patches.Circle((x, y+6), 2.5, facecolor='#ecf0f1', edgecolor='#2c3e50', lw=1.5)
        ax.add_patch(circle)
        # Body
        ax.plot([x, x], [y+3.5, y-3], color='#2c3e50', lw=1.5)
        # Arms
        ax.plot([x-3.5, x+3.5], [y+1, y+1], color='#2c3e50', lw=1.5)
        # Legs
        ax.plot([x, x-3], [y-3, y-8], color='#2c3e50', lw=1.5)
        ax.plot([x, x+3], [y-3, y-8], color='#2c3e50', lw=1.5)
        # Label
        ax.text(x, y-11, name, ha='center', va='top', fontsize=8.5, fontweight='bold', color='#2c3e50')

    draw_actor(ax, 12, 60, "Deaf Signer\n(Primary)")
    draw_actor(ax, 12, 25, "Administrator /\nEvaluator")
    draw_actor(ax, 93, 50, "Hearing\nInterlocutor")

    # Use Cases (Ovals)
    use_cases = [
        (57, 82, "Perform ISL Fingerspelling", 35),
        (57, 69, "View Real-Time Prediction & Skeleton", 38),
        (57, 56, "Assemble Words via Buffer State Machine", 40),
        (57, 43, "Translate Sign Buffer into Indian Languages", 42),
        (57, 30, "Synthesize Regional Neural Speech (TTS)", 40),
        (57, 17, "Run Accuracy Audit & Landmark Diagnostics", 42)
    ]

    for x, y, text, w in use_cases:
        ellipse = patches.Ellipse((x, y), w, 8.5, facecolor='#ffffff', edgecolor='#2980b9', lw=1.5)
        ax.add_patch(ellipse)
        ax.text(x, y, text, ha='center', va='center', fontsize=7.8, color='#1a5276', fontweight='bold')

    # Connections
    # Signer connects to 1, 2, 3, 4
    for y in [82, 69, 56, 43]:
        ax.plot([17, 39], [60, y], color='#34495e', lw=1.2)

    # Admin connects to 6
    ax.plot([17, 39], [25, 17], color='#34495e', lw=1.2)

    # Hearing interlocutor connects to 2, 4, 5
    for y in [69, 43, 30]:
        ax.plot([88, 75], [50, y], color='#34495e', lw=1.2)

    plt.title("Figure 4: UML Use-Case Diagram for System Actors and Functional Scenarios", fontsize=10.5, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig4_usecase_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def generate_class_diagram():
    """Figure 5: UML Class Diagram reflecting real codebase architecture."""
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def draw_class(ax, x, y, w, h, name, attrs, methods, color='#ffffff', border='#2980b9'):
        rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor=border, lw=1.5)
        ax.add_patch(rect)
        # Header
        h_header = 4.5
        rect_head = patches.Rectangle((x, y+h-h_header), w, h_header, facecolor='#ebf5fb', edgecolor=border, lw=1.5)
        ax.add_patch(rect_head)
        ax.text(x+w/2, y+h-h_header/2, name, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1a5276')
        
        # Attributes
        y_text = y + h - h_header - 2.5
        for attr in attrs:
            ax.text(x + 1.5, y_text, attr, ha='left', va='center', fontsize=7.0, color='#333333')
            y_text -= 2.6
        
        # Divider line
        y_div = y_text + 0.8
        ax.plot([x, x+w], [y_div, y_div], color=border, lw=1.0)
        
        # Methods
        y_text = y_div - 2.2
        for m in methods:
            ax.text(x + 1.5, y_text, m, ha='left', va='center', fontsize=7.0, color='#1b4f72')
            y_text -= 2.6

    # 1. InferenceEngine
    draw_class(ax, 3, 58, 28, 38, "InferenceEngine", 
               ["- holistic_model: Holistic", "- active_mode: str", "- frame_buffer: Deque", "- confidence_thresh: float"],
               ["+ process_frame(frame: ndarray)", "+ switch_mode(mode: str)", "+ extract_landmarks()", "+ get_state_metrics()"],
               color='#fdfefe', border='#2980b9')

    # 2. AlphabetEngine
    draw_class(ax, 36, 58, 30, 38, "AlphabetEngine",
               ["- landmark_mlp: Model", "- cnn_model: Model", "- spatial_extractor: Extractor", "- fusion_engine: Fusion"],
               ["+ predict(landmarks, crop_img)", "+ detect_hands_presence()", "+ route_dispatch()", "+ evaluate_confidence()"],
               color='#fdfefe', border='#16a085')

    # 3. DecisionFusionEngine
    draw_class(ax, 71, 58, 26, 38, "DecisionFusionEngine",
               ["- weight_geom: float = 0.60", "- weight_cnn: float = 0.40", "- bonus_agreement: float = 0.15", "- gate_threshold: float"],
               ["+ fuse_probabilities(P_g, P_c)", "+ apply_suppression_gates()", "+ get_top_prediction()"],
               color='#fdfefe', border='#8e44ad')

    # 4. WordBuffer
    draw_class(ax, 3, 8, 28, 42, "WordBuffer",
               ["- buffer: List[str]", "- last_detected: str", "- last_add_time: float", "- cooldown_secs: float = 1.2", "- stability_count: int = 5"],
               ["+ add_letter(char: str, conf)", "+ insert_space()", "+ backspace()", "+ clear()", "+ get_assembled_text(): str"],
               color='#fdfefe', border='#d35400')

    # 5. LanguageService (Facade)
    draw_class(ax, 36, 8, 30, 42, "LanguageService",
               ["- translation_engine: IndicTrans", "- speech_engine: MMS_TTS", "- cache_manager: Cache", "- is_initialized: bool"],
               ["+ translate(text, src, tgt): dict", "+ speak(text, lang): bytes", "+ get_supported_languages()", "+ health_check(): dict"],
               color='#fdfefe', border='#27ae60')

    # 6. SpatialFeatureExtractor
    draw_class(ax, 71, 8, 26, 42, "SpatialFeatureExtractor",
               ["- feature_dim: int = 144", "- normalize_scales: bool", "- canonical_wrist_anchor: bool"],
               ["+ extract_dual_hand_144d()", "+ compute_finger_extension()", "+ compute_inter_palm_dist()", "+ sort_bilateral_hands()"],
               color='#fdfefe', border='#2c3e50')

    # Relationships
    # InferenceEngine -> AlphabetEngine
    ax.annotate('', xy=(36, 77), xytext=(31, 77), arrowprops=dict(arrowstyle="->", lw=1.3, color='#333333'))
    ax.text(33.5, 78.5, "1..1", fontsize=7.2, ha='center')

    # AlphabetEngine -> DecisionFusionEngine
    ax.annotate('', xy=(71, 77), xytext=(66, 77), arrowprops=dict(arrowstyle="->", lw=1.3, color='#333333'))
    ax.text(68.5, 78.5, "1..1", fontsize=7.2, ha='center')

    # AlphabetEngine -> SpatialFeatureExtractor
    ax.annotate('', xy=(75, 50), xytext=(60, 58), arrowprops=dict(arrowstyle="->", lw=1.3, color='#333333'))

    # InferenceEngine -> WordBuffer
    ax.annotate('', xy=(17, 50), xytext=(17, 58), arrowprops=dict(arrowstyle="->", lw=1.3, color='#333333'))

    # WordBuffer -> LanguageService
    ax.annotate('', xy=(36, 29), xytext=(31, 29), arrowprops=dict(arrowstyle="->", lw=1.3, color='#333333'))

    plt.title("Figure 5: Detailed UML Class Diagram of the Software Subsystems", fontsize=10.5, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig5_class_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def generate_er_diagram():
    """Figure 6: Entity-Relationship & Data Store Schema Diagram."""
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def draw_entity(ax, x, y, w, h, name, pk_list, attr_list, color='#ffffff', border='#2980b9'):
        rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor=border, lw=1.5)
        ax.add_patch(rect)
        # Header
        h_head = 4.2
        rect_head = patches.Rectangle((x, y+h-h_head), w, h_head, facecolor='#ebf5fb', edgecolor=border, lw=1.5)
        ax.add_patch(rect_head)
        ax.text(x+w/2, y+h-h_head/2, name, ha='center', va='center', fontsize=8.2, fontweight='bold', color='#1a5276')

        y_text = y + h - h_head - 2.5
        for pk in pk_list:
            ax.text(x + 1.5, y_text, f"[PK] {pk}", ha='left', va='center', fontsize=7.0, fontweight='bold', color='#c0392b')
            y_text -= 2.4
        
        y_div = y_text + 0.8
        ax.plot([x, x+w], [y_div, y_div], color='#bdc3c7', lw=1.0)
        y_text = y_div - 2.0
        for attr in attr_list:
            ax.text(x + 1.5, y_text, attr, ha='left', va='center', fontsize=7.0, color='#2c3e50')
            y_text -= 2.4

    # 1. LANDMARK_FRAME (NPY / In-Memory Stream)
    draw_entity(ax, 5, 55, 26, 38, "LANDMARK_FRAME",
                ["frame_id: UUID"],
                ["timestamp: float", "signer_id: str", "num_hands: int", "landmarks_1662: ndarray", "is_valid_pose: bool"],
                border='#2980b9')

    # 2. ALPHABET_GESTURE (Ground Truth / Lexicon)
    draw_entity(ax, 38, 55, 26, 38, "ALPHABET_GESTURE",
                ["gesture_id: int"],
                ["letter_label: char (A-Z)", "bimanual_type: enum", "canonical_extension: float", "description: str"],
                border='#27ae60')

    # 3. EVALUATION_RECORD (Audit Trail)
    draw_entity(ax, 71, 55, 26, 38, "EVALUATION_RECORD",
                ["eval_run_id: str"],
                ["checkpoint_hash: str", "test_split_name: str", "accuracy_score: float", "macro_f1: float", "eval_timestamp: datetime"],
                border='#8e44ad')

    # 4. TRANSLATION_CACHE (JSON / Memory Cache)
    draw_entity(ax, 20, 8, 28, 36, "TRANSLATION_RECORD",
                ["cache_key: str (MD5)"],
                ["source_text: str", "source_lang: str (en)", "target_lang: str", "translated_text: str", "latency_ms: float"],
                border='#d35400')

    # 5. TTS_AUDIO_BLOB (WAV Buffer Store)
    draw_entity(ax, 56, 8, 28, 36, "AUDIO_SYNTHESIS",
                ["audio_id: UUID"],
                ["cache_key: str [FK]", "sample_rate: int (16kHz)", "audio_format: str (WAV)", "audio_bytes: blob", "rtf_metric: float"],
                border='#16a085')

    # Connectors
    # LANDMARK_FRAME -- ALPHABET_GESTURE (N:1)
    ax.plot([31, 38], [74, 74], color='#333333', lw=1.3)
    ax.text(32, 75.5, "N", fontsize=7.5, fontweight='bold')
    ax.text(36.5, 75.5, "1", fontsize=7.5, fontweight='bold')

    # ALPHABET_GESTURE -- EVALUATION_RECORD (1:N)
    ax.plot([64, 71], [74, 74], color='#333333', lw=1.3)
    ax.text(65, 75.5, "1", fontsize=7.5, fontweight='bold')
    ax.text(69.5, 75.5, "N", fontsize=7.5, fontweight='bold')

    # TRANSLATION_RECORD -- AUDIO_SYNTHESIS (1:1)
    ax.plot([48, 56], [26, 26], color='#333333', lw=1.3)
    ax.text(49, 27.5, "1", fontsize=7.5, fontweight='bold')
    ax.text(54.5, 27.5, "1", fontsize=7.5, fontweight='bold')

    # ALPHABET_GESTURE -- TRANSLATION_RECORD
    ax.plot([51, 34], [55, 44], color='#333333', lw=1.3)
    ax.text(49, 51, "1", fontsize=7.5, fontweight='bold')
    ax.text(37, 43, "N", fontsize=7.5, fontweight='bold')

    plt.title("Figure 6: Entity-Relationship & Dataset Schema Model", fontsize=10.5, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig6_er_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def generate_forensics_diagram():
    """Figure 9: Forensic Analysis of Class K -> T Collapse."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    # 1. Bar chart of Finger Extensions (K train vs K test vs T train)
    categories = ['Left Index Extension', 'Right Middle Extension', 'Inter-Wrist Leveling (|Δy|)']
    k_train = [1.503, 0.720, 2.842]
    k_test = [1.123, 0.572, 2.355]
    t_train = [1.154, 0.563, 2.173]

    x = range(len(categories))
    width = 0.25

    ax1.bar([i - width for i in x], k_train, width=width, label='Class K (Train Canonical)', color='#2980b9')
    ax1.bar(x, k_test, width=width, label='Class K (Held-Out Test)', color='#e74c3c')
    ax1.bar([i + width for i in x], t_train, width=width, label='Class T (Train Canonical)', color='#27ae60')

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=8)
    ax1.set_ylabel('Geometric Value (Dimensionless / Ratio)', fontsize=8.5)
    ax1.set_title('(A) Morphological Metric Drift (K Test Mimics T Train)', fontsize=9.5, fontweight='bold')
    ax1.legend(fontsize=7.8)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # 2. Counterfactual Ablation Probabilities
    interventions = ['Baseline (Raw K Test)', 'Force Δy = -2.84 (Train Mean)', 'Force Δx,Δy,Δz = K Mean', 'Force 126D Shape = K Mean']
    prob_t = [99.3, 99.3, 99.2, 0.34]
    prob_k = [0.1, 0.1, 0.1, 99.45]

    y = range(len(interventions))
    ax2.barh([i + 0.18 for i in y], prob_t, height=0.35, label='Predicted P(Class T)', color='#e74c3c')
    ax2.barh([i - 0.18 for i in y], prob_k, height=0.35, label='Predicted P(Class K)', color='#2980b9')

    ax2.set_yticks(y)
    ax2.set_yticklabels(interventions, fontsize=7.8)
    ax2.set_xlabel('Model Confidence (%)', fontsize=8.5)
    ax2.set_title('(B) Counterfactual Inference Forensic Ablation', fontsize=9.5, fontweight='bold')
    ax2.legend(fontsize=7.8, loc='center right')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)

    plt.suptitle("Figure 9: Forensic Root-Cause Analysis of Class K → Class T Classification Collapse", fontsize=10.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig9_class_k_t_forensics.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    # Also save to old path for backward compatibility
    plt.savefig(os.path.join(FIGURES_DIR, "fig7_class_k_t_forensics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def generate_benchmark_comparison_chart():
    """Figure 8: Benchmark Performance & Model Architecture Ablation."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)

    models = [
        'Config A\n(129D Minimal)',
        'Config C\n(132D Angle)',
        'Config D-Min\n(140D Baseline)',
        'Config D-Ablation\n(144D Bilateral)'
    ]
    accuracy = [77.78, 76.98, 77.38, 78.19]
    macro_f1 = [70.75, 68.91, 69.20, 73.49]

    x = range(len(models))
    w = 0.35

    rects1 = ax.bar([i - w/2 for i in x], accuracy, w, label='Overall Accuracy (%)', color='#2980b9')
    rects2 = ax.bar([i + w/2 for i in x], macro_f1, w, label='Macro F1-Score (×100)', color='#16a085')

    ax.set_ylabel('Score (%)', fontsize=9)
    ax.set_title('Figure 8: Reconciled Held-Out Benchmark Performance Across Dual-Hand Feature Architectures', fontsize=9.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8.5)
    ax.set_ylim(60, 85)
    ax.legend(fontsize=8.5)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for r in rects1:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 0.4, f"{h:.2f}%", ha='center', va='bottom', fontsize=7.5, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2., h + 0.4, f"{h:.2f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig8_benchmark_ablation.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def generate_all_diagrams():
    ensure_dir()
    copy_existing_figures()
    generate_architecture_diagram()
    generate_dfd_diagram()
    generate_sequence_diagram()
    generate_usecase_diagram()
    generate_class_diagram()
    generate_er_diagram()
    generate_forensics_diagram()
    generate_benchmark_comparison_chart()
    print("All diagrams generated successfully in docs/report/figures/")


if __name__ == "__main__":
    generate_all_diagrams()
