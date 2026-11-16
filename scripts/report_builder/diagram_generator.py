"""Publication-quality figure and diagram generator for ISL Project Report.
Generates all formal architecture, workflow, UML, ER, and diagnostic figures.
Strictly removes baked-in 'Figure N:' titles from images and ensures privacy.
"""

import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../docs/report/figures"))


def ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def copy_existing_figures():
    """Copy or sync existing figures into figures directory."""
    # UI screenshot: sync from tracked blurred ui_live_running_desktop.png
    ui_src = os.path.join(FIGURES_DIR, "ui_live_running_desktop.png")
    if os.path.exists(ui_src):
        shutil.copy2(ui_src, os.path.join(FIGURES_DIR, "fig_ui_screenshot.png"))
        print("Synced UI live running desktop screenshot.")


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
    draw_box(10, 75, 36, 8, "MediaPipe Hands Tracking", "21 3D Landmarks per Hand (x, y, z)", color='#eafaf1', edgecolor=c_green)
    draw_box(54, 75, 36, 8, "Spatial ROI Extraction", "Hand Bounding Box & 128x128 Crop", color='#fef5e7', edgecolor=c_orange)

    draw_arrow(43, 90, 28, 83)
    draw_arrow(57, 90, 72, 83)

    # 3. Model Backbones
    draw_box(4, 56, 26, 11, "Single-Hand MLP", "63D Invariant Coordinates\nDense(128-64) + Softmax (26C)", color='#ffffff', edgecolor=c_green)
    draw_box(32, 56, 30, 11, "Dual-Hand Spatial MLP", "144D Invariant Features\nDeep MLP(256-128-64) (20C)", color='#ffffff', edgecolor=c_teal)
    draw_box(64, 56, 32, 11, "RGB Appearance CNN", "4x Conv2D + GAP (128x128)\nColor / Texture Backbone (26C)", color='#ffffff', edgecolor=c_orange)

    draw_arrow(23, 75, 17, 67, "1 Hand")
    draw_arrow(33, 75, 47, 67, "2 Hands")
    draw_arrow(72, 75, 80, 67, "128x128 ROI")

    # 4. Multi-Modal Decision Fusion
    draw_box(18, 38, 64, 11, "Confidence-Aware Decision Fusion Engine", 
             "Calibrated Weights (W_geom=0.65, W_cnn=0.35) + Agreement Bonus (+0.10)\nLandmark Geometric Authority • Confidence Threshold 0.50", 
             color='#f5eef8', edgecolor=c_purple, lw=2.0)

    draw_arrow(17, 56, 35, 49)
    draw_arrow(47, 56, 50, 49)
    draw_arrow(80, 56, 65, 49)

    # 5. Temporal Buffer State Machine
    draw_box(20, 22, 60, 9, "WordBuffer Temporal State Machine", 
             "Temporal Cooldown Debounce (1.2 s) • Stability Threshold (5 Frames)\nSpace Insertion • Backspace • Letter Latching", 
             color='#fbfcfc', edgecolor=c_blue)

    draw_arrow(50, 38, 50, 31, "Verified Character")

    # 6. Downstream Subsystems (Translation & Speech)
    draw_box(4, 5, 44, 10, "Curated Dictionary & Lexicon (Default)", "Instantaneous (<0.01ms) + MyMemory Fallback\n10 Regional Indian Languages (Optional: IndicTrans2)", color='#e8f8f5', edgecolor='#16a085')
    draw_box(52, 5, 44, 10, "Multi-Tier Speech Engine (Default)", "macOS say -> pyttsx3 -> gTTS -> Web Speech API\nSub-100ms Synthesis Launch (Optional: MMS-TTS)", color='#f4ecf7', edgecolor='#8e44ad')

    draw_arrow(38, 22, 26, 15, "English Sign Buffer")
    draw_arrow(62, 22, 74, 15, "Translated Text")

    plt.title("Multi-Modal Decision Fusion and Translation Architecture", fontsize=11, fontweight='bold', pad=15)
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
    ax1.annotate('Translated Text &\nAudible Speech Audio', xy=(76, 47.5), xytext=(68, 47.5),
                arrowprops=dict(arrowstyle="->", color='#333333', lw=1.5), fontsize=8, ha='center', va='bottom')

    # --- DFD Level 1 (Decomposition) ---
    ax2.text(50, 96, "Data Flow Diagram — Level 1 (Process Decomposition)", ha='center', fontsize=10, fontweight='bold')

    def pbox(ax, x, y, num, text, color='#f4f6f6', border=c_blue):
        rect = patches.FancyBboxPatch((x, y), 18, 16, boxstyle="round,pad=0.5", facecolor=color, edgecolor=border, lw=1.5)
        ax.add_patch(rect)
        ax.text(x + 9, y + 11.5, num, ha='center', va='center', fontsize=8, fontweight='bold', color=border)
        ax.text(x + 9, y + 5.5, text, ha='center', va='center', fontsize=7.5, color='#222222')

    pbox(ax2, 3, 40, "1.0", "Frame Capture &\nMediaPipe Tracking", color='#ebf5fb', border=c_blue)
    pbox(ax2, 28, 62, "2.0", "Feature Extraction\n& Invariant Norm", color='#eafaf1', border=c_green)
    pbox(ax2, 28, 18, "3.0", "Appearance ROI\nCrop (128x128 CNN)", color='#fef5e7', border=c_orange)
    pbox(ax2, 53, 40, "4.0", "Multi-Modal Fusion\n& Temporal Buffer", color='#f5eef8', border='#8e44ad')
    pbox(ax2, 78, 40, "5.0", "Dictionary NMT &\nMulti-Tier TTS", color='#e8f8f5', border='#16a085')

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
        (71, 52, 52, "3: DecisionFusion (W_geom=0.65, W_cnn=0.35, +0.10)", False),
        (63, 52, 72, "4: update_buffer(char, conf >= 0.50)", True),
        (55, 72, 72, "5: debounce_check(cooldown=1.2s, N=5)", False),
        (47, 72, 32, "6: push_event('letter', 'W') via SSE", True),
        (39, 32, 12, "7: render_overlay() & update_dom()", True),
        (30, 72, 90, "8: translate_sentence('WATER') [10 Languages]", True),
        (22, 90, 90, "9: Dictionary / MyMemory + Multi-Tier Speech Engine", False),
        (14, 90, 12, "10: audio_playback(Local Voice / say / pyttsx3 / gTTS)", True)
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

    plt.title("Sequence Diagram of Live Prediction, Temporal Buffering, and Audio Synthesis", fontsize=10.5, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig3_sequence_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def generate_usecase_diagram():
    """Figure 4: UML Use-Case Diagram."""
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Boundary Box
    rect_bound = patches.Rectangle((22, 5), 56, 88, facecolor='#fdfefe', edgecolor='#2c3e50', lw=1.5, linestyle='-')
    ax.add_patch(rect_bound)
    ax.text(50, 90, "Indian Sign Language Recognition & Translation System", ha='center', fontsize=9.5, fontweight='bold', color='#2c3e50')

    # Actors
    def draw_actor(ax, x, y, name):
        circle = patches.Circle((x, y+4), 2.2, facecolor='#ffffff', edgecolor='#34495e', lw=1.5)
        ax.add_patch(circle)
        ax.plot([x, x], [y+1.8, y-4], color='#34495e', lw=1.5)
        ax.plot([x-3, x+3], [y-0.5, y-0.5], color='#34495e', lw=1.5)
        ax.plot([x, x-2.5], [y-4, y-9], color='#34495e', lw=1.5)
        ax.plot([x, x+2.5], [y-4, y-9], color='#34495e', lw=1.5)
        ax.text(x, y-12, name, ha='center', fontsize=8, fontweight='bold', color='#2c3e50')

    draw_actor(ax, 10, 65, "Deaf Signer\n(Primary)")
    draw_actor(ax, 90, 55, "Hearing\nInterlocutor")
    draw_actor(ax, 10, 22, "System Admin /\nEvaluator")

    # Use Cases (Ovals)
    usecases = [
        (50, 81, "Perform ISL Fingerspelling"),
        (50, 69, "View Real-Time Landmark Feedback"),
        (50, 56, "Assemble Words via WordBuffer"),
        (50, 43, "Translate Sign Buffer (10 Languages)"),
        (50, 30, "Synthesize Speech Output (Multi-Tier)"),
        (50, 17, "Run Accuracy Audit & Benchmark Diagnostics")
    ]

    for x, y, text in usecases:
        ellipse = patches.Ellipse((x, y), 38, 8.5, facecolor='#ebf5fb', edgecolor='#2980b9', lw=1.3)
        ax.add_patch(ellipse)
        ax.text(x, y, text, ha='center', va='center', fontsize=7.8, color='#1b4f72')

    # Connections
    # Deaf signer connects to 0, 1, 2
    for y in [81, 69, 56]:
        ax.plot([12, 31], [65, y], color='#34495e', lw=1.2)

    # Admin connects to 5
    ax.plot([12, 31], [22, 17], color='#34495e', lw=1.2)

    # Hearing interlocutor connects to 1, 3, 4
    for y in [69, 43, 30]:
        ax.plot([88, 69], [55, y], color='#34495e', lw=1.2)

    plt.title("UML Use-Case Diagram for System Actors and Functional Scenarios", fontsize=10.5, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig4_usecase_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def generate_class_diagram():
    """Figure 5: Detailed UML Class Diagram."""
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def draw_uml_class(ax, x, y, w, h, class_name, attributes, methods, color='#ffffff', border='#2980b9'):
        rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor=border, lw=1.4)
        ax.add_patch(rect)
        # Title compartment
        h_title = 4.0
        rect_title = patches.Rectangle((x, y+h-h_title), w, h_title, facecolor='#ebf5fb', edgecolor=border, lw=1.4)
        ax.add_patch(rect_title)
        ax.text(x+w/2, y+h-h_title/2, class_name, ha='center', va='center', fontsize=8.2, fontweight='bold', color='#1a5276')

        # Attributes
        y_text = y + h - h_title - 2.2
        for attr in attributes:
            ax.text(x + 1.2, y_text, attr, ha='left', va='center', fontsize=6.8, color='#333333')
            y_text -= 2.2

        # Divider
        ax.plot([x, x+w], [y_text+0.8, y_text+0.8], color=border, lw=0.8)
        y_text -= 1.6

        # Methods
        for m in methods:
            ax.text(x + 1.2, y_text, m, ha='left', va='center', fontsize=6.8, color='#1e8449')
            y_text -= 2.2

    # InferenceEngine
    draw_uml_class(ax, 34, 62, 32, 32, "InferenceEngine",
                   ["- camera_index: int", "- alpha_engine: AlphabetEngine", "- word_buffer: WordBuffer", "- is_mirrored: bool"],
                   ["+ process_frame(frame: ndarray)", "+ get_current_telemetry(): dict", "+ reset_buffer(): void"])

    # AlphabetEngine
    draw_uml_class(ax, 2, 60, 29, 34, "AlphabetEngine",
                   ["- single_model: KerasModel", "- dual_engine: DualHandEngine", "- cnn_engine: CNNEngine", "- fusion: FusionEngine"],
                   ["+ process_dual_landmarks()", "+ classify_single_hand()", "+ classify_cnn_crop()", "+ route_gesture()"])

    # DualHandEngine
    draw_uml_class(ax, 2, 16, 29, 36, "DualHandEngine",
                   ["- model_path: Path", "- config_key: str", "- expected_dim: 144", "- classes: list[20]"],
                   ["+ compute_features(c1, c2)", "+ predict(c1, c2): dict", "+ normalize_coords()", "+ extract_apertures()"])

    # FusionEngine
    draw_uml_class(ax, 36, 16, 30, 36, "FusionEngine",
                   ["- landmark_weight: 0.65", "- cnn_weight: 0.35", "- min_thresh: 0.50", "- agreement_bonus: 0.10"],
                   ["+ fuse(lm_res, cnn_res): dict", "+ check_agreement(): bool", "+ arbitrate_disagreement()", "+ reject_uncertain()"])

    # Language & Speech
    draw_uml_class(ax, 70, 16, 28, 36, "Translator & TTSEngine",
                   ["- translation_cache: dict", "- target_lang: str (10)", "- speech_engine: multi-tier", "- tts_tiers: list[4]"],
                   ["+ translate(phrase, lang)", "+ synthesize_speech(text)", "+ query_mymemory()", "+ fallback_browser_tts()"])

    # WordBuffer
    draw_uml_class(ax, 70, 62, 28, 32, "WordBuffer",
                   ["- buffer: list[str]", "- last_candidate: str", "- stability_count: int", "- cooldown_tau: 1.2s"],
                   ["+ add_character(char, conf)", "+ insert_space(): void", "+ backspace(): void", "+ clear(): void"])

    # Connections
    ax.annotate('', xy=(34, 78), xytext=(31, 78), arrowprops=dict(arrowstyle="->", lw=1.3, color='#333333'))
    ax.annotate('', xy=(70, 78), xytext=(66, 78), arrowprops=dict(arrowstyle="->", lw=1.3, color='#333333'))
    ax.annotate('', xy=(17, 52), xytext=(17, 60), arrowprops=dict(arrowstyle="->", lw=1.3, color='#333333'))
    ax.annotate('', xy=(36, 34), xytext=(31, 34), arrowprops=dict(arrowstyle="->", lw=1.3, color='#333333'))
    ax.annotate('', xy=(70, 34), xytext=(66, 34), arrowprops=dict(arrowstyle="->", lw=1.3, color='#333333'))

    plt.title("Detailed UML Class Diagram of the Software Subsystems", fontsize=10.5, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig5_class_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def generate_er_diagram():
    """Figure 6: Logical Data Model Diagram (In-Memory and File-Based Data Structures)."""
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
            ax.text(x + 1.5, y_text, f"[KEY] {pk}", ha='left', va='center', fontsize=7.0, fontweight='bold', color='#c0392b')
            y_text -= 2.4
        
        y_div = y_text + 0.8
        ax.plot([x, x+w], [y_div, y_div], color='#bdc3c7', lw=1.0)
        y_text = y_div - 2.0
        for attr in attr_list:
            ax.text(x + 1.5, y_text, attr, ha='left', va='center', fontsize=7.0, color='#2c3e50')
            y_text -= 2.4

    # 1. FRAME_LANDMARKS (In-Memory Pipeline Stream)
    draw_entity(ax, 5, 55, 27, 38, "FRAME_LANDMARKS",
                ["frame_id: int"],
                ["timestamp: float", "hand_count: int", "features_63d: ndarray", "features_144d: ndarray", "is_valid_pose: bool"],
                border='#2980b9')

    # 2. ALPHABET_GESTURE (Canonical Lexicon)
    draw_entity(ax, 38, 55, 26, 38, "ALPHABET_GESTURE",
                ["class_id: int (0..25)"],
                ["letter_label: char (A-Z)", "bimanual_type: enum", "target_pool: enum", "description: str"],
                border='#27ae60')

    # 3. EVALUATION_SAMPLE (Offline Test Data)
    draw_entity(ax, 71, 55, 26, 38, "EVALUATION_SAMPLE",
                ["sample_id: str"],
                ["dataset_source: str", "ground_truth: char", "predicted_label: char", "confidence: float", "is_correct: bool"],
                border='#8e44ad')

    # 4. TRANSLATION_RECORD (data/translation_cache.json)
    draw_entity(ax, 20, 8, 28, 36, "TRANSLATION_RECORD",
                ["cache_key: str (phrase)"],
                ["source_phrase: str", "target_lang: str (10)", "translated_text: str", "cached_at: float"],
                border='#d35400')

    # 5. TTS_AUDIO_ENTRY (Multi-Tier Audio Cache)
    draw_entity(ax, 56, 8, 28, 36, "TTS_AUDIO_ENTRY",
                ["audio_path: str"],
                ["cache_key: str [FK]", "language_code: str", "tts_engine_tier: str", "audio_format: str (WAV)"],
                border='#16a085')

    # Connectors
    # FRAME_LANDMARKS -- ALPHABET_GESTURE (N:1)
    ax.plot([32, 38], [74, 74], color='#333333', lw=1.3)
    ax.text(33, 75.5, "N", fontsize=7.5, fontweight='bold')
    ax.text(36.5, 75.5, "1", fontsize=7.5, fontweight='bold')

    # ALPHABET_GESTURE -- EVALUATION_SAMPLE (1:N)
    ax.plot([64, 71], [74, 74], color='#333333', lw=1.3)
    ax.text(65, 75.5, "1", fontsize=7.5, fontweight='bold')
    ax.text(69.5, 75.5, "N", fontsize=7.5, fontweight='bold')

    # TRANSLATION_RECORD -- TTS_AUDIO_ENTRY (1:1)
    ax.plot([48, 56], [26, 26], color='#333333', lw=1.3)
    ax.text(49, 27.5, "1", fontsize=7.5, fontweight='bold')
    ax.text(54.5, 27.5, "1", fontsize=7.5, fontweight='bold')

    # ALPHABET_GESTURE -- TRANSLATION_RECORD
    ax.plot([51, 34], [55, 44], color='#333333', lw=1.3)
    ax.text(49, 51, "1", fontsize=7.5, fontweight='bold')
    ax.text(37, 43, "N", fontsize=7.5, fontweight='bold')

    plt.title("Logical Data Model (In-Memory and File-Based Data Structures)", fontsize=10.5, fontweight='bold', pad=15)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig6_er_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_path}")


def generate_forensics_diagram():
    """Figure 9: Analysis of Class K to Class T Misclassifications."""
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
    ax2.set_title('(B) Counterfactual Inference Ablation', fontsize=9.5, fontweight='bold')
    ax2.legend(fontsize=7.8, loc='center right')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)

    plt.suptitle("Analysis of Class K to Class T Misclassifications", fontsize=10.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig9_class_k_t_forensics.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
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
    ax.set_title('Reconciled Held-Out Benchmark Performance Across Dual-Hand Feature Architectures', fontsize=9.5, fontweight='bold')
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


def generate_normalization_comparison_diagram():
    """Figure 10: Skeletal Normalization and Cross-Hand Distortion Comparison.
    Regenerated with pure skeleton diagrams only (no human photos, no internal titles).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=300)

    for ax in (ax1, ax2):
        ax.set_xlim(-15, 115)
        ax.set_ylim(-15, 105)
        ax.axis('off')

    # Canonical 2D hand joint offsets from wrist (0, 0)
    # [Wrist(0), Thumb(1-4), Index(5-8), Middle(9-12), Ring(13-16), Pinky(17-20)]
    hand_template = np.array([
        [0.0, 0.0],    # 0: Wrist
        [-4.0, 5.0],   # 1: Thumb CMC
        [-7.0, 10.0],  # 2: Thumb MCP
        [-9.0, 15.0],  # 3: Thumb IP
        [-10.0, 20.0], # 4: Thumb Tip
        [-4.0, 14.0],  # 5: Index MCP
        [-5.0, 20.0],  # 6: Index PIP
        [-5.5, 25.0],  # 7: Index DIP
        [-6.0, 30.0],  # 8: Index Tip
        [0.0, 15.0],   # 9: Middle MCP
        [0.0, 22.0],   # 10: Middle PIP
        [0.0, 28.0],   # 11: Middle DIP
        [0.0, 34.0],   # 12: Middle Tip
        [4.0, 14.0],   # 13: Ring MCP
        [4.5, 20.0],   # 14: Ring PIP
        [5.0, 25.0],   # 15: Ring DIP
        [5.5, 29.0],   # 16: Ring Tip
        [8.0, 12.0],   # 17: Pinky MCP
        [9.5, 17.0],   # 18: Pinky PIP
        [10.5, 21.0],  # 19: Pinky DIP
        [11.5, 25.0],  # 20: Pinky Tip
    ])

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)              # Metacarpal arch
    ]

    def draw_skeleton(ax, origin_x, origin_y, scale=1.0, mirror=False, joint_color='#2980b9', bone_color='#3498db'):
        pts = hand_template.copy()
        if mirror:
            pts[:, 0] = -pts[:, 0]
        pts = pts * scale
        pts[:, 0] += origin_x
        pts[:, 1] += origin_y

        for i, j in connections:
            ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=bone_color, lw=2.0, zorder=2)
        ax.scatter(pts[:, 0], pts[:, 1], color=joint_color, s=22, zorder=3)
        return pts

    # =========================================================================
    # Panel A: Unified Bounding Box Normalization (Flawed Legacy)
    # =========================================================================
    ax1.text(50, 98, "(A) Unified Bounding Box Normalization (Legacy Flaw)", ha='center', fontsize=9.5, fontweight='bold', color='#c0392b')

    # Draw widely separated hands
    left_pts = draw_skeleton(ax1, 20, 30, scale=0.8, mirror=False, joint_color='#c0392b', bone_color='#e74c3c')
    right_pts = draw_skeleton(ax1, 80, 25, scale=0.8, mirror=True, joint_color='#c0392b', bone_color='#e74c3c')

    # Single giant bounding box encompassing both hands
    box_x1 = min(np.min(left_pts[:, 0]), np.min(right_pts[:, 0])) - 6
    box_x2 = max(np.max(left_pts[:, 0]), max(right_pts[:, 0])) + 6
    box_y1 = min(np.min(left_pts[:, 1]), np.min(right_pts[:, 1])) - 6
    box_y2 = max(np.max(left_pts[:, 1]), max(right_pts[:, 1])) + 6

    rect_unified = patches.Rectangle((box_x1, box_y1), box_x2 - box_x1, box_y2 - box_y1,
                                     facecolor='#fdedec', edgecolor='#c0392b', lw=2.0, linestyle='--', zorder=1)
    ax1.add_patch(rect_unified)

    # Separation dimension line
    ax1.annotate('', xy=(80, 15), xytext=(20, 15),
                 arrowprops=dict(arrowstyle="<->", color='#c0392b', lw=1.5))
    ax1.text(50, 10, "Wide Hand Separation (Variable ΔX)", ha='center', fontsize=7.8, fontweight='bold', color='#922b21')

    # Bounding box scale label
    ax1.text(50, box_y2 + 2, "Unified Scale Denominator: S = max(ΔX_box, ΔY_box)", ha='center', fontsize=7.8, fontweight='bold', color='#c0392b')

    # Diagnostic text callout box
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor='#ffffff', edgecolor='#c0392b', lw=1.2)
    ax1.text(50, -8, "FLAW: Moving hands apart inflates S, collapsing relative\n"
                     "finger coordinates (P / S -> 0) and distorting true handshape.",
             ha='center', va='center', fontsize=7.5, color='#78281f', bbox=bbox_props)

    # =========================================================================
    # Panel B: Independent Skeletal Normalization (Current Invariant Pipeline)
    # =========================================================================
    ax2.text(50, 98, "(B) Independent Skeletal Normalization (Reconciled)", ha='center', fontsize=9.5, fontweight='bold', color='#1e8449')

    # Draw hands
    left_pts2 = draw_skeleton(ax2, 20, 30, scale=0.8, mirror=False, joint_color='#1e8449', bone_color='#27ae60')
    right_pts2 = draw_skeleton(ax2, 80, 25, scale=0.8, mirror=True, joint_color='#1e8449', bone_color='#27ae60')

    # Independent hand boxes
    for pts, name, w_x, w_y, m_x, m_y in [
        (left_pts2, "Left Hand (63D)", 20, 30, left_pts2[9, 0], left_pts2[9, 1]),
        (right_pts2, "Right Hand (63D)", 80, 25, right_pts2[9, 0], right_pts2[9, 1])
    ]:
        bx1, bx2 = np.min(pts[:, 0]) - 3, np.max(pts[:, 0]) + 3
        by1, by2 = np.min(pts[:, 1]) - 3, np.max(pts[:, 1]) + 3
        rect_ind = patches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1,
                                     facecolor='#eafaf1', edgecolor='#27ae60', lw=1.5, linestyle='-', zorder=1)
        ax2.add_patch(rect_ind)
        # Metacarpal scale line (Wrist to Middle MCP)
        ax2.plot([w_x, m_x], [w_y, m_y], color='#d35400', lw=2.5, zorder=4)
        ax2.text((w_x + m_x)/2 - 5, (w_y + m_y)/2, "s_h", fontsize=8, fontweight='bold', color='#d35400')
        ax2.text((bx1 + bx2)/2, by2 + 2, name, ha='center', fontsize=7.5, fontweight='bold', color='#145a32')

    # Explicit relative wrist vector
    ax2.annotate('', xy=(80, 25), xytext=(20, 30),
                 arrowprops=dict(arrowstyle="->", color='#2980b9', lw=2.0))
    ax2.text(50, 32, "Explicit Relative Wrist: ΔW = (W2 - W1) / s_ref", ha='center', fontsize=7.5, fontweight='bold', color='#1a5276')

    # Diagnostic text callout box
    bbox_props2 = dict(boxstyle="round,pad=0.5", facecolor='#ffffff', edgecolor='#27ae60', lw=1.2)
    ax2.text(50, -8, "SOLUTION: Each hand scaled by its own metacarpal bone (s_h);\n"
                     "finger geometry remains 100% invariant to hand separation.",
             ha='center', va='center', fontsize=7.5, color='#145a32', bbox=bbox_props2)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_normalization_flaw.png")
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
    generate_normalization_comparison_diagram()
    print("All diagrams generated successfully in docs/report/figures/")


if __name__ == "__main__":
    generate_all_diagrams()
