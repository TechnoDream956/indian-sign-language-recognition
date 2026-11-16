"""Front matter builder for ISL Project Report.
Updates Declaration, Acknowledgement, Tools & AI Assistance, Table of Contents,
List of Tables, List of Figures, List of Abbreviations, and Abstract in-place,
strictly preserving section breaks at paragraphs 39 and 141.
"""

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph
from .xml_helpers import (
    style_table_cell, apply_table_borders,
    add_dot_leader_tab, set_page_break_before, add_todo_placeholder,
    clear_caps_and_format_run
)
from .team_config import (
    TEAM_MEMBERS,
    MENTOR_NAME,
    MENTOR_DEPARTMENT,
    INSTITUTION_NAME,
    PROJECT_START_DATE,
    PROJECT_END_DATE,
)


def build_front_matter(doc: Document):
    """Build all front matter sections in-place."""
    
    # -------------------------------------------------------------
    # 1. DECLARATION (Paragraphs 21-38, sectPr at 39)
    # -------------------------------------------------------------
    p21 = doc.paragraphs[21]
    p21.text = "DECLARATION"
    p21.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p21.runs:
        clear_caps_and_format_run(p21.runs[0], font_size=16, bold=True)

    p23 = doc.paragraphs[23]
    p23.text = (
        "We hereby declare that the work which is being presented in the report entitled "
        "“Real-Time Indian Sign Language Alphabet Recognition with Multilingual Translation and Speech Output”, "
        f"is an authentic record of our own work carried out during the period from {PROJECT_START_DATE} to {PROJECT_END_DATE} "
        f"at {MENTOR_DEPARTMENT}, {INSTITUTION_NAME}, Greater Noida."
    )
    p23.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p23.paragraph_format.line_spacing = 1.15
    p23.paragraph_format.space_before = Pt(2)
    p23.paragraph_format.space_after = Pt(2)
    if p23.runs:
        clear_caps_and_format_run(p23.runs[0], font_size=11)

    p24 = doc.paragraphs[24]
    p24.text = (
        "The matters and the results presented in this report have not been submitted by us for the award "
        "of any other degree elsewhere."
    )
    p24.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p24.paragraph_format.line_spacing = 1.15
    p24.paragraph_format.space_before = Pt(2)
    p24.paragraph_format.space_after = Pt(4)
    if p24.runs:
        clear_caps_and_format_run(p24.runs[0], font_size=11)

    # Student Signatures (4 members) from team_config
    p27 = doc.paragraphs[27]
    p27.text = "Signature of Candidates:"
    p27.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p27.paragraph_format.space_before = Pt(6)
    p27.paragraph_format.space_after = Pt(2)
    p27.paragraph_format.line_spacing = 1.0
    if p27.runs:
        clear_caps_and_format_run(p27.runs[0], font_size=11, bold=True)

    members = [(m["name"], f"Enroll. No. {m['enrollment']}") for m in TEAM_MEMBERS]

    p29 = doc.paragraphs[29]
    p29.text = f"{members[0][0]} ({members[0][1]})"
    p30 = doc.paragraphs[30]
    p30.text = f"{members[1][0]} ({members[1][1]})"
    p31 = doc.paragraphs[31]
    p31.text = f"{members[2][0]} ({members[2][1]})"
    p32 = doc.paragraphs[32]
    p32.text = f"{members[3][0]} ({members[3][1]})"

    for p in [p29, p30, p31, p32]:
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.0
        if p.runs:
            clear_caps_and_format_run(p.runs[0], font_size=10.5)

    # Paragraphs 33..38 clear
    for idx in range(33, 39):
        p = doc.paragraphs[idx]
        p.text = ""
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = Pt(0.1)
    # Paragraph 39 has sectPr - DO NOT TOUCH ITS P_ELEMENT OR SECTION BREAK

    # -------------------------------------------------------------
    # 2. ACKNOWLEDGEMENT (Paragraphs 40-64)
    # -------------------------------------------------------------
    p40 = doc.paragraphs[40]
    p40.text = "ACKNOWLEDGEMENT"
    p40.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p40.runs:
        clear_caps_and_format_run(p40.runs[0], font_size=16, bold=True)

    p42 = doc.paragraphs[42]
    p42.style = "Normal"
    p42.text = (
        "We express our gratitude to our project mentor, "
        f"{MENTOR_NAME}, {MENTOR_DEPARTMENT}, {INSTITUTION_NAME}, "
        "for guidance, constructive critique, and encouragement throughout the design, development, and evaluation "
        "of this project. Technical insights into computer vision, machine learning architectures, and academic research "
        "were instrumental in guiding our engineering decisions and methodological discipline."
    )
    p42.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p42.paragraph_format.line_spacing = 1.15
    p42.paragraph_format.space_before = Pt(2)
    p42.paragraph_format.space_after = Pt(2)
    if p42.runs:
        clear_caps_and_format_run(p42.runs[0], font_size=10.5)

    p43 = doc.paragraphs[43]
    p43.style = "Normal"
    p43.text = (
        "We also extend our sincere appreciation to the faculty, laboratory staff, and academic administration of "
        f"the {MENTOR_DEPARTMENT} at {INSTITUTION_NAME} for providing the computational "
        "infrastructure, laboratory resources, and scholarly environment essential for carrying out this project."
    )
    p43.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p43.paragraph_format.line_spacing = 1.15
    p43.paragraph_format.space_before = Pt(2)
    p43.paragraph_format.space_after = Pt(2)
    if p43.runs:
        clear_caps_and_format_run(p43.runs[0], font_size=10.5)

    p44 = doc.paragraphs[44]
    p44.style = "Normal"
    p44.text = (
        "Finally, we express our heartfelt thanks to our families, peers, and members of the deaf and hard-of-hearing "
        "community whose daily communication challenges inspired this endeavor. Their lived experiences motivated our "
        "commitment to objective evaluation in reporting model performance and building accessible assistive technology."
    )
    p44.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p44.paragraph_format.line_spacing = 1.15
    p44.paragraph_format.space_before = Pt(2)
    p44.paragraph_format.space_after = Pt(2)
    if p44.runs:
        clear_caps_and_format_run(p44.runs[0], font_size=10.5)

    p45 = doc.paragraphs[45]
    p45.style = "Normal"
    p45.text = "Signature of Candidates:"
    p45.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p45.paragraph_format.space_before = Pt(4)
    p45.paragraph_format.space_after = Pt(1)
    p45.paragraph_format.line_spacing = 1.0
    if p45.runs:
        clear_caps_and_format_run(p45.runs[0], font_size=10.5, bold=True)

    p47 = doc.paragraphs[47]
    p47.style = "Normal"
    p47.text = f"{members[0][0]} ({members[0][1]})"
    p48 = doc.paragraphs[48]
    p48.style = "Normal"
    p48.text = f"{members[1][0]} ({members[1][1]})"
    p50 = doc.paragraphs[50]
    p50.style = "Normal"
    p50.text = f"{members[2][0]} ({members[2][1]})"
    p51 = doc.paragraphs[51]
    p51.style = "Normal"
    p51.text = f"{members[3][0]} ({members[3][1]})"

    for p in [p47, p48, p50, p51]:
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.0
        if p.runs:
            clear_caps_and_format_run(p.runs[0], font_size=10.0)

    # Clear trailing paragraphs 52..55
    for idx in range(52, 56):
        p = doc.paragraphs[idx]
        p.text = ""
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = Pt(0.1)

    # Tools and AI Assistance statement (fits on Page iii)
    p56 = doc.paragraphs[56]
    p56.style = "Normal"
    p56.text = "Tools and AI Assistance:"
    p56.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p56.paragraph_format.space_before = Pt(4)
    p56.paragraph_format.space_after = Pt(1)
    p56.paragraph_format.line_spacing = 1.0
    if p56.runs:
        clear_caps_and_format_run(p56.runs[0], font_size=10.0, bold=True)

    p57 = doc.paragraphs[57]
    p57.style = "Normal"
    p57.text = (
        "We acknowledge the use of AI coding assistants for parts of code development, documentation formatting, "
        "and initial report drafting. All generated code, benchmark analyses, and technical descriptions were reviewed, "
        "tested, and verified by the authors, who take full responsibility for the contents of this report."
    )
    p57.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p57.paragraph_format.space_before = Pt(1)
    p57.paragraph_format.space_after = Pt(1)
    p57.paragraph_format.line_spacing = 1.0
    if p57.runs:
        clear_caps_and_format_run(p57.runs[0], font_size=9.5)
    add_todo_placeholder(p57, "Check college/department AI-use policy regarding declaration format and acceptable tool usage")

    for idx in range(58, 65):
        p = doc.paragraphs[idx]
        p.text = ""
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = Pt(0.1)

    # -------------------------------------------------------------
    # 3. TABLE OF CONTENTS (Paragraphs 65-95)
    # -------------------------------------------------------------
    p65 = doc.paragraphs[65]
    p65.text = "TABLE OF CONTENTS"
    p65.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_page_break_before(p65)
    if p65.runs:
        clear_caps_and_format_run(p65.runs[0], font_size=16, bold=True)

    # Strip instruction <<...>> at p66
    doc.paragraphs[66].text = ""

    # Pre-rendered TOC entries with verified Roman numerals and body page numbers
    toc_data = [
        ("LIST OF TABLES", "v", "toc 1"),
        ("LIST OF FIGURES", "vi", "toc 1"),
        ("LIST OF ABBREVIATIONS", "vii", "toc 1"),
        ("ABSTRACT", "viii", "toc 1"),
        ("1. INTRODUCTION", "1", "toc 1"),
        ("1.1. Problem Statement", "3", "toc 2"),
        ("2. BACKGROUND RESEARCH", "4", "toc 1"),
        ("2.1. Objectives", "4", "toc 2"),
        ("2.2. Literature Review & Project Feasibility", "5", "toc 2"),
        ("3. METHODOLOGY", "10", "toc 1"),
        ("3.1. Proposed System Overview", "10", "toc 2"),
        ("3.2. Users and Roles", "11", "toc 2"),
        ("3.3. Detailed Workflow", "12", "toc 2"),
        ("3.4. Design Diagrams", "15", "toc 2"),
        ("3.4.1. Use-case Diagrams", "15", "toc 3"),
        ("3.4.2. Class Diagrams", "16", "toc 3"),
        ("3.4.3. ER Diagrams", "17", "toc 3"),
        ("3.5. User Interface", "18", "toc 2"),
        ("3.5.1. User Description and Interface Mockup", "18", "toc 3"),
        ("3.6. System Implementation", "19", "toc 2"),
        ("3.6.1. Algorithms/ Pseudo Code", "19", "toc 3"),
        ("3.6.2. Source Code with Documentation", "22", "toc 3"),
        ("3.6.3. Experimental Results, Ablation Studies, and Failure Diagnostics", "24", "toc 3"),
        ("4. CONCLUSION", "30", "toc 1"),
        ("4.1. Goals/ Vision", "30", "toc 2"),
        ("4.2. Delivered Solution", "31", "toc 2"),
        ("4.3. Remaining Work", "33", "toc 2"),
        ("5. REFERENCES", "34", "toc 1")
    ]

    for idx, (title, page, style) in enumerate(toc_data):
        target_p = doc.paragraphs[67 + idx]
        target_p.text = f"{title}\t{page}"
        target_p.style = doc.styles[style] if style in doc.styles else doc.styles['Normal']
        target_p.paragraph_format.line_spacing = 1.15
        target_p.paragraph_format.space_before = Pt(1)
        target_p.paragraph_format.space_after = Pt(1)
        add_dot_leader_tab(target_p, pos_dxa=9350)
        for r in target_p.runs:
            clear_caps_and_format_run(r, font_size=10.5)

    # Clear remaining TOC paragraphs
    for idx in range(67 + len(toc_data), 96):
        doc.paragraphs[idx].text = ""

    # -------------------------------------------------------------
    # 4. LIST OF TABLES (Paragraphs 96-100 + inserted for 4 and 5)
    # -------------------------------------------------------------
    p96 = doc.paragraphs[96]
    p96.text = "LIST OF TABLES"
    p96.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_page_break_before(p96)
    if p96.runs:
        clear_caps_and_format_run(p96.runs[0], font_size=16, bold=True)

    p97 = doc.paragraphs[97]
    p97.text = "Table\tPage"
    p97.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_dot_leader_tab(p97, pos_dxa=9350)
    if p97.runs:
        clear_caps_and_format_run(p97.runs[0], font_size=11, bold=True)

    tables_data = [
        ("Table 1: System Goals and Quantitative Engineering Objectives", "5"),
        ("Table 2: Comparative Literature Review of Existing ISL Recognition Systems", "8"),
        ("Table 3: Reconciled Held-Out Benchmark Performance Across Model Architectures", "24"),
        ("Table 4: Per-Class Diagnostic Failure Modes and Skeletal Root Causes", "26"),
        ("Table 5: Measured Hardware Latency Benchmarks on Apple Silicon M2 (CPU)", "30")
    ]

    # First 3 tables in template paragraphs 98, 99, 100
    for idx in range(min(len(tables_data), 3)):
        title, page = tables_data[idx]
        target_p = doc.paragraphs[98 + idx]
        target_p.text = f"{title}\t{page}"
        target_p.paragraph_format.line_spacing = 1.15
        target_p.paragraph_format.space_before = Pt(1)
        target_p.paragraph_format.space_after = Pt(1)
        add_dot_leader_tab(target_p, pos_dxa=9350)
        for r in target_p.runs:
            clear_caps_and_format_run(r, font_size=11)

    # Insert remaining tables (4 and 5) into document XML
    last_table_p = doc.paragraphs[100]
    for idx in range(3, len(tables_data)):
        title, page = tables_data[idx]
        new_p_elm = OxmlElement('w:p')
        last_table_p._p.addnext(new_p_elm)
        new_p = Paragraph(new_p_elm, last_table_p._parent)
        new_p.text = f"{title}\t{page}"
        new_p.paragraph_format.line_spacing = 1.15
        new_p.paragraph_format.space_before = Pt(1)
        new_p.paragraph_format.space_after = Pt(1)
        add_dot_leader_tab(new_p, pos_dxa=9350)
        for r in new_p.runs:
            clear_caps_and_format_run(r, font_size=11)
        last_table_p = new_p

    table_offset = max(0, len(tables_data) - 3)

    # -------------------------------------------------------------
    # 5. LIST OF FIGURES (Paragraphs 101+table_offset to 112+table_offset)
    # -------------------------------------------------------------
    p101 = doc.paragraphs[101 + table_offset]
    p101.text = "LIST OF FIGURES"
    p101.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_page_break_before(p101)
    if p101.runs:
        clear_caps_and_format_run(p101.runs[0], font_size=16, bold=True)

    # Strip instruction <<...>> at p102, p103
    doc.paragraphs[102 + table_offset].text = ""
    doc.paragraphs[103 + table_offset].text = ""

    p104 = doc.paragraphs[104 + table_offset]
    p104.text = "Figure\tPage"
    p104.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_dot_leader_tab(p104, pos_dxa=9350)
    if p104.runs:
        clear_caps_and_format_run(p104.runs[0], font_size=11, bold=True)

    figures_data = [
        ("Figure 1: Multi-Modal Decision Fusion and Translation System Architecture", "11"),
        ("Figure 2: Data Flow Diagrams (Level 0 Context and Level 1 Process Decomposition)", "14"),
        ("Figure 3: Sequence Diagram of Live Prediction, Temporal Buffering, and Audio Synthesis", "15"),
        ("Figure 4: UML Use-Case Diagram for System Actors and Functional Scenarios", "16"),
        ("Figure 5: Detailed UML Class Diagram of the Software Subsystems", "17"),
        ("Figure 6: Logical Data Model (In-Memory and File-Based Data Structures)", "18"),
        ("Figure 7: Real-Time User Interface with Live Video Overlay and Multilingual Buffer", "19"),
        ("Figure 8: Reconciled Held-Out Benchmark Performance Across Dual-Hand Feature Architectures", "26"),
        ("Figure 9: Analysis of Class K to Class T Misclassifications", "28"),
        ("Figure 10: Skeletal Normalization and Cross-Hand Distortion Comparison", "29")
    ]

    # Populate the first min(len(figures_data), 8) into template paragraphs
    for idx in range(min(len(figures_data), 8)):
        title, page = figures_data[idx]
        target_p = doc.paragraphs[105 + table_offset + idx]
        target_p.text = f"{title}\t{page}"
        target_p.paragraph_format.line_spacing = 1.15
        target_p.paragraph_format.space_before = Pt(1)
        target_p.paragraph_format.space_after = Pt(1)
        add_dot_leader_tab(target_p, pos_dxa=9350)
        for r in target_p.runs:
            clear_caps_and_format_run(r, font_size=11)

    # Insert any remaining figures beyond 8 into document xml
    last_fig_p = doc.paragraphs[112 + table_offset]
    for idx in range(8, len(figures_data)):
        title, page = figures_data[idx]
        new_p_elm = OxmlElement('w:p')
        last_fig_p._p.addnext(new_p_elm)
        new_p = Paragraph(new_p_elm, last_fig_p._parent)
        new_p.text = f"{title}\t{page}"
        new_p.paragraph_format.line_spacing = 1.15
        new_p.paragraph_format.space_before = Pt(1)
        new_p.paragraph_format.space_after = Pt(1)
        add_dot_leader_tab(new_p, pos_dxa=9350)
        for r in new_p.runs:
            clear_caps_and_format_run(r, font_size=11)
        last_fig_p = new_p

    figure_offset = max(0, len(figures_data) - 8)
    total_offset = table_offset + figure_offset

    # -------------------------------------------------------------
    # 6. LIST OF ABBREVIATIONS (Paragraphs 113+total_offset to 137+total_offset)
    # -------------------------------------------------------------
    p113 = doc.paragraphs[113 + total_offset]
    p113.text = "LIST OF ABBREVIATIONS"
    p113.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_page_break_before(p113)
    if p113.runs:
        clear_caps_and_format_run(p113.runs[0], font_size=16, bold=True)

    p114 = doc.paragraphs[114 + total_offset]
    p114.text = "Abbreviation\tExplanation of the Abbreviation"
    add_dot_leader_tab(p114, pos_dxa=9350)
    if p114.runs:
        clear_caps_and_format_run(p114.runs[0], font_size=11, bold=True)

    abbrevs = [
        ("ISL", "Indian Sign Language"),
        ("ASL", "American Sign Language"),
        ("BSL", "British Sign Language"),
        ("MLP", "Multi-Layer Perceptron"),
        ("CNN", "Convolutional Neural Network"),
        ("GRU", "Gated Recurrent Unit"),
        ("MoP-GRU", "MediaPipe-Optimized Gated Recurrent Unit"),
        ("ROI", "Region of Interest"),
        ("SSE", "Server-Sent Events"),
        ("MJPEG", "Motion Joint Photographic Experts Group"),
        ("FPS", "Frames Per Second"),
        ("TTS", "Text-to-Speech"),
        ("NMT", "Neural Machine Translation"),
        ("ONNX", "Open Neural Network Exchange"),
        ("VITS", "Variational Inference with adversarial learning for end-to-end Text-to-Speech"),
        ("MMS", "Massively Multilingual Speech (Meta AI)"),
        ("RTF", "Real-Time Factor"),
        ("FIFO", "First-In, First-Out"),
        ("DFD", "Data Flow Diagram"),
        ("UML", "Unified Modeling Language"),
        ("ER", "Entity-Relationship"),
        ("PCA", "Principal Component Analysis")
    ]

    for idx, (abbr, desc) in enumerate(abbrevs):
        target_p = doc.paragraphs[115 + total_offset + idx]
        target_p.text = f"{abbr}\t\t\t\t\t\t{desc}"
        target_p.paragraph_format.line_spacing = 1.15
        target_p.paragraph_format.space_before = Pt(1)
        target_p.paragraph_format.space_after = Pt(1)
        for r in target_p.runs:
            clear_caps_and_format_run(r, font_size=10.0)

    # Clear remaining abbreviation paragraphs up to 137 + total_offset
    for idx in range(115 + total_offset + len(abbrevs), 138 + total_offset):
        doc.paragraphs[idx].text = ""

    # -------------------------------------------------------------
    # 7. ABSTRACT (Paragraphs 138+total_offset to 140+total_offset, sectPr at 141+total_offset)
    # -------------------------------------------------------------
    p138 = doc.paragraphs[138 + total_offset]
    p138.text = "ABSTRACT"
    p138.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_page_break_before(p138)
    if p138.runs:
        clear_caps_and_format_run(p138.runs[0], font_size=16, bold=True)

    abstract_text = (
        "Indian Sign Language (ISL) serves as the primary visual-gestural communication medium for millions of "
        "deaf and hard-of-hearing individuals across the Indian subcontinent. However, the scarcity of certified "
        "human interpreters creates societal, educational, and professional barriers. "
        "Unlike Western sign systems such as American Sign Language (ASL), ISL is structurally distinguished by extensive "
        "reliance on bimanual (two-handed) fingerspelling, subtle inter-finger contacts, and non-rigid torso coordinate "
        "interactions, rendering standard single-hand vision models ineffective.\n\n"
        "This project presents the design, implementation, and empirical evaluation of a standalone, edge-deployable "
        "real-time ISL alphabet recognition system integrated with multilingual translation and speech synthesis. "
        "The architecture introduces a multi-modal decision framework that couples kinematic skeletal landmarks extracted "
        "via Google MediaPipe Hands (21 3D landmarks per hand) with an RGB appearance Convolutional Neural Network (CNN, 128x128x3) "
        "operating on cropped regions of interest. Hand-presence routing dispatches frames between single-hand invariant multi-layer "
        "perceptrons (MLPs, 63D) and specialized dual-hand spatial feature extractors (144 invariant geometric dimensions). "
        "A temporal WordBuffer state machine applies dynamic debounce cooldowns (1.2 s) and stability voting to assemble "
        "spelled characters into complete words without physical keyboard interaction.\n\n"
        "Downstream linguistic accessibility supports translation into 10 regional Indian languages via dictionary mapping "
        "with online fallback, while an int8-quantized IndicTrans2 model is implemented in the codebase as an optional offline "
        "neural backend. Auditory feedback uses a multi-tier speech fallback chain (macOS say -> pyttsx3 -> gTTS -> Web Speech API), "
        "with Meta's Massively Multilingual Speech (MMS-TTS) implemented as an optional neural service.\n\n"
        "Offline empirical evaluation was conducted on controlled same-signer held-out test partitions. The clean dual-hand "
        "spatial MLP achieves 78.19% Top-1 accuracy and 0.7349 Macro F1 on the held-out test partition (N=243), while the paired "
        "visual CNN achieves 82.72% (0.7565 Macro F1). Multi-modal decision fusion without heuristic overrides matches the geometry "
        "baseline at 78.19%. A post-hoc K/T orientation override yielded 83.13% Top-1 accuracy (+4.94%), though this heuristic is "
        "possibly tuned on test data. An engineering diagnosis identified critical pipeline fragility: camera mirroring inverts coordinate "
        "geometry, causing a 48.97% drop in accuracy (from 78.19% to 29.22% on held-out landmark data; live validation pending), while intermittent occlusion drops were mitigated using "
        "a 2-frame temporal hysteresis filter (-85.7% flapping on a synthetic sequence). Formal live accuracy across novel signers remains "
        "pending future multi-signer benchmark recording."
    )

    p139 = doc.paragraphs[139 + total_offset]
    p139.text = abstract_text
    p139.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p139.paragraph_format.line_spacing = 1.15
    p139.paragraph_format.space_before = Pt(2)
    p139.paragraph_format.space_after = Pt(2)
    if p139.runs:
        clear_caps_and_format_run(p139.runs[0], font_size=10.0)

    doc.paragraphs[140 + total_offset].text = ""
    # Paragraph 141 has sectPr - PRESERVE WITHOUT ALTERING P_ELEMENT

    print("Front matter updated successfully.")
