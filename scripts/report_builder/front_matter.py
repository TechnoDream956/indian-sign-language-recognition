"""Front matter builder for ISL Project Report.
Updates Declaration, Acknowledgement, Table of Contents, List of Tables,
List of Figures, List of Abbreviations, and Abstract in-place,
strictly preserving section breaks at paragraphs 39 and 141.
"""

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.text.paragraph import Paragraph
from .xml_helpers import style_table_cell, apply_table_borders
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
        p21.runs[0].font.name = 'Times New Roman'
        p21.runs[0].font.size = Pt(16)
        p21.runs[0].bold = True

    p23 = doc.paragraphs[23]
    p23.text = (
        "We hereby declare that the work which is being presented in the report entitled "
        "“REAL-TIME INDIAN SIGN LANGUAGE ALPHABET RECOGNITION WITH MULTILINGUAL TRANSLATION AND SPEECH OUTPUT”, "
        f"is an authentic record of our own work carried out during the period from {PROJECT_START_DATE} to {PROJECT_END_DATE} "
        f"at {MENTOR_DEPARTMENT}, {INSTITUTION_NAME}, Greater Noida."
    )
    p23.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if p23.runs:
        p23.runs[0].font.name = 'Times New Roman'
        p23.runs[0].font.size = Pt(12)

    p24 = doc.paragraphs[24]
    p24.text = (
        "The matters and the results presented in this report have not been submitted by us for the award "
        "of any other degree elsewhere."
    )
    p24.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if p24.runs:
        p24.runs[0].font.name = 'Times New Roman'
        p24.runs[0].font.size = Pt(12)

    # Student Signatures (4 members) from team_config
    p27 = doc.paragraphs[27]
    p27.text = "Signature of Candidates:"
    p27.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if p27.runs:
        p27.runs[0].font.name = 'Times New Roman'
        p27.runs[0].font.size = Pt(12)
        p27.runs[0].bold = True

    members = [(m["name"], f"Enroll. No. {m['enrollment']}") for m in TEAM_MEMBERS]

    p29 = doc.paragraphs[29]
    p29.text = members[0][0]
    p30 = doc.paragraphs[30]
    p30.text = f"({members[0][1]})"

    p31 = doc.paragraphs[31]
    p31.text = members[1][0]
    p32 = doc.paragraphs[32]
    p32.text = f"({members[1][1]})"

    p33 = doc.paragraphs[33]
    p33.text = members[2][0]
    p34 = doc.paragraphs[34]
    p34.text = f"({members[2][1]})"

    # Add member 4 into p35 and p36
    p35 = doc.paragraphs[35]
    p35.text = members[3][0]
    p36 = doc.paragraphs[36]
    p36.text = f"({members[3][1]})"

    for p in [p29, p30, p31, p32, p33, p34, p35, p36]:
        if p.runs:
            p.runs[0].font.name = 'Times New Roman'
            p.runs[0].font.size = Pt(11)

    # Paragraph 37, 38 clear
    doc.paragraphs[37].text = ""
    doc.paragraphs[38].text = ""
    # Paragraph 39 has sectPr - DO NOT TOUCH ITS P_ELEMENT OR SECTION BREAK

    # -------------------------------------------------------------
    # 2. ACKNOWLEDGEMENT (Paragraphs 40-64)
    # -------------------------------------------------------------
    p40 = doc.paragraphs[40]
    p40.text = "ACKNOWLEDGEMENT"
    p40.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p40.runs:
        p40.runs[0].font.name = 'Times New Roman'
        p40.runs[0].font.size = Pt(16)
        p40.runs[0].bold = True

    p42 = doc.paragraphs[42]
    p42.text = (
        "We would like to take this opportunity to express our deepest gratitude to our project mentor, "
        f"{MENTOR_NAME}, {MENTOR_DEPARTMENT}, {INSTITUTION_NAME}, "
        "for guidance, constructive critique, and encouragement throughout the design, development, and evaluation "
        "of this project. Technical insights into computer vision, machine learning architectures, and academic research rigor "
        "were instrumental in guiding our engineering decisions and methodological discipline."
    )
    p42.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if p42.runs:
        p42.runs[0].font.name = 'Times New Roman'
        p42.runs[0].font.size = Pt(12)

    p43 = doc.paragraphs[43]
    p43.text = (
        "We also extend our sincere appreciation to the faculty, laboratory staff, and academic administration of "
        f"the {MENTOR_DEPARTMENT} at {INSTITUTION_NAME} for providing the computational "
        "infrastructure, laboratory resources, and scholarly environment essential for carrying out this research."
    )
    p43.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if p43.runs:
        p43.runs[0].font.name = 'Times New Roman'
        p43.runs[0].font.size = Pt(12)

    p44 = doc.paragraphs[44]
    p44.text = (
        "Finally, we express our heartfelt thanks to our families, peers, and members of the deaf and hard-of-hearing "
        "community whose daily communication challenges inspired this endeavor. Their lived experiences motivated our "
        "commitment to thorough honesty in reporting model performance and building accessible assistive technology."
    )
    p44.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if p44.runs:
        p44.runs[0].font.name = 'Times New Roman'
        p44.runs[0].font.size = Pt(12)

    p45 = doc.paragraphs[45]
    p45.text = "Signature of Candidates:"
    p45.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if p45.runs:
        p45.runs[0].font.name = 'Times New Roman'
        p45.runs[0].font.size = Pt(12)
        p45.runs[0].bold = True

    p47 = doc.paragraphs[47]
    p47.text = members[0][0]
    p48 = doc.paragraphs[48]
    p48.text = f"({members[0][1]})"

    p50 = doc.paragraphs[50]
    p50.text = members[1][0]
    p51 = doc.paragraphs[51]
    p51.text = f"({members[1][1]})"

    p52 = doc.paragraphs[52]
    p52.text = members[2][0]
    p53 = doc.paragraphs[53]
    p53.text = f"({members[2][1]})"

    p54 = doc.paragraphs[54]
    p54.text = members[3][0]
    p55 = doc.paragraphs[55]
    p55.text = f"({members[3][1]})"

    # Clear trailing paragraphs up to 64
    for idx in range(56, 65):
        doc.paragraphs[idx].text = ""

    # -------------------------------------------------------------
    # 3. TABLE OF CONTENTS (Paragraphs 65-95)
    # -------------------------------------------------------------
    p65 = doc.paragraphs[65]
    p65.text = "TABLE OF CONTENTS"
    p65.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p65.runs:
        p65.runs[0].font.name = 'Times New Roman'
        p65.runs[0].font.size = Pt(16)
        p65.runs[0].bold = True

    # Strip instruction <<...>> at p66
    doc.paragraphs[66].text = ""

    # Pre-rendered TOC entries (clean, no template instruction pollution)
    toc_data = [
        ("LIST OF TABLES", "vi", "toc 1"),
        ("LIST OF FIGURES", "vii", "toc 1"),
        ("LIST OF ABBREVIATIONS", "viii", "toc 1"),
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
        for r in target_p.runs:
            r.font.name = 'Times New Roman'

    # Clear remaining TOC paragraphs
    for idx in range(67 + len(toc_data), 96):
        doc.paragraphs[idx].text = ""

    # -------------------------------------------------------------
    # 4. LIST OF TABLES (Paragraphs 96-100)
    # -------------------------------------------------------------
    p96 = doc.paragraphs[96]
    p96.text = "LIST OF TABLES"
    p96.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p96.runs:
        p96.runs[0].font.name = 'Times New Roman'
        p96.runs[0].font.size = Pt(16)
        p96.runs[0].bold = True

    p97 = doc.paragraphs[97]
    p97.text = "Table\tPage"
    p97.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if p97.runs:
        p97.runs[0].font.name = 'Times New Roman'
        p97.runs[0].font.size = Pt(11)
        p97.runs[0].bold = True

    tables_data = [
        ("Table 1: System Goals and Quantitative Engineering Objectives", "5"),
        ("Table 2: Comparative Literature Review of State-of-the-Art ISL Recognition Systems", "8"),
        ("Table 3: Reconciled Held-Out Benchmark Performance Across Model Architectures", "24"),
        ("Table 4: Per-Class Diagnostic Failure Modes and Biomechanical Root Causes", "26"),
        ("Table 5: Real-Time Latency, Memory Footprint, and Runtime Resource Profiling", "30")
    ]

    p98 = doc.paragraphs[98]
    p98.text = f"{tables_data[0][0]}\t{tables_data[0][1]}"
    p99 = doc.paragraphs[99]
    p99.text = f"{tables_data[1][0]}\t{tables_data[1][1]}"
    p100 = doc.paragraphs[100]
    p100.text = f"{tables_data[2][0]}\t{tables_data[2][1]}"

    # -------------------------------------------------------------
    # 5. LIST OF FIGURES (Paragraphs 101-112)
    # -------------------------------------------------------------
    p101 = doc.paragraphs[101]
    p101.text = "LIST OF FIGURES"
    p101.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p101.runs:
        p101.runs[0].font.name = 'Times New Roman'
        p101.runs[0].font.size = Pt(16)
        p101.runs[0].bold = True

    # Strip instruction <<...>> at p102
    doc.paragraphs[102].text = ""
    doc.paragraphs[103].text = ""

    p104 = doc.paragraphs[104]
    p104.text = "Figure\tPage"
    p104.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if p104.runs:
        p104.runs[0].font.name = 'Times New Roman'
        p104.runs[0].font.size = Pt(11)
        p104.runs[0].bold = True

    figures_data = [
        ("Figure 1: Multi-Modal Decision Fusion and Translation System Architecture", "11"),
        ("Figure 2: Data Flow Diagrams (Level 0 Context and Level 1 Process Decomposition)", "14"),
        ("Figure 3: Sequence Diagram of Live Prediction, Temporal Buffering, and Audio Synthesis", "15"),
        ("Figure 4: UML Use-Case Diagram for System Actors and Functional Scenarios", "16"),
        ("Figure 5: Detailed UML Class Diagram of the Software Subsystems", "17"),
        ("Figure 6: Entity-Relationship & Dataset Schema Model", "18"),
        ("Figure 7: Real-Time User Interface with Live Video Overlay and Multilingual Buffer", "19"),
        ("Figure 8: Reconciled Held-Out Benchmark Performance Across Dual-Hand Feature Architectures", "26"),
        ("Figure 9: Forensic Root-Cause Analysis of Class K → Class T Classification Collapse", "28"),
        ("Figure 10: Biomechanical Skeletal Normalization and Cross-Hand Distortion Comparison", "29")
    ]

    # Populate the first min(len(figures_data), 8) into template paragraphs 105..112
    for idx in range(min(len(figures_data), 8)):
        title, page = figures_data[idx]
        target_p = doc.paragraphs[105 + idx]
        target_p.text = f"{title}\t{page}"
        for r in target_p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(11)

    # Insert any remaining figures beyond 8 into document xml
    last_p = doc.paragraphs[112]
    for idx in range(8, len(figures_data)):
        title, page = figures_data[idx]
        new_p_elm = OxmlElement('w:p')
        last_p._p.addnext(new_p_elm)
        new_p = Paragraph(new_p_elm, last_p._parent)
        new_p.text = f"{title}\t{page}"
        for r in new_p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(11)
        last_p = new_p

    offset = max(0, len(figures_data) - 8)

    # -------------------------------------------------------------
    # 6. LIST OF ABBREVIATIONS (Paragraphs 113+offset to 137+offset)
    # -------------------------------------------------------------
    p113 = doc.paragraphs[113 + offset]
    p113.text = "LIST OF ABBREVIATIONS"
    p113.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p113.runs:
        p113.runs[0].font.name = 'Times New Roman'
        p113.runs[0].font.size = Pt(16)
        p113.runs[0].bold = True

    p114 = doc.paragraphs[114 + offset]
    p114.text = "Abbreviation\tExplanation of the Abbreviation"
    if p114.runs:
        p114.runs[0].font.name = 'Times New Roman'
        p114.runs[0].font.size = Pt(11)
        p114.runs[0].bold = True

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
        target_p = doc.paragraphs[115 + offset + idx]
        target_p.text = f"{abbr}\t\t\t\t\t\t{desc}"
        for r in target_p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(10.5)

    # Clear remaining abbreviation paragraphs up to 137 + offset
    for idx in range(115 + offset + len(abbrevs), 138 + offset):
        doc.paragraphs[idx].text = ""

    # -------------------------------------------------------------
    # 7. ABSTRACT (Paragraphs 138+offset to 140+offset, sectPr at 141+offset)
    # -------------------------------------------------------------
    p138 = doc.paragraphs[138 + offset]
    p138.text = "ABSTRACT"
    p138.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p138.runs:
        p138.runs[0].font.name = 'Times New Roman'
        p138.runs[0].font.size = Pt(16)
        p138.runs[0].bold = True
        p138.runs[0].font.size = Pt(16)
        p138.runs[0].bold = True

    abstract_text = (
        "Indian Sign Language (ISL) serves as the primary visual-gestural communication medium for over 18 million "
        "deaf and hard-of-hearing individuals across the Indian subcontinent. However, the severe shortage of certified "
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

    p139 = doc.paragraphs[139 + offset]
    p139.text = abstract_text
    p139.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if p139.runs:
        p139.runs[0].font.name = 'Times New Roman'
        p139.runs[0].font.size = Pt(11.5)

    doc.paragraphs[140 + offset].text = ""
    # Paragraph 141 has sectPr - PRESERVE WITHOUT ALTERING P_ELEMENT

    print("Front matter updated successfully.")
