"""Chapter 1 builder: Introduction and Problem Statement.
"""

from docx import Document
from .xml_helpers import add_heading_1, add_heading_2, add_body_p, add_bullet_p


def build_chapter1(doc: Document):
    """Generate Chapter 1: Introduction and Problem Statement."""
    
    # 1. INTRODUCTION (Style GS1 provides automatic chapter numbering)
    add_heading_1(doc, "INTRODUCTION")

    add_body_p(
        doc,
        "Human communication relies heavily on spoken language and acoustic feedback, creating a significant "
        "barrier between hearing societies and deaf or hard-of-hearing individuals. According to reported "
        "demographic and advocacy estimates, millions of people in India experience significant hearing impairment. "
        "Indian Sign Language (ISL) serves as their primary linguistic and gestural mother tongue. Despite its foundational "
        "importance, the acute scarcity of qualified sign language interpreters across the country creates severe exclusion "
        "in healthcare clinics, judicial proceedings, academic institutions, public transportation, and daily commercial transactions."
    )

    add_body_p(
        doc,
        "In recent years, advances in computer vision, deep learning, and lightweight kinematic tracking have opened "
        "practical opportunities for automated assistive sign language translation. However, the vast majority of prior "
        "academic literature and commercial prototypes have focused almost exclusively on Western sign languages, particularly "
        "American Sign Language (ASL) and British Sign Language (BSL). Extending these methodologies directly to Indian Sign "
        "Language fails due to fundamental linguistic and kinematic disparities. While ASL fingerspelling is predominantly "
        "unimanual (single-handed), ISL fingerspelling is largely two-handed (bimanual), characterized by complex "
        "bilateral hand-to-hand contacts, interlaced fingers, cross-palm occlusions, and dynamic spatial coordinates relative "
        "to the signer's torso. A computer vision pipeline designed for ASL cannot accommodate the geometric complexity, "
        "bilateral symmetry, and hand-swapping ambiguities inherent to ISL."
    )

    add_body_p(
        doc,
        "Historically, sign language recognition systems have fallen into two distinct architectural paradigms: sensor-based "
        "wearables and vision-based perception. Wearable approaches utilize sensor-instrumented data gloves embedded with flex "
        "sensors, inertial measurement units (IMUs), and tactile contacts. Although data gloves can bypass visual occlusion, "
        "they are often costly, mechanically fragile, cumbersome to calibrate, "
        "and socially stigmatizing for daily use. Conversely, vision-based approaches offer a natural, non-invasive alternative "
        "using standard consumer webcams. Yet, traditional vision pipelines have suffered from computational bottlenecks, "
        "requiring high-end discrete GPUs or continuous cloud video streaming. Streaming video "
        "to cloud servers introduces noticeable transmission latency, consumes substantial bandwidth unavailable "
        "in tier-2 and rural regions of India, and raises privacy concerns regarding the transmission of private domestic video."
    )

    add_body_p(
        doc,
        "This project bridges this assistive and computational gap by engineering an edge-deployable, "
        "real-time Indian Sign Language alphabet recognition and translation platform. The system operates locally on "
        "consumer hardware (such as Apple Silicon or Intel-based laptops) at frame rates of approximately 22.6 FPS on active "
        "dual-hand frames without requiring paid cloud APIs. By fusing 3D kinematic landmarks extracted via Google MediaPipe Hands "
        "(21 3D landmarks per hand) with an appearance-based Convolutional Neural Network (CNN, 128x128x3), the platform provides multi-modal perception. "
        "The system advances beyond raw character output by integrating a temporal WordBuffer state machine that "
        "assembles spelled letters into words, a multilingual translation engine supporting 10 regional Indian languages "
        "with offline dictionary mapping and fallback, and a multi-tier speech fallback engine (macOS say, pyttsx3, gTTS, "
        "and browser speech synthesis). Optional on-device neural translation (IndicTrans2) and speech synthesis (MMS-TTS) "
        "modules are also implemented in the codebase for offline research evaluation. "
        "This end-to-end prototype provides an accessible, private, and bidirectional communication tool for the Indian deaf community."
    )

    # 1.1 Problem Statement (Style GS2 provides automatic section numbering)
    add_heading_2(doc, "Problem Statement")

    add_body_p(
        doc,
        "The technical challenge addressed in this research is the formulation, edge deployment, and empirical "
        "auditing of a real-time, non-invasive assistive pipeline capable of transcribing continuous, camera-based "
        "Indian Sign Language alphabet gestures into stabilized English text, translating the assembled phrases into regional "
        "Indian languages, and synthesizing speech output. Specifically, this "
        "investigation resolves four interrelated computational, kinematic, and linguistic bottlenecks:"
    )

    add_bullet_p(
        doc,
        "Standard single-hand deep learning classifiers fail when applied to ISL due to the presence of both single-handed "
        "and dual-handed letter signs (e.g., A, B, C, D, K, P, T, W). When two hands enter the visual field, landmarks frequently "
        "swap identities (left versus right hand confusion), and Euclidean feature vectors distort due to inter-signer "
        "distance and camera perspective differences.",
        bold_prefix="Bimanual Occlusion & Landmark Invariance: "
    )

    add_bullet_p(
        doc,
        "In isolated frame-by-frame inference, natural transitional hand movements between signs produce erratic predictions, "
        "leading to visual flickering and spurious character insertions. Without temporal smoothing, debouncing, "
        "and gesture stability confirmation, constructing intelligible words from fingerspelling is unreliable.",
        bold_prefix="Temporal Gesture Segmentation & Debouncing: "
    )

    add_bullet_p(
        doc,
        "India is linguistically diverse, comprising 22 officially recognized scheduled languages and hundreds of regional "
        "dialects. Existing translation tools either depend on expensive, latency-prone cloud APIs (such as Google Translate "
        "or OpenAI) or rely on static hardcoded dictionaries that lack grammatical inflection, gender agreement, and contextual "
        "fluency. Delivering on-device translation and speech synthesis across 10 Indian languages "
        "within edge computational constraints represents a significant systems engineering hurdle.",
        bold_prefix="Edge Multilingual Translation & Speech Barrier: "
    )

    add_bullet_p(
        doc,
        "A recognized issue in contemporary sign language machine learning literature is accuracy inflation, where models report "
        "high test accuracy on randomly shuffled data splits that contain the same signer across train and test sets (data leakage). "
        "When deployed in front of a live webcam with an unseen user, performance degrades due to domain shift, "
        "morphological variations, and lighting changes. This project formulates a "
        "controlled held-out evaluation methodology that examines these generalization challenges with methodological discipline.",
        bold_prefix="The Generalization Gap & Evaluation Integrity: "
    )

    print("Chapter 1 generated successfully.")
