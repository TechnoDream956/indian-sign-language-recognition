"""Chapter 1 builder: Introduction and Problem Statement.
"""

from docx import Document
from .xml_helpers import add_heading_1, add_heading_2, add_body_p, add_bullet_p


def build_chapter1(doc: Document):
    """Generate Chapter 1: Introduction and Problem Statement."""
    
    # 1. INTRODUCTION
    add_heading_1(doc, "1. INTRODUCTION")

    add_body_p(
        doc,
        "Human communication relies heavily on spoken language and acoustic feedback, creating a profound, "
        "often invisible divide between hearing societies and deaf or hard-of-hearing individuals. According to the "
        "World Health Organization (WHO) and the National Association of the Deaf (NAD) in India, approximately 18 million "
        "people in India experience significant hearing impairment. Indian Sign Language (ISL) serves as their primary "
        "linguistic and gestural mother tongue. Despite its foundational importance, there are fewer than 300 certified "
        "sign language interpreters across the entire country—representing an astonishing interpreter-to-deaf ratio of "
        "roughly 1:60,000. Consequently, deaf individuals face severe systemic exclusion in healthcare clinics, judicial "
        "proceedings, academic institutions, public transportation, and daily commercial transactions."
    )

    add_body_p(
        doc,
        "In recent years, breakthroughs in computer vision, deep learning, and lightweight kinematic tracking have opened "
        "unprecedented opportunities for automated assistive sign language translation. However, the vast majority of prior "
        "academic literature and commercial prototypes have focused almost exclusively on Western sign languages, particularly "
        "American Sign Language (ASL) and British Sign Language (BSL). Extending these methodologies directly to Indian Sign "
        "Language fails due to fundamental linguistic and biomechanical disparities. While ASL fingerspelling is predominantly "
        "unimanual (single-handed), ISL fingerspelling is overwhelmingly bimanual (two-handed), characterized by complex "
        "bilateral hand-to-hand contacts, interlaced fingers, cross-palm occlusions, and dynamic spatial coordinates relative "
        "to the signer's torso. A computer vision pipeline designed for ASL cannot accommodate the geometric complexity, "
        "bilateral symmetry, and hand-swapping ambiguities inherent to ISL."
    )

    add_body_p(
        doc,
        "Historically, sign language recognition systems have fallen into two distinct architectural paradigms: sensor-based "
        "wearables and vision-based perception. Wearable approaches utilize sensor-instrumented data gloves embedded with flex "
        "sensors, inertial measurement units (IMUs), and tactile contacts. Although data gloves can bypass visual occlusion, "
        "they are commercially cost-prohibitive (frequently exceeding $500), mechanically fragile, cumbersome to calibrate, "
        "and socially stigmatizing for daily use. Conversely, vision-based approaches offer a natural, non-invasive alternative "
        "using standard consumer webcams. Yet, traditional vision pipelines have suffered from prohibitive computational latency, "
        "requiring high-end discrete GPUs or continuous cloud video streaming. Streaming 30 high-definition frames per second "
        "to cloud servers introduces unmanageable transmission latency (>300 ms), consumes substantial bandwidth unavailable "
        "in tier-2 and rural regions of India, and raises severe privacy concerns regarding the transmission of private domestic video."
    )

    add_body_p(
        doc,
        "This project bridges this critical assistive and computational gap by engineering a complete, edge-deployable, "
        "real-time Indian Sign Language alphabet recognition and translation platform. The system operates entirely on local "
        "consumer hardware (such as Apple Silicon or Intel-based laptops) at frame rates between 22 and 27 FPS without requiring "
        "an active internet connection or paid cloud APIs. By fusing 3D kinematic landmarks extracted via Google MediaPipe Holistic "
        "with an appearance-based Convolutional Neural Network (CNN), the platform achieves robust multi-modal perception. "
        "Crucially, the system advances beyond raw character output by integrating a temporal WordBuffer state machine that "
        "assembles spelled letters into words, an on-device int8-quantized IndicTrans2 neural translation engine supporting "
        "eight scheduled Indian regional languages, and a Meta Massively Multilingual Speech (MMS-TTS) neural speech synthesis engine. "
        "This end-to-end prototype provides an accessible, private, and bidirectional communication bridge for the Indian deaf community."
    )

    # 1.1 Problem Statement
    add_heading_2(doc, "1.1. Problem Statement")

    add_body_p(
        doc,
        "The overarching technical challenge addressed in this research is the formulation, edge deployment, and empirical "
        "forensic auditing of a real-time, non-invasive assistive pipeline capable of transcribing continuous, camera-based "
        "Indian Sign Language alphabet gestures into stabilized English text, translating the assembled phrases into regional "
        "Indian languages, and synthesizing natural neural speech with sub-100 ms feedback latency. Specifically, this "
        "investigation resolves four interrelated computational, biomechanical, and linguistic bottlenecks:"
    )

    add_bullet_p(
        doc,
        "Standard single-hand deep learning classifiers fail when applied to ISL due to the presence of both single-handed "
        "and dual-handed letter signs (e.g., A, B, C, D, K, P, T, W). When two hands enter the visual field, landmarks frequently "
        "swap identities (left versus right hand confusion), and Euclidean feature vectors distort due to inter-signer "
        "distance and camera perspective differences.",
        bold_prefix="1. Bimanual Occlusion & Landmark Invariance: "
    )

    add_bullet_p(
        doc,
        "In isolated frame-by-frame inference, natural transitional hand movements between signs produce erratic predictions, "
        "leading to visual flickering and spurious character insertions. Without intelligent temporal smoothing, debouncing, "
        "and gesture stability confirmation, constructing intelligible words from fingerspelling is impossible.",
        bold_prefix="2. Temporal Gesture Segmentation & Debouncing: "
    )

    add_bullet_p(
        doc,
        "India is linguistically diverse, comprising 22 officially recognized scheduled languages and hundreds of regional "
        "dialects. Existing translation tools either depend on expensive, latency-prone cloud APIs (such as Google Translate "
        "or OpenAI) or rely on static hardcoded dictionaries that lack grammatical inflection, gender agreement, and contextual "
        "fluency. Delivering genuine, quantized on-device Neural Machine Translation (NMT) and neural Text-to-Speech (TTS) "
        "within strict edge memory constraints (<1 GB RAM) represents a significant systems engineering hurdle.",
        bold_prefix="3. Edge Multilingual Translation & Speech Barrier: "
    )

    add_bullet_p(
        doc,
        "A critical flaw in contemporary sign language machine learning literature is 'accuracy inflation,' where models report "
        ">95% test accuracy on randomly shuffled data splits that contain the same signer across train and test sets (data leakage). "
        "When deployed in front of a live webcam with an unseen user, performance catastrophically degrades due to domain shift, "
        "subtle morphological variations, and lighting changes. This project explicitly formulates an objective, read-only forensic "
        "evaluation methodology that confronts these generalization failures with radical scientific honesty.",
        bold_prefix="4. The Generalization Gap & Evaluation Integrity: "
    )

    print("Chapter 1 generated successfully.")
