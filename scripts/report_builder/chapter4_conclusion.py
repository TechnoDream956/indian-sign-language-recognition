"""Chapter 4 builder: Conclusion, Goals/Vision, Delivered Solution, and Remaining Work.
"""

from docx import Document
from .xml_helpers import add_heading_1, add_heading_2, add_body_p, add_bullet_p


def build_chapter4(doc: Document):
    """Generate Chapter 4: Conclusion."""
    
    # 4. CONCLUSION
    add_heading_1(doc, "4. CONCLUSION")

    # 4.1 Goals/ Vision
    add_heading_2(doc, "4.1. Goals/ Vision")

    add_body_p(
        doc,
        "The overarching vision of this project was to formulate an accessible, low-cost, non-invasive assistive technology "
        "capable of breaking down the profound communication barrier separating India's 18 million deaf citizens from the hearing "
        "majority. By recognizing that certified human interpreters are unattainable for daily interactions (<300 nationwide), "
        "our goal was to eliminate reliance on expensive, cumbersome wearable sensors (data gloves) and latency-prone, "
        "privacy-compromising cloud APIs. We envisioned a fully local, edge-deployable pipeline running on commodity consumer "
        "hardware that translates real-time Indian Sign Language gestures into stabilized text, converts phrases into regional "
        "Indian languages, and speaks aloud through neural audio synthesis."
    )

    add_body_p(
        doc,
        "Throughout this research, our technical focus underwent critical maturation. While initial milestones explored "
        "whole-sign dynamic sequence classification (Word Mode using MoP-GRU), extensive user feedback and data audits demonstrated "
        "that vocabulary-limited word models cannot support open-ended communication. Consequently, the project pivoted toward "
        "mastering the 26-letter ISL fingerspelling alphabet—a combinatorial medium enabling signers to spell any word, proper noun, "
        "or technical term. Reaching this vision demanded resolving complex bimanual occlusions, enforcing bilateral spatial invariance, "
        "and coupling machine learning predictions with an intelligent temporal state machine."
    )

    # 4.2 Delivered Solution
    add_heading_2(doc, "4.2. Delivered Solution")

    add_body_p(
        doc,
        "The completed engineering effort delivers an end-to-end, fully functional, and independently tested software platform "
        "for real-time Indian Sign Language recognition, multilingual translation, and neural speech synthesis. The delivered "
        "system incorporates the following verified components:"
    )

    add_bullet_p(
        doc,
        "A multi-modal perception architecture that extracts 543 spatial landmarks via MediaPipe Holistic and cropped 64x64x3 "
        "RGB appearance regions at 22–27 FPS. Includes a dynamic hand-presence router dispatching to single-hand 63D MLPs or "
        "dual-hand 144D bilateral invariant MLPs.",
        bold_prefix="1. Multi-Modal Dual-Branch Vision Pipeline: "
    )

    add_bullet_p(
        doc,
        "A confidence-aware fusion engine that synthesizes geometric and appearance probability vectors (W_geom=0.60, W_cnn=0.40) "
        "with a +0.15 agreement bonus, coupled with a WordBuffer state machine featuring 1.2 s debouncing, 5-frame stability confirmation, "
        "and hands-free space, backspace, and clear controls.",
        bold_prefix="2. Confidence Fusion & WordBuffer Debouncing: "
    )

    add_bullet_p(
        doc,
        "A completely offline, on-device translation subsystem powered by the 200M distilled int8-quantized IndicTrans2 ONNX model, "
        "translating English phrases into eight scheduled Indian languages (Hindi, Bengali, Marathi, Tamil, Telugu, Gujarati, "
        "Kannada, Malayalam) in 28–66 ms per sentence on CPU.",
        bold_prefix="3. On-Device Indic Neural Machine Translation: "
    )

    add_bullet_p(
        doc,
        "A local neural speech engine powered by Meta MMS-TTS (VITS architecture), generating 16 kHz broadcast-quality WAV audio "
        "with a Real-Time Factor (RTF) of 0.148 (6.7x faster than real-time) within a memory footprint under 450 MB.",
        bold_prefix="4. Multilingual Neural Speech Synthesis: "
    )

    add_bullet_p(
        doc,
        "A decoupled web architecture streaming video via MJPEG and inference states via Server-Sent Events (SSE) with sub-50 ms "
        "latency, wrapped in a self-contained macOS desktop application package (.app) that launches without developer tools.",
        bold_prefix="5. Asynchronous Web UI & Standalone Packaging: "
    )

    add_bullet_p(
        doc,
        "An exhaustive suite of 245 automated unit, regression, and integration tests covering inference logic, debounce timing, "
        "translation accuracy, and API endpoints, maintaining a 100% pass rate under continuous integration.",
        bold_prefix="6. Automated Quality Assurance: "
    )

    # 4.3 Remaining Work
    add_heading_2(doc, "4.3. Remaining Work")

    add_body_p(
        doc,
        "In strict adherence to academic rigor, our evaluation documented that live camera performance remains vulnerable "
        "to inter-signer morphological variation and camera perspective shifts. To transition this prototype into a clinical "
        "or commercial product, the following concrete technical roadmap is established:"
    )

    add_bullet_p(
        doc,
        "As established in our root-cause forensics (Prompt 25), simple sample duplication fails to expand the geometric manifold. "
        "Future work must implement 3D skeletal synthetic augmentation that perturbs finger extension profiles (mimicking foreshortening "
        "and curling) and translates wrist anchors level to the camera plane, preventing the Class K -> Class T collapse.",
        bold_prefix="1. Biomechanical Multi-Signer Data Augmentation: "
    )

    add_bullet_p(
        doc,
        "To resolve mutual confusions between M vs N, and R vs U vs V, future iterations will extract explicit Euclidean "
        "distances between adjacent fingertips and compute closed-loop polygon areas between the thumb and index finger "
        "(resolving D vs P vs Q).",
        bold_prefix="2. Explicit 3D Inter-Finger Contact & Loop Metrics: "
    )

    add_bullet_p(
        doc,
        "Implement dynamic entry suppression gates (threshold >= 0.75) specifically for open palm (B) and resting horizontal "
        "two-finger (H) postures, ensuring transitional movements do not trigger false positive insertions.",
        bold_prefix="3. Transitional Anchor Suppression Gating: "
    )

    add_bullet_p(
        doc,
        "Extend the architecture from isolated fingerspelling character recognition to continuous sentence-level sign language "
        "translation using Connectionist Temporal Classification (CTC) and lightweight Temporal Transformers, enabling signers "
        "to communicate using fluid grammatical sentences.",
        bold_prefix="4. Continuous Sentence-Level Sign Translation: "
    )

    add_bullet_p(
        doc,
        "Quantize the entire pipeline for mobile hardware by compiling the MediaPipe perception graph and neural models into "
        "Apple CoreML (iOS) and TensorFlow Lite (Android), enabling direct smartphones deployment for deaf individuals in the field.",
        bold_prefix="5. Mobile Edge Deployment (iOS & Android): "
    )

    add_bullet_p(
        doc,
        "Empirical evaluation revealed that when clean landmark geometry and visual CNN branches disagree (57 of 243 samples), neither branch "
        "is correct 66.67% of the time (38/57; 84.4% or 38/45 under the K/T override). We formulate the hypothesis that gating or displaying an explicit 'Uncertain / Re-sign' status "
        "when branches diverge could suppress spurious letter insertions. This represents a research hypothesis to be tested on live "
        "multi-signer recordings, not an established result.",
        bold_prefix="6. Disagreement-Gated Uncertainty Output (Hypothesis): "
    )

    print("Chapter 4 generated successfully.")
