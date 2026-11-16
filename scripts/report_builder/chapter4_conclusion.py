"""Chapter 4 builder: Conclusion, Goals/Vision, Delivered Solution, and Remaining Work.
"""

from docx import Document
from .xml_helpers import (
    add_heading_1, add_heading_2, add_body_p, add_bullet_p,
    add_table_caption, format_academic_table
)


def build_chapter4(doc: Document):
    """Generate Chapter 4: Conclusion."""
    
    # 4. CONCLUSION (Style GS1 provides automatic chapter numbering)
    add_heading_1(doc, "CONCLUSION")

    # 4.1 Goals/ Vision (Style GS2 provides automatic section numbering)
    add_heading_2(doc, "Goals/ Vision")

    add_body_p(
        doc,
        "The overarching vision of this project was to formulate an accessible, low-cost, non-invasive assistive technology "
        "capable of breaking down communication barriers for deaf individuals. By recognizing that certified human "
        "interpreters are scarce for daily interactions, our goal was to eliminate reliance on expensive, cumbersome "
        "wearable sensors (data gloves) and latency-prone, privacy-compromising cloud APIs. We envisioned a fully local, "
        "edge-deployable pipeline running on commodity consumer hardware that translates real-time Indian Sign Language "
        "gestures into stabilized text, converts phrases into regional Indian languages, and vocalizes through multi-tier speech synthesis."
    )

    add_body_p(
        doc,
        "Throughout this research, our technical focus underwent necessary maturation. While initial milestones explored "
        "whole-sign dynamic sequence classification (Word Mode using MoP-GRU), data audits demonstrated that vocabulary-limited "
        "word models cannot support open-ended communication. Consequently, the project pivoted toward mastering the 26-letter "
        "ISL fingerspelling alphabet—a combinatorial medium enabling signers to spell any word, proper noun, or technical term. "
        "Reaching this vision demanded addressing bimanual occlusions, enforcing bilateral spatial invariance, and coupling "
        "machine learning predictions with an intelligent temporal state machine."
    )

    add_body_p(
        doc,
        "Table 6 summarizes the engineering status of each initial project goal (G-01 through G-06), documenting whether each "
        "objective was fully met or partly met based on empirical measurements from the codebase."
    )

    # Goal Fulfillment Table
    add_table_caption(doc, 6, "System Goal Fulfillment and Quantitative Evidence Summary")
    table6 = doc.add_table(rows=7, cols=4)
    headers6 = ["Goal ID", "Functional Domain", "Fulfillment Status", "Empirical Evidence & Quantitative Measurement"]
    for idx, h in enumerate(headers6):
        table6.rows[0].cells[idx].text = h

    goals_eval_data = [
        ("G-01", "Kinematic Perception", "Partly Met", "MediaPipe Hands extracts 21 3D landmarks per hand. Measured full pipeline sustained throughput is 23.4 FPS (43.18 ms median / 42.68 ms mean on CPU on active dual-hand frames; 74.4 FPS / 13.14 ms on empty frames). While landmark inference itself is fast (16.90 ms median for AlphabetEngine decision), full frame processing exceeds the 20 ms target due to camera ingestion, OpenCV drawing, and JPEG encoding. Hand-count hysteresis buffer (2 frames) in AlphabetEngine mitigates occlusion drops (85.7% flicker reduction on synthetic 50-frame dropout test sequence)."),
        ("G-02", "Alphabet Recognition", "Met on point estimate", "Dual-hand 144D baseline achieves 78.19% Top-1 accuracy [72.43%, 83.13%] on held-out same-signer test partition (N=243) and 77.16% on deduplicated partition (N=232). Single-hand 63D baseline achieves 75.93% on held-out partition (N=536), but its 95% bootstrap CI is [72.20%, 79.66%], which includes values below 75%; near-duplicate and signer-overlap audits were not performed on the 536-sample single-hand partition due to unpreserved training landmarks."),
        ("G-03", "Decision Fusion & Gating", "Partly Met", "Multi-modal decision fusion is implemented in src/fusion_engine.py with code-aligned weights (0.65 geometry, 0.35 CNN, +0.10 agreement bonus, 0.50 threshold). However, clean fused accuracy (78.19%) equals the geometry branch alone and is below CNN alone (82.72%) on the held-out test set, demonstrating no empirical accuracy gain offline."),
        ("G-04", "Temporal Smoothing", "Met", "WordBuffer state machine in src/alphabet_engine.py implements hands-free word assembly, character commits, backspaces, and space insertion, verified in tests/test_alphabet_engine.py with 1.2 s debounce cooldown and 5-frame stability confirmation thresholds."),
        ("G-05", "Multilingual Translation", "Met", "Dictionary-based translations for a limited vocabulary (not reviewed by native speakers) in src/translator.py translates text into 10 regional Indian languages in < 0.01 ms (< 5 ms target) with zero cloud dependency for listed vocabulary; optional IndicTrans2 module implemented in src/language_services/."),
        ("G-06", "Speech Output", "Partly Met", "Multi-tier speech engine implemented in src/tts_engine.py (macOS say -> pyttsx3 -> gTTS -> browser Web Speech API) and unit-tested across languages; optional Meta MMS-TTS module implemented in src/language_services/. However: (1) time-to-first-audio latency to physical speakers was not measured; (2) audible pronunciation correctness was not evaluated by human listening checks; and (3) Gujarati, Punjabi, and Malayalam lack native offline OS voices on standard systems, falling back to online gTTS or browser speech synthesis.")
    ]

    for r_idx, row in enumerate(goals_eval_data, start=1):
        for c_idx, val in enumerate(row):
            table6.rows[r_idx].cells[c_idx].text = val

    format_academic_table(table6, col_widths=[0.8, 1.4, 1.2, 3.6])

    # 4.2 Delivered Solution (Style GS2 provides automatic section numbering)
    add_heading_2(doc, "Delivered Solution")

    add_body_p(
        doc,
        "The completed engineering effort delivers an end-to-end, functional, and locally tested software platform "
        "for real-time Indian Sign Language recognition, multilingual translation, and speech synthesis. The delivered "
        "system incorporates the following verified components:"
    )

    add_bullet_p(
        doc,
        "A multi-modal perception architecture that extracts 3D skeletal landmarks via MediaPipe Hands (21 per hand) and cropped 128x128x3 "
        "RGB appearance regions at approximately 23.4 FPS on active dual-hand frames (74.4 FPS on empty frames). Includes a dynamic hand-presence router dispatching to single-hand 63D MLPs or "
        "dual-hand 144D bilateral invariant MLPs.",
        bold_prefix="Multi-Modal Dual-Branch Vision Pipeline: "
    )

    add_bullet_p(
        doc,
        "A confidence-aware fusion engine that synthesizes geometric and appearance probability vectors (W_geom=0.65, W_cnn=0.35) "
        "with a +0.10 agreement bonus, coupled with a WordBuffer state machine in src/alphabet_engine.py featuring 1.2 s debouncing, 5-frame stability confirmation, "
        "and hands-free space, backspace, and clear controls.",
        bold_prefix="Confidence Fusion & WordBuffer Debouncing: "
    )

    add_bullet_p(
        doc,
        "A multilingual translation engine supporting 10 regional Indian languages via dictionary mapping for a limited vocabulary "
        "(not reviewed by native speakers), alongside an int8-quantized IndicTrans2 neural translation module implemented in "
        "src/language_services/ as an optional offline module.",
        bold_prefix="Multilingual Translation Subsystem: "
    )

    add_bullet_p(
        doc,
        "A multi-tier speech fallback engine (macOS say -> pyttsx3 -> gTTS -> Web Speech API) ensuring reliable vocalization across platforms, "
        "with Meta MMS-TTS implemented in src/language_services as an optional neural speech synthesis module.",
        bold_prefix="Multilingual Speech Synthesis: "
    )

    add_bullet_p(
        doc,
        "A decoupled web architecture streaming video via MJPEG and inference states via Server-Sent Events (SSE) with low latency, "
        "wrapped in a self-contained desktop application package that executes locally without developer tools.",
        bold_prefix="Asynchronous Web UI & Standalone Packaging: "
    )

    add_bullet_p(
        doc,
        "An automated test suite of 245 unit, regression, and integration tests covering inference logic, debounce timing, "
        "translation accuracy, and API endpoints; all 245 tests pass locally.",
        bold_prefix="Automated Quality Assurance: "
    )

    # 4.3 Remaining Work (Style GS2 provides automatic section numbering)
    add_heading_2(doc, "Remaining Work")

    add_body_p(
        doc,
        "In adherence to academic integrity, our evaluation documented that live camera performance remains sensitive "
        "to inter-signer morphological variation and camera perspective shifts. To transition this prototype into a clinical "
        "or commercial product, the following concrete technical roadmap is established:"
    )

    add_bullet_p(
        doc,
        "As established in our experimental root-cause investigations, simple sample duplication fails to expand the geometric manifold. "
        "Future work should implement 3D skeletal synthetic augmentation that perturbs finger extension profiles (mimicking foreshortening "
        "and curling) and translates wrist anchors level to the camera plane, preventing the Class K -> Class T collapse.",
        bold_prefix="Skeletal Postural Augmentation: "
    )

    add_bullet_p(
        doc,
        "To resolve mutual confusions between M vs N, and R vs U vs V, future iterations will extract explicit Euclidean "
        "distances between adjacent fingertips and compute closed-loop polygon areas between the thumb and index finger "
        "(resolving D vs P vs Q).",
        bold_prefix="Explicit 3D Inter-Finger Contact & Loop Metrics: "
    )

    add_bullet_p(
        doc,
        "Implement dynamic entry suppression gates (threshold >= 0.75) specifically for open palm (B) and resting horizontal "
        "two-finger (H) postures, ensuring transitional movements do not trigger false positive insertions.",
        bold_prefix="Transitional Anchor Suppression Gating: "
    )

    add_bullet_p(
        doc,
        "Extend the architecture from isolated fingerspelling character recognition to continuous sentence-level sign language "
        "translation using Connectionist Temporal Classification (CTC) and lightweight Temporal Transformers, enabling signers "
        "to communicate using fluid grammatical sentences.",
        bold_prefix="Continuous Sentence-Level Sign Translation: "
    )

    add_bullet_p(
        doc,
        "Quantize the entire pipeline for mobile hardware by compiling the MediaPipe perception graph and neural models into "
        "Apple CoreML (iOS) and TensorFlow Lite (Android), enabling direct smartphones deployment for deaf individuals in the field.",
        bold_prefix="Mobile Edge Deployment (iOS & Android): "
    )

    add_bullet_p(
        doc,
        "Empirical evaluation revealed that when clean landmark geometry and visual CNN branches disagree (57 of 243 samples), neither branch "
        "is correct 66.67% of the time (38/57; 84.4% or 38/45 under the K/T override). We formulate the hypothesis that gating or displaying an explicit 'Uncertain / Re-sign' status "
        "when branches diverge could suppress spurious letter insertions. This represents a research hypothesis to be tested on live "
        "multi-signer recordings, not an established result.",
        bold_prefix="Disagreement-Gated Uncertainty Output (Hypothesis): "
    )

    add_bullet_p(
        doc,
        "Formal live recognition accuracy across novel signers has not been measured yet. We will record multi-signer live "
        "benchmark data using tools/collection/record_live_benchmark_samples.py and report empirical live accuracy once multi-signer collection is complete.",
        bold_prefix="Live Multi-Signer Benchmark Validation (Pending): "
    )

    print("Chapter 4 generated successfully.")
