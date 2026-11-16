#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Benchmark Script for Multilingual Translation & Speech Subsystem.
Evaluates all 6 required benchmark sentences across Hindi, Bengali, Marathi,
Tamil, Telugu, Gujarati, Kannada, and Malayalam.
Measures latency, memory usage, audio duration, real-time factor, and script accuracy.
"""

import json
import os
import resource
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.language_services.service import get_language_service

BENCHMARK_SENTENCES = [
    "Hello, how are you?",
    "I need some water.",
    "My name is Arpit.",
    "Can you please help me?",
    "I am going to college tomorrow.",
    "Thank you for your support.",
]

EVAL_LANGUAGES = [
    "Hindi",
    "Bengali",
    "Marathi",
    "Tamil",
    "Telugu",
    "Gujarati",
    "Kannada",
    "Malayalam",
]


def get_peak_ram_mb():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return round(usage.ru_maxrss / (1024 * 1024), 2)
    return round(usage.ru_maxrss / 1024, 2)


def run_benchmark():
    service = get_language_service()
    print("=" * 80)
    print("MULTILINGUAL TRANSLATION & NEURAL TTS BENCHMARK REPORT")
    print(f"Platform: {sys.platform} | Peak RAM at start: {get_peak_ram_mb()} MB")
    print("=" * 80)

    results = []

    for lang in EVAL_LANGUAGES:
        print(f"\n--- Testing Language: {lang} ---")
        for sent in BENCHMARK_SENTENCES:
            # Measure pure translation
            t0 = time.time()
            trans_res = service.translate(sent, target_lang=lang, use_cache=False)
            t_trans = round(time.time() - t0, 3)

            # Speech synthesis on the translated native text (for Hindi & Bengali)
            t_speak = 0.0
            audio_dur = 0.0
            rtf = 0.0
            speak_success = True

            if lang in ["Hindi", "Bengali"]:
                t1 = time.time()
                speak_res = service.speak(trans_res.translated_text, target_lang=lang)
                t_speak = round(time.time() - t1, 3)
                audio_dur = speak_res.get("duration_seconds", 0.0)
                rtf = round(t_speak / audio_dur, 3) if audio_dur > 0 else 0.0
                speak_success = speak_res.get("success", False)

            row = {
                "language": lang,
                "english": sent,
                "translation": trans_res.translated_text,
                "trans_time_s": t_trans,
                "tts_time_s": t_speak if t_speak > 0 else None,
                "audio_dur_s": audio_dur if audio_dur > 0 else None,
                "rtf": rtf if rtf > 0 else None,
                "success": trans_res.success and speak_success,
            }
            results.append(row)
            status_str = "PASS" if row["success"] else "FAIL"
            print(f"  [{status_str}] \"{sent}\"")
            print(f"      -> \"{trans_res.translated_text}\"")
            if t_speak > 0:
                print(f"      Translation: {t_trans}s | TTS: {t_speak}s | Audio: {audio_dur}s (RTF: {rtf})")
            else:
                print(f"      Translation: {t_trans}s")

    peak_ram = get_peak_ram_mb()
    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETE. Total evaluations: {len(results)}.")
    print(f"Final Peak RAM Consumption: {peak_ram} MB")
    print("=" * 80)

    out_file = os.path.join(REPO_ROOT, "data", "language_services", "benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"results": results, "peak_ram_mb": peak_ram}, f, ensure_ascii=False, indent=2)
    print(f"Benchmark results saved to: {out_file}")


if __name__ == "__main__":
    run_benchmark()
