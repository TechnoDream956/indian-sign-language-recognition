#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 5P: Dual-Hand ISL Alphabet Dataset Audit Utility
======================================================
Comprehensive quality-assurance and audit tool for validating the
data/phase5p_dual_hand/ dataset.

Audit checks:
1. Total valid and corrupted sample count.
2. Label representation and sample counts per class.
3. Signer diversity and session distribution.
4. Feature integrity: shape == (126,), finite coordinates (no NaN/Inf).
5. Duplicate sample detection (intra-class distance thresholding).
6. Metadata consistency: file existence vs. metadata.csv sync.
7. Class imbalance metrics.
8. Train / Validation / Test split readiness assessment.

Non-destructive: Does not alter data, train models, or modify production files.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Ensure workspace root is importable
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tools.collection.collect_dual_hand_alphabet import ISL_ALPHABET_HAND_CONFIG

DEFAULT_DATA_DIR = WORKSPACE_ROOT / "data" / "phase5p_dual_hand"
DEFAULT_REPORT_PATH = WORKSPACE_ROOT / "docs" / "PHASE_5P_DATASET_AUDIT.md"


def run_dataset_audit(
    data_dir: Path = DEFAULT_DATA_DIR,
    report_output: Optional[Path] = DEFAULT_REPORT_PATH,
    duplicate_threshold: float = 1e-3,
    min_samples_per_class: int = 15,
) -> Dict[str, Any]:
    """
    Perform a complete audit of the dual-hand dataset.
    """
    data_dir = Path(data_dir)
    samples_dir = data_dir / "samples"
    metadata_csv = data_dir / "metadata.csv"

    audit_results: Dict[str, Any] = {
        "dataset_exists": False,
        "metadata_exists": False,
        "total_metadata_rows": 0,
        "total_sample_files": 0,
        "valid_samples": 0,
        "corrupted_samples": 0,
        "missing_files": 0,
        "orphaned_files": 0,
        "class_counts": {},
        "signer_counts": {},
        "session_counts": {},
        "hand_count_distribution": {0: 0, 1: 0, 2: 0},
        "hand_config_conforming": 0,
        "hand_config_non_conforming": [],
        "duplicate_pairs_count": 0,
        "class_imbalance_ratio": 0.0,
        "readiness_status": "NOT READY",
        "readiness_notes": [],
    }

    if not data_dir.exists():
        audit_results["readiness_notes"].append(f"Directory {data_dir} does not exist.")
        return audit_results

    audit_results["dataset_exists"] = True

    # 1. Read metadata.csv
    rows: List[Dict[str, str]] = []
    if metadata_csv.exists():
        audit_results["metadata_exists"] = True
        try:
            with open(metadata_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                audit_results["total_metadata_rows"] = len(rows)
        except Exception as e:
            audit_results["readiness_notes"].append(f"Error reading metadata.csv: {e}")

    # 2. Scan physical samples directory
    physical_files = {}
    if samples_dir.exists():
        for p in samples_dir.glob("*.npy"):
            physical_files[p.name] = p
        audit_results["total_sample_files"] = len(physical_files)

    # 3. Cross-reference metadata rows with files
    referenced_files = set()
    label_samples: Dict[str, List[np.ndarray]] = {}

    for row in rows:
        sample_id = row.get("sample_id", "")
        label = row.get("label", "UNKNOWN")
        signer = row.get("signer_id", "unknown")
        session = row.get("session_id", "unknown")
        rel_path = row.get("filepath", "")
        hand_cnt_str = row.get("hand_count", "0")

        try:
            hand_cnt = int(hand_cnt_str)
        except ValueError:
            hand_cnt = 0
        audit_results["hand_count_distribution"][hand_cnt] = (
            audit_results["hand_count_distribution"].get(hand_cnt, 0) + 1
        )

        # Check compliance against centralized ISL alphabet hand configuration
        exp_h = ISL_ALPHABET_HAND_CONFIG.get(label, 2)
        if hand_cnt == exp_h:
            audit_results["hand_config_conforming"] += 1
        else:
            audit_results["hand_config_non_conforming"].append(
                f"{sample_id} ({label}): expected {exp_h} hands, got {hand_cnt}"
            )

        audit_results["class_counts"][label] = audit_results["class_counts"].get(label, 0) + 1
        audit_results["signer_counts"][signer] = audit_results["signer_counts"].get(signer, 0) + 1
        audit_results["session_counts"][session] = audit_results["session_counts"].get(session, 0) + 1

        # Check physical file
        filename = Path(rel_path).name
        referenced_files.add(filename)
        actual_path = samples_dir / filename

        if not actual_path.exists():
            audit_results["missing_files"] += 1
            continue

        # Validate feature array
        try:
            arr = np.load(actual_path)
            if not isinstance(arr, np.ndarray) or arr.shape != (126,):
                audit_results["corrupted_samples"] += 1
                audit_results["readiness_notes"].append(
                    f"{sample_id}: invalid shape {getattr(arr, 'shape', None)}, expected (126,)"
                )
            elif not np.all(np.isfinite(arr)):
                audit_results["corrupted_samples"] += 1
                audit_results["readiness_notes"].append(f"{sample_id}: contains NaN or Inf coordinates")
            else:
                audit_results["valid_samples"] += 1
                if label not in label_samples:
                    label_samples[label] = []
                label_samples[label].append(arr)
        except Exception as e:
            audit_results["corrupted_samples"] += 1
            audit_results["readiness_notes"].append(f"{sample_id}: failed to load: {e}")

    # Check orphaned files
    orphans = set(physical_files.keys()) - referenced_files
    audit_results["orphaned_files"] = len(orphans)

    # 4. Intra-class duplicate vector detection
    total_duplicates = 0
    for lbl, sample_list in label_samples.items():
        n = len(sample_list)
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(sample_list[i] - sample_list[j])
                if dist < duplicate_threshold:
                    total_duplicates += 1
    audit_results["duplicate_pairs_count"] = total_duplicates

    # 5. Class imbalance metrics
    counts = list(audit_results["class_counts"].values())
    if counts:
        min_c = min(counts)
        max_c = max(counts)
        audit_results["class_imbalance_ratio"] = round(float(max_c / max(1, min_c)), 2)
    else:
        audit_results["class_imbalance_ratio"] = 0.0

    # 6. Readiness evaluation
    valid_count = audit_results["valid_samples"]
    classes_present = len(audit_results["class_counts"])

    if valid_count == 0:
        audit_results["readiness_status"] = "EMPTY"
        audit_results["readiness_notes"].append("Dataset contains 0 valid samples. Run data collection first.")
    elif classes_present < 26:
        audit_results["readiness_status"] = "PARTIAL_VOCABULARY"
        audit_results["readiness_notes"].append(
            f"Only {classes_present}/26 alphabet classes are represented in the dataset."
        )
    elif any(c < min_samples_per_class for c in counts):
        audit_results["readiness_status"] = "INSUFFICIENT_SAMPLES"
        underrepresented = [k for k, v in audit_results["class_counts"].items() if v < min_samples_per_class]
        audit_results["readiness_notes"].append(
            f"Classes with fewer than {min_samples_per_class} samples: {underrepresented}"
        )
    elif audit_results["corrupted_samples"] > 0:
        audit_results["readiness_status"] = "CORRUPTED"
        audit_results["readiness_notes"].append(
            f"{audit_results['corrupted_samples']} samples have corrupted shapes or NaN/Inf values."
        )
    else:
        audit_results["readiness_status"] = "READY_FOR_SPLITTING"
        audit_results["readiness_notes"].append(
            "Dataset meets quality, dimensional, and minimum sample thresholds for training preparation."
        )

    # Generate Markdown Report
    if report_output:
        _generate_markdown_report(audit_results, Path(report_output))

    return audit_results


def _generate_markdown_report(results: Dict[str, Any], output_path: Path) -> None:
    """Write human-readable audit report in Markdown format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Phase 5P — Dual-Hand Alphabet Dataset Audit Report",
        "",
        f"**Audit Status:** `{results['readiness_status']}`  ",
        f"**Total Valid Samples:** `{results['valid_samples']}`  ",
        f"**Total Classes Represented:** `{len(results['class_counts'])}`  ",
        f"**Corrupted Samples:** `{results['corrupted_samples']}`  ",
        f"**Duplicate Pairs Detected:** `{results['duplicate_pairs_count']}`  ",
        "",
        "---",
        "",
        "## 1. Class Distribution",
        "",
        "| Class | Sample Count | Status |",
        "|:---:|:---:|:---:|",
    ]

    if not results["class_counts"]:
        lines.append("| — | 0 | No samples recorded |")
    else:
        for lbl in sorted(results["class_counts"].keys()):
            cnt = results["class_counts"][lbl]
            st = "✅ OK" if cnt >= 15 else "⚠️ LOW"
            lines.append(f"| `{lbl}` | {cnt} | {st} |")

    lines.extend([
        "",
        "## 2. Hand Count & ISL Reference Compliance",
        "",
        "- **ISL Hand Configuration Reference:** Single-handed: `C, L, O, U, V` (5 letters); Two-handed: all other 21 letters.",
        f"- **Reference-Conforming Samples:** `{results.get('hand_config_conforming', 0)} / {results['valid_samples']}`",
        f"- **0 Hands:** {results['hand_count_distribution'].get(0, 0)}",
        f"- **1 Hand:** {results['hand_count_distribution'].get(1, 0)}",
        f"- **2 Hands:** {results['hand_count_distribution'].get(2, 0)}",
    ])
    if results.get("hand_config_non_conforming"):
        lines.append("\n### Non-Conforming Hand Count Samples:")
        for nc in results["hand_config_non_conforming"]:
            lines.append(f"- ⚠️ `{nc}`")

    lines.extend([
        "",
        "## 3. Signer & Session Breakdown",
        "",
    ])

    if results["signer_counts"]:
        lines.append("### Signers:")
        for s, c in results["signer_counts"].items():
            lines.append(f"- `{s}`: {c} samples")
    else:
        lines.append("- No signer metadata recorded.")

    lines.extend([
        "",
        "## 4. Integrity & Readiness Assessment",
        "",
        f"- **Missing Files:** {results['missing_files']}",
        f"- **Orphaned Files:** {results['orphaned_files']}",
        f"- **Class Imbalance Ratio:** {results['class_imbalance_ratio']}:1",
        "",
        "### Audit Observations & Notes:",
    ])

    for note in results["readiness_notes"]:
        lines.append(f"- {note}")

    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Phase 5P Dual-Hand Dataset")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to dataset root")
    parser.add_argument("--run-id", type=str, default=None, help="Audit an isolated collection run under data/phase5p_dual_hand/runs/<run_id>")
    parser.add_argument("--output-report", type=str, default=None, help="Path to write markdown audit report")
    parser.add_argument("--dup-thresh", type=float, default=1e-3, help="Euclidean distance threshold for duplicates")
    parser.add_argument("--min-samples", type=int, default=15, help="Minimum samples per class threshold")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.run_id:
        if data_dir == DEFAULT_DATA_DIR:
            data_dir = DEFAULT_DATA_DIR / "runs" / args.run_id

    if args.output_report:
        report_output = Path(args.output_report)
    elif args.run_id:
        report_output = WORKSPACE_ROOT / "docs" / f"PHASE_5P_DATASET_AUDIT_{args.run_id}.md"
    else:
        report_output = DEFAULT_REPORT_PATH

    results = run_dataset_audit(
        data_dir=data_dir,
        report_output=report_output,
        duplicate_threshold=args.dup_thresh,
        min_samples_per_class=args.min_samples,
    )

    print("\n" + "=" * 60)
    print("  DUAL-HAND DATASET AUDIT SUMMARY")
    print("=" * 60)
    print(f"Readiness Status:        {results['readiness_status']}")
    print(f"Total Valid Samples:     {results['valid_samples']}")
    print(f"Corrupted Samples:       {results['corrupted_samples']}")
    print(f"Duplicate Pairs:         {results['duplicate_pairs_count']}")
    print(f"Classes Represented:     {len(results['class_counts'])}")
    print(f"Signers:                 {len(results['signer_counts'])}")
    print(f"Report Written To:       {report_output}")
    print("=" * 60 + "\n")
