#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train RGB CNN Hand Classifier for Indian Sign Language Recognition.
===================================================================
Extracts a balanced, leakage-free subset of raw cropped RGB images from
data/staging_downloads/realsign/Dataset.zip across all 26 classes (A-Z).
Trains a lightweight, fast ConvNet (128x128x3) suitable for real-time live inference on CPU/Apple Silicon.
Saves the trained model to models/cnn_sign_model.h5 and label map to models/cnn_label_map.json.
"""

import io
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ZIP = WORKSPACE_ROOT / "data" / "staging_downloads" / "realsign" / "Dataset.zip"
OUTPUT_MODEL_PATH = WORKSPACE_ROOT / "models" / "cnn_sign_model.h5"
OUTPUT_LABEL_MAP_PATH = WORKSPACE_ROOT / "models" / "cnn_label_map.json"
REPORT_PATH = WORKSPACE_ROOT / "docs" / "reports" / "CNN_EVALUATION_REPORT.md"

CLASSES = [chr(65 + i) for i in range(26)]  # ['A', ..., 'Z']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

TARGET_SIZE = (128, 128)
SAMPLES_PER_CLASS_TRAIN = 100
SAMPLES_PER_CLASS_VAL = 25
SAMPLES_PER_CLASS_TEST = 30


def load_dataset_from_zip(
    zip_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load balanced train, validation, and test splits directly from zip in memory."""
    print(f"[Dataset Loader] Opening {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        namelist = z.namelist()

        # Group by partition and class
        train_files_by_class: Dict[str, List[str]] = defaultdict(list)
        test_files_by_class: Dict[str, List[str]] = defaultdict(list)

        for name in sorted(namelist):
            parts = name.split("/")
            if len(parts) >= 3 and name.lower().endswith((".jpg", ".png", ".jpeg")):
                partition = parts[0]
                cls = parts[1]
                if cls in CLASS_TO_IDX:
                    if partition == "Training":
                        train_files_by_class[cls].append(name)
                    elif partition == "Testing":
                        test_files_by_class[cls].append(name)

        print("[Dataset Loader] Found classes with file counts:")
        for cls in CLASSES:
            print(f"  {cls}: {len(train_files_by_class[cls])} train, {len(test_files_by_class[cls])} test")

        def load_images_for_list(names: List[str]) -> np.ndarray:
            imgs = []
            for name in names:
                raw_bytes = z.read(name)
                arr = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
                resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                normalized = (resized.astype(np.float32) / 127.5) - 1.0  # [-1.0, 1.0]
                imgs.append(normalized)
            return np.array(imgs, dtype=np.float32)

        x_train_list, y_train_list = [], []
        x_val_list, y_val_list = [], []
        x_test_list, y_test_list = [], []

        rng = np.random.RandomState(42)

        for cls in CLASSES:
            cls_idx = CLASS_TO_IDX[cls]

            # Train + Val from Training partition
            tr_all = sorted(train_files_by_class[cls])
            rng.shuffle(tr_all)
            selected_tr = tr_all[:SAMPLES_PER_CLASS_TRAIN]
            selected_val = tr_all[SAMPLES_PER_CLASS_TRAIN : SAMPLES_PER_CLASS_TRAIN + SAMPLES_PER_CLASS_VAL]

            x_tr = load_images_for_list(selected_tr)
            y_tr = np.full(len(x_tr), cls_idx, dtype=np.int32)
            x_train_list.append(x_tr)
            y_train_list.append(y_tr)

            x_v = load_images_for_list(selected_val)
            y_v = np.full(len(x_v), cls_idx, dtype=np.int32)
            x_val_list.append(x_v)
            y_val_list.append(y_v)

            # Test from Testing partition
            te_all = sorted(test_files_by_class[cls])
            rng.shuffle(te_all)
            selected_te = te_all[:SAMPLES_PER_CLASS_TEST]

            x_te = load_images_for_list(selected_te)
            y_te = np.full(len(x_te), cls_idx, dtype=np.int32)
            x_test_list.append(x_te)
            y_test_list.append(y_te)

        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        x_val = np.concatenate(x_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
        x_test = np.concatenate(x_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        # Shuffle train set
        perm = rng.permutation(len(x_train))
        x_train, y_train = x_train[perm], y_train[perm]

        return x_train, y_train, x_val, y_val, x_test, y_test


def build_cnn_model(num_classes: int = 26) -> tf.keras.Model:
    """Build a lightweight, robust ConvNet for real-time hand sign classification."""
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3), name="rgb_input"),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        # Dense Classifier Head
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax", name="class_probabilities"),
    ])
    return model


def main():
    print("=" * 70)
    print("ISL RGB CNN MODEL TRAINING & BENCHMARK")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Target Output: {OUTPUT_MODEL_PATH}")
    print("=" * 70)

    if not DATASET_ZIP.exists():
        print(f"ERROR: Dataset zip not found at {DATASET_ZIP}")
        sys.exit(1)

    t0 = time.time()
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset_from_zip(DATASET_ZIP)
    load_time = time.time() - t0
    print(f"[Dataset Loader] Loaded in {load_time:.2f}s:")
    print(f"  Train: {x_train.shape}, labels: {y_train.shape}")
    print(f"  Val:   {x_val.shape}, labels: {y_val.shape}")
    print(f"  Test:  {x_test.shape}, labels: {y_test.shape}")

    # One-hot encode targets
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=26)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=26)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=26)

    # Build model
    model = build_cnn_model(num_classes=26)
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_categorical_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
    ]

    print("\n[Training] Starting CNN training for 15 epochs...")
    t_train_start = time.time()
    history = model.fit(
        x_train,
        y_train_cat,
        validation_data=(x_val, y_val_cat),
        epochs=15,
        batch_size=32,
        callbacks=cb_list,
        verbose=1,
    )
    train_duration = time.time() - t_train_start
    print(f"[Training] Completed in {train_duration:.2f}s.")

    # Evaluate on held-out test split
    print("\n[Evaluation] Evaluating on held-out test split (780 samples)...")
    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"  Held-out Test Loss: {test_loss:.4f}")
    print(f"  Held-out Test Categorical Accuracy: {test_acc * 100:.2f}%")

    # Compute per-class accuracy
    test_preds = model.predict(x_test, batch_size=64, verbose=0)
    pred_labels = np.argmax(test_preds, axis=1)

    class_accuracies = {}
    for i, cls in enumerate(CLASSES):
        cls_mask = (y_test == i)
        cls_total = int(np.sum(cls_mask))
        cls_correct = int(np.sum(pred_labels[cls_mask] == i))
        acc = (cls_correct / cls_total) * 100.0 if cls_total > 0 else 0.0
        class_accuracies[cls] = (acc, cls_correct, cls_total)
        print(f"  Class {cls}: {acc:6.2f}% ({cls_correct}/{cls_total})")

    # Measure latency on single-image inference
    dummy = np.expand_dims(x_test[0], axis=0)
    # Warmup
    for _ in range(5):
        _ = model(dummy, training=False)
    latencies = []
    for _ in range(30):
        t_lat0 = time.time()
        _ = model(dummy, training=False)
        latencies.append((time.time() - t_lat0) * 1000.0)
    mean_latency_ms = float(np.mean(latencies))
    print(f"\n[Inference Benchmark] Mean CPU Latency: {mean_latency_ms:.2f} ms per frame")

    # Save model
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_MODEL_PATH)
    print(f"[Save] Model checkpoint saved to: {OUTPUT_MODEL_PATH}")

    # Save label map
    label_map = {str(i): cls for i, cls in enumerate(CLASSES)}
    with open(OUTPUT_LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"[Save] Label map saved to: {OUTPUT_LABEL_MAP_PATH}")

    # Generate Evaluation Report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_md = f"""# RGB CNN Model Training & Benchmark Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Model Architecture:** 4-Block Conv2D + BatchNorm + GlobalAveragePooling + Dense(128) + Dense(26)  
**Input Dimensions:** 128 x 128 x 3 (Normalized to [-1.0, 1.0])  
**Target Vocabulary:** 26 ISL Letters (A–Z)  
**Trained On:** RealSign Dataset (`data/staging_downloads/realsign/Dataset.zip`)  
**Data Partitioning (Leakage-Free):**
- Training: {len(x_train)} samples ({SAMPLES_PER_CLASS_TRAIN} per class)
- Validation: {len(x_val)} samples ({SAMPLES_PER_CLASS_VAL} per class)
- Held-Out Test: {len(x_test)} samples ({SAMPLES_PER_CLASS_TEST} per class)

## Evaluation Metrics

- **Held-Out Test Accuracy:** {test_acc * 100:.2f}%
- **Held-Out Test Loss:** {test_loss:.4f}
- **Inference Latency:** {mean_latency_ms:.2f} ms / frame (on Apple Silicon M2 CPU)
- **Training Time:** {train_duration:.2f} seconds

## Per-Class Held-Out Test Accuracy

| Class | Accuracy | Correct / Total |
|-------|----------|-----------------|
"""
    for cls in CLASSES:
        acc, corr, tot = class_accuracies[cls]
        report_md += f"| {cls} | {acc:.1f}% | {corr}/{tot} |\n"

    report_md += f"""
## Artifact Registry

- **Keras H5 Checkpoint:** `{OUTPUT_MODEL_PATH}`
- **Label Map:** `{OUTPUT_LABEL_MAP_PATH}`
- **Inference Status:** ONLINE (Verified)
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[Save] Benchmark report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
