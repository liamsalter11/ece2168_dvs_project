#!/usr/bin/env python3
"""
train_gesture_model.py

Fine-tunes MobileNetV2 on HaGRID gesture images and exports a quantized
TFLite model for deployment on PC (x86), Raspberry Pi 4, and E1 EVK.

─── Data setup ────────────────────────────────────────────────────────────────
Collect training data using collect_gesture_data.py before running this script:

  python collect_gesture_data.py --gesture palm
  python collect_gesture_data.py --gesture fist
  python collect_gesture_data.py --gesture one
  python collect_gesture_data.py --gesture peace
  python collect_gesture_data.py --gesture thumb_up

This produces DVS activity map images in:
  data/
    palm/        →  OPEN_HAND
    fist/        →  FIST
    one/         →  POINTING
    peace/       →  PEACE
    thumb_up/    →  THUMBS_UP

─── Usage ─────────────────────────────────────────────────────────────────────
  pip install tensorflow[and-cuda] pillow tqdm
  python train_gesture_model.py --data_dir data/
  python train_gesture_model.py --data_dir data/ --output gesture_model.tflite

─── Output ────────────────────────────────────────────────────────────────────
  gesture_model.tflite   — int8-quantized TFLite model (~300–600 KB)
  gesture_labels.txt     — label index → gesture name mapping
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_SIZE        = 96      # spatial resolution (96×96×3) — sized to fit E1x SRAM
ALPHA             = 0.35    # MobileNetV2 width multiplier — 0.35 trims activation arena
BATCH_SIZE        = 32
EPOCHS_HEAD       = 10      # Phase 1: train classification head only
EPOCHS_FINETUNE   = 10      # Phase 2: fine-tune top layers of base
LR_HEAD           = 1e-3
LR_FINETUNE       = 1e-5

# Maps HaGRID subdirectory name → gesture_class_t index (must match protocol.h)
CLASS_MAP = {
    'palm':      1,   # GESTURE_OPEN_HAND
    'fist':      2,   # GESTURE_FIST
    'one':       3,   # GESTURE_POINTING
    'peace':     4,   # GESTURE_PEACE
    'thumb_up':  5,   # GESTURE_THUMBS_UP
}
CLASS_NAMES = sorted(CLASS_MAP.keys())  # alphabetical order for Keras


# ── Dataset ────────────────────────────────────────────────────────────────────

def preprocess(image, label):
    """Convert RGB image to grayscale-replicated float32.

    Training on grayscale closes the domain gap between RGB photos and the
    single-channel DVS activity accumulation frame that will be fed at runtime.
    """
    gray = tf.image.rgb_to_grayscale(image)          # [H, W, 1]
    rgb  = tf.repeat(gray, 3, axis=-1)               # [H, W, 3]
    rgb  = tf.cast(rgb, tf.float32) / 255.0          # normalize to [0, 1]
    return rgb, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def load_dataset(data_dir, split, batch_size):
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        class_names=CLASS_NAMES,
        image_size=(INPUT_SIZE, INPUT_SIZE),
        batch_size=batch_size,
        validation_split=0.2,
        subset=split,
        seed=42,
        label_mode='categorical',
    )
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if split == 'training':
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        alpha=ALPHA,
        include_top=False,
        weights='imagenet',
    )
    base.trainable = False

    inputs  = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    x       = base(inputs, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs), base


# ── Export ─────────────────────────────────────────────────────────────────────

def export_tflite(model, output_path, rep_ds):
    """Export with full int8 quantization (weights + activations)."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for images, _ in rep_ds.take(50):
            for img in images:
                yield [tf.expand_dims(img, 0)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"[export] {output_path}  ({len(tflite_model) / 1024:.1f} KB)")


def write_labels(output_path):
    """Write label file mapping index → gesture name (matching protocol.h)."""
    # Keras assigns class indices alphabetically over CLASS_NAMES.
    # We remap to gesture_class_t values from protocol.h.
    lines = []
    for keras_idx, name in enumerate(CLASS_NAMES):
        proto_idx = CLASS_MAP[name]
        lines.append(f"{keras_idx} {proto_idx} {name.upper()}")
    labels_path = Path(output_path).with_name('gesture_labels.txt')
    labels_path.write_text('\n'.join(lines) + '\n')
    print(f"[export] {labels_path}")
    for l in lines:
        print(f"         {l}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train gesture TFLite model')
    parser.add_argument('--data_dir',      required=True,
                        help='Root directory with palm/, fist/, one/, peace/, thumb_up/ subdirs')
    parser.add_argument('--output',        default='gesture_model.tflite')
    parser.add_argument('--epochs_head',   type=int, default=EPOCHS_HEAD)
    parser.add_argument('--epochs_fine',   type=int, default=EPOCHS_FINETUNE)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    missing = [c for c in CLASS_NAMES if not (data_dir / c).is_dir()]
    if missing:
        print(f"[error] Missing subdirectories in {data_dir}: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"[train] TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"[train] GPUs available: {[g.name for g in gpus]}")

    train_ds = load_dataset(data_dir, 'training',   BATCH_SIZE)
    val_ds   = load_dataset(data_dir, 'validation', BATCH_SIZE)

    model, base = build_model(len(CLASS_NAMES))
    model.summary()

    # ── Phase 1: head only ────────────────────────────────────────────────────
    print(f"\n[train] Phase 1 — training head ({args.epochs_head} epochs)")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_HEAD),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head)

    # ── Phase 2: fine-tune top layers of base ─────────────────────────────────
    print(f"\n[train] Phase 2 — fine-tuning top layers ({args.epochs_fine} epochs)")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_fine)

    # ── Export ────────────────────────────────────────────────────────────────
    print(f"\n[export] Quantizing and exporting to {args.output}")
    export_tflite(model, args.output, train_ds)
    write_labels(args.output)
    print("\n[done] Run the C++ receiver with: --model gesture_model.tflite")


if __name__ == '__main__':
    main()
