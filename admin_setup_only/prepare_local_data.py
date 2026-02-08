#!/usr/bin/env python3
"""
Prepare Local Workshop Data (No W&B Required)

Emergency fallback script that downloads the AQUA20 dataset from HuggingFace,
creates train/val/test splits, downloads pretrained model weights, and places
everything into ../workshop_material/ for participants.

This script does NOT require a W&B account or internet access to a W&B instance.
It only needs internet access to HuggingFace and PyTorch model hubs.

Usage:
    pip install datasets torch timm Pillow numpy scikit-learn
    python prepare_local_data.py
"""

import os
import json
import shutil
import random
import argparse
from datetime import datetime
from PIL import Image

import numpy as np
import torch
import timm
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_NAME = "taufiktrf/aqua20"
WORKSHOP_MODELS = ["resnet50", "efficientnet_b0"]

# Temporary working directories (cleaned up after)
DATA_DIR = "aqua_raw_data"
SPLITS_DIR = "aqua_splits"

# Output directories
PARTICIPANT_DIR = os.path.join("..", "workshop_material")
PARTICIPANT_DATA_DIR = os.path.join(PARTICIPANT_DIR, "data")
PARTICIPANT_WEIGHTS_DIR = os.path.join(PARTICIPANT_DIR, "pretrained_weights")

# Split configuration (must match setup.py to produce identical splits)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42


def create_split_indices(image_files, train_ratio, val_ratio, test_ratio, seed):
    """Create train/val/test split indices."""
    random.seed(seed)
    indices = list(range(len(image_files)))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio / (train_ratio + val_ratio), random_state=seed
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def get_all_image_files(data_dir, class_names):
    """Get all image files from the data directory, organized by class."""
    image_files = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append({
                        "path": os.path.join(class_dir, filename),
                        "class_name": class_name,
                        "filename": filename
                    })
    return image_files


def create_split_directory(split_name, image_files, indices, class_names, base_dir):
    """Create a directory with images for a specific split."""
    split_dir = os.path.join(base_dir, split_name)
    if os.path.exists(split_dir):
        shutil.rmtree(split_dir)
    for class_name in class_names:
        os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)

    samples_per_class = {name: 0 for name in class_names}
    for idx in indices:
        img_info = image_files[idx]
        src_path = img_info["path"]
        class_name = img_info["class_name"]
        new_filename = f"{class_name}_{samples_per_class[class_name]:05d}.jpg"
        dst_path = os.path.join(split_dir, class_name, new_filename)
        shutil.copy2(src_path, dst_path)
        samples_per_class[class_name] += 1

    return split_dir, samples_per_class


def main():
    print("=" * 60)
    print("  Prepare Local Workshop Data (No W&B Required)")
    print("=" * 60)

    # Step 1: Download dataset from HuggingFace
    print("\nStep 1: Loading dataset from HuggingFace...")
    dataset = load_dataset(DATASET_NAME)
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    data = dataset[split_name]

    if hasattr(data.features.get("label", None), "names"):
        class_names = data.features["label"].names
    else:
        class_names = [str(c) for c in sorted(set(data["label"]))]

    print(f"  {len(data)} images, {len(class_names)} classes")

    # Step 2: Save images to disk
    print("\nStep 2: Saving images to disk...")
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    for class_name in class_names:
        os.makedirs(os.path.join(DATA_DIR, class_name), exist_ok=True)

    saved_counts = {name: 0 for name in class_names}
    for idx in range(len(data)):
        sample = data[idx]
        image = sample["image"]
        label = sample["label"]
        class_name = class_names[label]
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_filename = f"{class_name}_{saved_counts[class_name]:05d}.jpg"
        img_path = os.path.join(DATA_DIR, class_name, img_filename)
        image.save(img_path, "JPEG", quality=95)
        saved_counts[class_name] += 1

    total_saved = sum(saved_counts.values())
    print(f"  Saved {total_saved} images to {DATA_DIR}/")

    # Step 3: Create train/val/test splits
    print("\nStep 3: Creating train/val/test splits...")
    image_files = get_all_image_files(DATA_DIR, class_names)
    split_indices = create_split_indices(
        image_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED
    )

    if os.path.exists(SPLITS_DIR):
        shutil.rmtree(SPLITS_DIR)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    for name, indices in split_indices.items():
        split_dir, _ = create_split_directory(
            name, image_files, indices, class_names, SPLITS_DIR
        )
        n_files = sum(len(files) for _, _, files in os.walk(split_dir))
        print(f"  {name}: {n_files} images")

    # Step 4: Download pretrained model weights
    print("\nStep 4: Downloading pretrained model weights...")
    for model_name in WORKSHOP_MODELS:
        print(f"  Downloading {model_name} from timm (HuggingFace)...")
        model = timm.create_model(model_name, pretrained=True)
        weights_path = f"{model_name}_imagenet.pth"
        torch.save(model.state_dict(), weights_path)
        print(f"  Saved {weights_path}")

    # Step 5: Copy everything to participant directory
    print(f"\nStep 5: Copying to {os.path.abspath(PARTICIPANT_DIR)}/...")

    # Copy splits
    if os.path.exists(PARTICIPANT_DATA_DIR):
        shutil.rmtree(PARTICIPANT_DATA_DIR)
    shutil.copytree(SPLITS_DIR, PARTICIPANT_DATA_DIR)
    print(f"  Copied splits to {PARTICIPANT_DATA_DIR}/")

    # Copy weights
    if os.path.exists(PARTICIPANT_WEIGHTS_DIR):
        shutil.rmtree(PARTICIPANT_WEIGHTS_DIR)
    os.makedirs(PARTICIPANT_WEIGHTS_DIR, exist_ok=True)
    for model_name in WORKSHOP_MODELS:
        src = f"{model_name}_imagenet.pth"
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(PARTICIPANT_WEIGHTS_DIR, src))
    print(f"  Copied weights to {PARTICIPANT_WEIGHTS_DIR}/")

    # Cleanup
    print("\nCleaning up temporary files...")
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    shutil.rmtree(SPLITS_DIR, ignore_errors=True)
    for model_name in WORKSHOP_MODELS:
        weights_file = f"{model_name}_imagenet.pth"
        if os.path.exists(weights_file):
            os.remove(weights_file)

    print("\n" + "=" * 60)
    print("  Done! Local data is ready.")
    print("=" * 60)
    print(f"\n  {PARTICIPANT_DATA_DIR}/train/")
    print(f"  {PARTICIPANT_DATA_DIR}/val/")
    print(f"  {PARTICIPANT_DATA_DIR}/test/")
    print(f"  {PARTICIPANT_WEIGHTS_DIR}/resnet50_imagenet.pth")
    print(f"  {PARTICIPANT_WEIGHTS_DIR}/efficientnet_b0_imagenet.pth")
    print(f"\n  Participants can now open the notebook and start the workshop.")


if __name__ == "__main__":
    main()
