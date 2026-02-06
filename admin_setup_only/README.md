# Workshop Admin Setup

Run this **once** before the workshop from a machine with internet access and connectivity to the W&B instance.

This script downloads the AQUA dataset from HuggingFace, creates train/val/test splits, builds an EDA exploration table, and uploads pretrained model weights (resnet50, efficientnet_b0) -- all as W&B artifacts. Participants pull everything from W&B during the workshop.

## Prerequisites

- Python 3.10+
- Access to the target W&B instance
- Internet access (HuggingFace downloads)
- A W&B team created on the instance for workshop participants (this is the `--entity` value)

## Setup

```bash
git clone <repo-url>
cd 2025-Workshop/admin_setup_only
python -m venv workshop-setup
source workshop-setup/bin/activate
pip install -r ../requirements.txt
```

## Run

```bash
python setup.py --entity <your-wandb-team> --host <wandb-instance-url>
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--entity` | Yes | — | W&B team or username |
| `--host` | No | `https://api.wandb.ai` | W&B instance URL (for self-hosted) |

The project is fixed to `SIE-Workshop-2026`.

## What it creates in W&B

| Artifact | Type | Description |
|----------|------|-------------|
| `aqua-raw-dataset` | dataset | Full AQUA20 dataset (8,171 images, 20 classes) |
| `aqua-train` | dataset | Training split (80%) |
| `aqua-val` | dataset | Validation split (10%) |
| `aqua-test` | dataset | Test split (10%) |
| `pretrained-resnet50` | pretrained-weights | ImageNet weights for resnet50 |
| `pretrained-efficientnet_b0` | pretrained-weights | ImageNet weights for efficientnet_b0 |

It also logs a `dataset-eda-exploration` run with an interactive table for data exploration.

## Participant environment

Participants need these packages pre-installed (see `../requirements.txt`):

```
wandb torch torchvision timm scikit-learn numpy Pillow tqdm
```

They do **not** need `datasets` (HuggingFace) -- all data comes from W&B artifacts.
