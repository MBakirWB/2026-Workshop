# W&B Workshop 2026: Aquatic Species Classification

Build an image classifier for marine species and learn the full MLOps lifecycle with Weights & Biases -- from first experiment to production model.

## What you'll do

Using the AQUA20 underwater dataset (20 marine species, 8K+ images), you'll train a baseline model, run a hyperparameter sweep, and promote the best model to production -- all tracked end-to-end in W&B.

**Topics covered:**
- Experiment tracking (runs, config, metrics, alerts)
- Visual logging (images, tables, ROC curves)
- Artifacts (versioning, lineage, TTL, reference artifacts)
- Model Registry (staging, promotion)
- Sweeps (hyperparameter optimization, sweep controls)

## Getting started

### 1. Environment

Make sure these packages are available in your environment:

```
wandb torch torchvision timm scikit-learn numpy Pillow tqdm
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

### 2. Workshop notebook

Everything lives in `workshop_material/`:

```
workshop_material/
  aqua20_with_wandb.ipynb   # Main workshop notebook
  workshop_utils.py         # ML boilerplate (data loading, training loops, etc.)
```

Open `aqua20_with_wandb.ipynb` and follow along. The notebook is self-contained -- datasets and pretrained model weights are pre-loaded as W&B artifacts.

### 3. W&B login

You'll need your W&B credentials. The notebook walks you through authentication in the first few cells.

## Supplementary material

New to W&B? Check `supplementary_101_notebooks/` for standalone introductions to experiment tracking and artifacts/registry.

## Repo structure

```
2025-Workshop/
  workshop_material/        # Notebook + utilities (start here)
  supplementary_101_notebooks/  # Optional deep-dives on W&B concepts
  admin_setup_only/         # Admin-only: dataset + model upload scripts
  requirements.txt          # Python dependencies
```
