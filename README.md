# W&B Workshop 2026: Aquatic Species Classification

Build an image classifier for marine species and learn the full MLOps lifecycle with Weights & Biases -- from first experiment to production model.

## What you'll do

Using the AQUA underwater dataset (20 marine species, 8K+ images), you'll train a baseline model, run a hyperparameter sweep, and promote the best model to production -- all tracked end-to-end in W&B.

**Topics covered:**
- Experiment tracking (runs, config, metrics, alerts)
- Visual logging (images, tables, ROC curves)
- Artifacts (versioning, lineage, TTL, reference artifacts)
- Model Registry (staging, promotion)
- Sweeps (hyperparameter optimization, sweep controls)

## Getting started

### 1. Set up your Python environment

If you already have a pre-provisioned environment (e.g., a JupyterHub kernel), skip to step 2.

Otherwise, create a virtual environment and install dependencies:

```bash
cd 2026-Workshop
python -m venv workshop
source workshop/bin/activate   # On Windows: workshop\Scripts\activate
pip install -r requirements.txt
```

> **Note:** If you plan to run the notebook in Jupyter, make sure `jupyter` and `ipykernel` are installed in the same environment, then register the kernel:
> ```bash
> pip install jupyter ipykernel
> python -m ipykernel install --user --name workshop --display-name "Workshop (Python)"
> ```

### 2. Verify local data

The datasets and pretrained model weights should already be pre-loaded in `workshop_material/`:

```
workshop_material/
  data/
    train/               # ~6,500 training images (20 class subfolders)
    val/                 # ~800 validation images
    test/                # ~800 test images
  pretrained_weights/
    resnet50_imagenet.pth
    efficientnet_b0_imagenet.pth
```

The notebook reads data from these local directories. It also calls `use_artifact()` to track lineage in W&B -- so your training runs record exactly which dataset versions were used, without needing to download anything.

If the `data/` or `pretrained_weights/` directories are missing, ask your workshop facilitator. In a pinch, you can run the fallback script yourself (requires internet access):

```bash
cd admin_setup_only
pip install datasets torch timm Pillow numpy scikit-learn
python prepare_local_data.py
```

### 3. Open the workshop notebook

Everything lives in `workshop_material/`:

```
workshop_material/
  aqua_with_wandb.ipynb     # Main workshop notebook (start here)
  workshop_utils.py         # ML boilerplate (data loading, training loops, etc.)
```

Open `aqua_with_wandb.ipynb` and follow along.

### 4. W&B login

You'll need your W&B credentials. The notebook walks you through authentication in the first few cells.

## Supplementary material

New to W&B? Check `supplementary_101_notebooks/` for standalone introductions to experiment tracking and artifacts/registry.

## Repo structure

```
2026-Workshop/
  workshop_material/            # Notebook + utilities + pre-loaded data (start here)
  supplementary_101_notebooks/  # Optional deep-dives on W&B concepts
  admin_setup_only/             # Admin-only: dataset + model upload scripts
  requirements.txt              # Python dependencies
```
