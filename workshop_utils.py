"""
Workshop Utilities - AQUA20 Marine Species Classification

This module contains all the ML boilerplate code for the workshop.
Participants can focus on W&B concepts while these utilities handle:
- Data loading and transforms
- Model creation
- Training and evaluation loops

Usage:
    from workshop_utils import *
"""

import os
import random
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
import timm
import wandb

warnings.filterwarnings('ignore')

# =============================================================================
# CONSTANTS
# =============================================================================

# AQUA20 class names (20 marine species)
CLASS_NAMES = [
    "coral", "crab", "diver", "eel", "fish", "fishInGroups", "flatworm", 
    "jellyfish", "marine_dolphin", "octopus", "rayfish", "seaAnemone", 
    "seaCucumber", "seaSlug", "seaUrchin", "shark", "shrimp", "squid", 
    "starfish", "turtle"
]
NUM_CLASSES = len(CLASS_NAMES)

# ImageNet normalization (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# DEVICE SETUP
# =============================================================================

def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int = 42, deterministic: bool = False):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =============================================================================
# DATA TRANSFORMS
# =============================================================================

def get_transforms(image_size: int = 224, is_training: bool = True):
    """Get image transforms optimized for underwater imagery."""
    if is_training:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

# =============================================================================
# DATASET CLASS
# =============================================================================

class AquaticDataset(Dataset):
    """PyTorch Dataset for images organized by class folders."""
    
    def __init__(self, root_dir: str, transform=None, class_names: List[str] = None, 
                 max_samples: int = None):
        """
        Args:
            root_dir: Path to directory with class subfolders
            transform: Torchvision transforms to apply
            class_names: List of class names (uses CLASS_NAMES if None)
            max_samples: Limit samples for fast training (None = use all)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = class_names or CLASS_NAMES
        self.samples = []  # List of (image_path, class_idx)
        
        # Collect all image files
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(class_dir, img_name), class_idx))
        
        # Optionally limit samples for fast training
        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_raw_image(self, idx):
        """Get raw PIL image without transforms (for visualization)."""
        img_path, label = self.samples[idx]
        return Image.open(img_path).convert("RGB"), label

# =============================================================================
# MODEL CREATION
# =============================================================================

def create_model(model_name: str = "resnet50", num_classes: int = NUM_CLASSES, 
                 pretrained: bool = True):
    """Create a model using the timm library."""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_one_epoch(
    model, 
    train_loader, 
    criterion, 
    optimizer, 
    scaler, 
    device,
    epoch: int,
    use_amp: bool = True,
    log_interval: int = 1,
    run=None
) -> Tuple[float, float]:
    """Train for one epoch with mixed precision and W&B logging."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            "loss": running_loss / (batch_idx + 1),
            "acc": 100. * correct / total
        })
        
        # Log to W&B
        if run and (batch_idx + 1) % log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            wandb.log({
                "train/loss_step": loss.item(),
                "train/acc_step": 100. * correct / total,
            }, step=global_step)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model, 
    data_loader, 
    criterion, 
    device,
    use_amp: bool = True,
    desc: str = "Eval"
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model and return metrics + predictions."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(data_loader, desc=desc)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
        pbar.set_postfix({"loss": running_loss / (len(all_preds) // data_loader.batch_size + 1)})

    epoch_loss = running_loss / len(data_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)

# =============================================================================
# W&B HELPERS
# =============================================================================

def generate_run_name(config: Dict) -> str:
    """Generate a descriptive run name from config.
    
    Format: {model}-lr{learning_rate}-bs{batch_size}-ep{epochs}
    Example: resnet50-lr1e-3-bs32-ep3
    """
    model = config.get("model_name", "model")
    lr = config.get("learning_rate", 1e-3)
    bs = config.get("batch_size", 32)
    ep = config.get("epochs", 3)
    
    # Format learning rate nicely (1e-3 -> 1e-3, 0.001 -> 1e-3)
    lr_str = f"{lr:.0e}".replace("-0", "-")
    
    return f"{model}-lr{lr_str}-bs{bs}-ep{ep}"

def update_best_metric(run, key: str, value: float, mode: str = "max"):
    """Update run summary with best value (for leaderboards)."""
    current_best = run.summary.get(key)
    if current_best is None:
        run.summary[key] = value
    elif mode == "max" and value > current_best:
        run.summary[key] = value
    elif mode == "min" and value < current_best:
        run.summary[key] = value

# =============================================================================
# QUICK START FUNCTION
# =============================================================================

def setup_training(config: Dict) -> Tuple:
    """
    One-line setup for training components.
    
    Returns: model, criterion, optimizer, scaler, device
    """
    set_seed(config.get("seed", 42))
    
    model = create_model(
        config.get("model_name", "resnet50"),
        config.get("num_classes", NUM_CLASSES),
        config.get("pretrained", True)
    )
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 0.01)
    )
    scaler = GradScaler(enabled=config.get("use_amp", True))
    
    return model, criterion, optimizer, scaler, DEVICE


def load_datasets_from_artifacts(run, train_artifact: str, val_artifact: str, 
                                  config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Download W&B artifacts and create DataLoaders.
    
    This creates lineage in W&B showing which datasets were used!
    """
    # Download artifacts (creates lineage!)
    train_art = run.use_artifact(train_artifact, type='dataset')
    train_dir = train_art.download()
    
    val_art = run.use_artifact(val_artifact, type='dataset')
    val_dir = val_art.download()
    
    # Create datasets
    image_size = config.get("image_size", 224)
    train_dataset = AquaticDataset(
        train_dir, 
        transform=get_transforms(image_size, is_training=True),
        max_samples=config.get("max_samples")
    )
    val_dataset = AquaticDataset(
        val_dir, 
        transform=get_transforms(image_size, is_training=False)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True
    )
    
    print(f"📦 Loaded {len(train_dataset)} training samples")
    print(f"📦 Loaded {len(val_dataset)} validation samples")
    
    return train_loader, val_loader, train_dataset, val_dataset


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def create_prediction_images(dataset, preds, probs, class_names, n_samples=16, image_size=224):
    """
    Create a list of wandb.Image objects with prediction captions.
    
    Args:
        dataset: Dataset with get_raw_image() method
        preds: Predicted class indices
        probs: Prediction probabilities (n_samples x n_classes)
        class_names: List of class names
        n_samples: Number of samples to create
        image_size: Resize images to this size for consistent display
        
    Returns:
        List of wandb.Image objects ready for logging
    """
    images_to_log = []
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    for idx in indices:
        image, true_label = dataset.get_raw_image(idx)
        pred_label = preds[idx]
        confidence = probs[idx][pred_label] * 100

        true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
        pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)

        is_correct = "✓" if pred_label == true_label else "✗"
        caption = f"{is_correct} True: {true_name} | Pred: {pred_name} ({confidence:.1f}%)"

        # Resize for consistent display in W&B UI
        image = image.resize((image_size, image_size))
        images_to_log.append(wandb.Image(image, caption=caption))

    return images_to_log


def create_predictions_table(dataset, preds, probs, class_names, n_samples=100, image_size=224):
    """
    Create a W&B Table with predictions for detailed analysis.
    
    Features in W&B UI:
    - Group by 'truth' to see recall (false negatives per class)
    - Group by 'guess' to see precision (false positives per class)  
    - Filter by row['truth'] != row['guess'] to find all errors
    - Sort by score columns to find high-confidence mistakes
    
    Args:
        dataset: Dataset with get_raw_image() method
        preds: Predicted class indices
        probs: Prediction probabilities (n_samples x n_classes)
        class_names: List of class names
        n_samples: Number of samples to include
        image_size: Resize images to this size
        
    Returns:
        wandb.Table ready for logging
    """
    # Build columns: image, prediction info, then score for each class
    columns = ["image", "id", "guess", "truth", "correct"]
    columns.extend([f"score_{name}" for name in class_names])
    
    table = wandb.Table(columns=columns)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    for idx in indices:
        image, true_label = dataset.get_raw_image(idx)
        pred_label = preds[idx]
        is_correct = pred_label == true_label

        true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
        pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)

        # Resize for consistent display
        image = image.resize((image_size, image_size))
        
        # Build row data
        row_data = [
            wandb.Image(image),
            str(idx),  # Unique ID for joining tables across runs
            pred_name,
            true_name,
            is_correct,
        ]
        
        # Add confidence scores for each class (enables histogram visualization)
        for class_idx in range(len(class_names)):
            score = float(probs[idx][class_idx]) if class_idx < len(probs[idx]) else 0.0
            row_data.append(round(score, 4))
        
        table.add_data(*row_data)

    return table


def create_confusion_matrix(y_true, y_pred, class_names, normalize=True):
    """
    Create a confusion matrix figure.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the matrix
        
    Returns:
        matplotlib Figure object ready for wandb.Image()
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    # Get unique classes that appear in the data (handles subsets)
    unique_classes = sorted(set(y_true) | set(y_pred))
    display_names = [class_names[i] if i < len(class_names) else str(i) for i in unique_classes]

    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)  # Handle division by zero

    fig, ax = plt.subplots(figsize=(max(10, len(unique_classes)), max(8, len(unique_classes) * 0.8)))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(display_names)),
        yticks=np.arange(len(display_names)),
        xticklabels=display_names,
        yticklabels=display_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix (Normalized)' if normalize else 'Confusion Matrix'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(len(display_names)):
        for j in range(len(display_names)):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8 if len(display_names) < 15 else 6)

    plt.tight_layout()
    return fig


def compute_per_class_metrics(y_true, y_pred, class_names):
    """
    Compute per-class precision, recall, and F1 scores.
    
    Returns:
        Dict with per-class metrics ready for wandb.log()
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    metrics = {}
    for i, name in enumerate(class_names):
        if support[i] > 0:  # Only include classes present in data
            metrics[f"per_class/{name}/precision"] = precision[i]
            metrics[f"per_class/{name}/recall"] = recall[i]
            metrics[f"per_class/{name}/f1"] = f1[i]
            metrics[f"per_class/{name}/support"] = int(support[i])
    
    return metrics


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "CLASS_NAMES", "NUM_CLASSES", "DEVICE",
    # Functions
    "set_seed", "get_device", "get_transforms",
    "create_model", "count_parameters",
    "train_one_epoch", "evaluate",
    "generate_run_name", "update_best_metric",
    "setup_training", "load_datasets_from_artifacts",
    # Visualization helpers
    "create_prediction_images", "create_predictions_table",
    "create_confusion_matrix", "compute_per_class_metrics",
    # Classes
    "AquaticDataset",
]

# Print confirmation when imported
print(f"   Workshop utilities loaded")
print(f"   Device: {DEVICE}")
print(f"   Classes: {NUM_CLASSES} marine species")
