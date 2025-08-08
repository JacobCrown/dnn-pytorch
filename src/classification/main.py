"""Main training pipeline for image classification with Hydra configuration."""
# pyright: reportMissingImports=false

import logging
import os
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from src.common.augmentations import get_augmentations
from src.common.model_loader import get_model
from src.common.logging import setup_logging
from src.classification.dataset import ClassificationDataset
from src.classification.trainer import ClassificationTrainer
from src.common.utils import (
    create_data_loaders,
    set_seed,
    plot_training_history,
    plot_confusion_matrix,
)
from sklearn.metrics import confusion_matrix

# Note: real log file will be set inside main() after Hydra changes cwd


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="classification/main.yaml",
)
def main(cfg: DictConfig) -> None:
    # ---------------------------------------------------------------------
    # Logging setup
    # ---------------------------------------------------------------------
    # Configure logging into the Hydra run dir
    log_dir = Path(os.getcwd()) / "logs"
    setup_logging(str(log_dir / "training.log"))
    logging.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    # Support both shapes: cfg.model.* or cfg.classification.model.*
    root_cfg = cfg
    if not hasattr(cfg, "model") and hasattr(cfg, "classification"):
        root_cfg = cfg.classification

    # Train/aug nodes may live at root or under classification
    train_cfg = getattr(root_cfg, "train", None) or getattr(cfg, "train", None)
    augs_cfg = getattr(root_cfg, "augmentations", None) or getattr(
        cfg, "augmentations", None
    )

    # ---------------------------------------------------------------------
    # Seed & device
    # ---------------------------------------------------------------------
    # Experiment block optional; default seed=42
    try:
        seed_val = int(getattr(getattr(root_cfg, "experiment", {}), "seed", 42))
    except Exception:
        seed_val = 42
    set_seed(seed_val)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, data_preproc = get_model(root_cfg)
    model.to(device)

    # ---------------------------------------------------------------------
    # Augmentations + Preprocessing
    # ---------------------------------------------------------------------
    logging.info("Building augmentation and preprocessing pipelines…")
    # Compose Albumentations pipeline: augmentations + resize + normalize + ToTensor
    # Leverage timm's data_config for size and normalization
    _ = getattr(
        data_preproc, "transforms", None
    )  # not used; normalization derived below
    # Resolve via timm's resolve_data_config instead (encoded inside get_model)
    # Using typical keys from timm data_config
    # We can derive from the model's default config via timm again if needed,
    # but here we just use the prepared normalization values.

    # Rebuild using augmentations but avoid duplicate ToTensorV2
    if augs_cfg is not None:
        train_augs = get_augmentations(
            OmegaConf.create({"augmentations": augs_cfg}), append_to_tensor=False
        )
    else:
        train_augs = A.Compose([])

    # Determine expected size and norm from the model's transform
    # Fallbacks
    try:
        from timm.data.config import resolve_data_config

        data_config = resolve_data_config({}, model=model)
        mean = data_config.get("mean", (0.485, 0.456, 0.406))
        std = data_config.get("std", (0.229, 0.224, 0.225))
        h, w = data_config.get("input_size", (3, 224, 224))[1:]
    except Exception:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        h, w = 224, 224

    train_transforms = A.Compose(
        list(train_augs.transforms)
        + [A.Resize(height=h, width=w), A.Normalize(mean=mean, std=std), ToTensorV2()]
    )

    val_transforms = A.Compose(
        [A.Resize(height=h, width=w), A.Normalize(mean=mean, std=std), ToTensorV2()]
    )

    # ---------------------------------------------------------------------
    # Datasets & loaders
    # ---------------------------------------------------------------------
    data_root = Path(root_cfg.data.root)
    train_dataset = ClassificationDataset(
        data_root / root_cfg.data.train_dir,
        transforms=train_transforms,
        extensions=root_cfg.data.extensions,
    )
    val_dataset = ClassificationDataset(
        data_root / root_cfg.data.val_dir,
        transforms=val_transforms,
        extensions=root_cfg.data.extensions,
    )

    train_loader, val_loader, _ = create_data_loaders(
        train_dataset,
        val_dataset=val_dataset,
        batch_size=int(getattr(train_cfg, "batch_size", 32)),
        num_workers=4,
        shuffle=True,
    )
    logging.info(
        "Datasets and loaders ready. Train samples: %d, Val samples: %d",
        len(train_dataset),
        len(val_dataset),
    )

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    logging.info("Creating model %s", root_cfg.model.name)
    logging.info(
        "Model created — total params: %d", sum(p.numel() for p in model.parameters())
    )

    # ---------------------------------------------------------------------
    # Optimizer & scheduler
    # ---------------------------------------------------------------------
    logging.info("Setting up optimizer and scheduler…")
    # Determine epochs early so we can default scheduler params like T_max
    num_epochs = int(getattr(train_cfg, "epochs", 100))
    optim_name = str(getattr(train_cfg, "optimizer", "adam")).lower()
    if optim_name == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(getattr(train_cfg, "learning_rate", 1e-3)),
            weight_decay=float(getattr(train_cfg, "weight_decay", 1e-4)),
        )
    elif optim_name == "adamw":
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(getattr(train_cfg, "learning_rate", 1e-3)),
            weight_decay=float(getattr(train_cfg, "weight_decay", 1e-4)),
        )
    elif optim_name == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(getattr(train_cfg, "learning_rate", 1e-3)),
            weight_decay=float(getattr(train_cfg, "weight_decay", 1e-4)),
            momentum=float(getattr(train_cfg, "momentum", 0.9)),
            nesterov=True,
        )
    else:
        raise ValueError(f"Unsupported optimizer {optim_name}")

    scheduler = None
    sched_name = str(getattr(train_cfg, "scheduler", "cosine")).lower()
    sched_params = dict(getattr(train_cfg, "scheduler_params", {}) or {})
    if sched_name == "cosine":
        # Default T_max to num_epochs when stepping per epoch
        sched_params.setdefault("T_max", num_epochs)
        sched_params.setdefault("eta_min", 0.0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **sched_params)
    elif sched_name == "step":
        # Provide reasonable defaults
        sched_params.setdefault("step_size", max(1, num_epochs // 3))
        sched_params.setdefault("gamma", 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, **sched_params)
    elif sched_name == "plateau":
        sched_params.setdefault("patience", 3)
        sched_params.setdefault("factor", 0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_params)

    # ---------------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------------
    # Loss function
    loss_name = str(getattr(train_cfg, "loss_function", "cross_entropy")).lower()
    label_smoothing = float(getattr(train_cfg, "label_smoothing", 0.0))

    if loss_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    elif loss_name == "focal":

        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
                super().__init__()
                self.gamma = gamma
                self.alpha = alpha
                self.reduction = reduction

            def forward(self, logits, target):
                ce = nn.functional.cross_entropy(
                    logits, target, reduction="none", weight=None
                )
                pt = torch.exp(-ce)
                loss = ((1 - pt) ** self.gamma) * ce
                if self.reduction == "mean":
                    return loss.mean()
                if self.reduction == "sum":
                    return loss.sum()
                return loss

        criterion = FocalLoss(
            gamma=float(getattr(train_cfg, "focal_gamma", 2.0)),
            alpha=getattr(train_cfg, "focal_alpha", None),
        )
    else:
        raise ValueError(f"Unsupported loss_function {loss_name}")

    # Pyright sometimes narrows scheduler to ReduceLROnPlateau which is not a subclass of _LRScheduler
    # Our trainer supports both types, so type: ignore on this argument
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,  # type: ignore[arg-type]
        cfg=root_cfg,
        train_cfg=train_cfg,
        device=device,
        save_dir=str(Path(os.getcwd()) / "checkpoints"),
        log_dir=str(log_dir),
    )

    # Resume from checkpoint if provided
    resume_path = getattr(train_cfg, "resume", None)
    if resume_path:
        logging.info("Resuming training from %s", resume_path)
        trainer.load_checkpoint(resume_path)

    # ---------------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------------
    logging.info("Starting training for %d epochs", num_epochs)
    history = trainer.train(
        num_epochs=num_epochs,
        save_freq=int(getattr(train_cfg, "save_freq", 1)),
        early_stopping_patience=getattr(train_cfg, "early_stopping_patience", None),
    )

    # ---------------------------------------------------------------------
    # Final evaluation
    # ---------------------------------------------------------------------
    best_ckpt = Path(os.getcwd()) / "checkpoints" / "best.pth"
    if best_ckpt.exists():
        logging.info("Loading best checkpoint for final validation evaluation")
        trainer.load_checkpoint(str(best_ckpt))
        val_loss, val_acc = trainer.validate()
        logging.info(
            "Validation after training — loss: %.4f, acc: %.2f%%", val_loss, val_acc
        )

    # Visualizations: training curves and confusion matrix on validation set
    plots_dir = Path(os.getcwd()) / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_training_history(history, save_path=str(plots_dir / "training_curves.png"))

    # Confusion matrix using validation set
    assert val_loader is not None, (
        "Validation loader must not be None to plot confusion matrix"
    )
    y_true, y_pred = trainer.predict_labels(val_loader)
    cm = confusion_matrix(y_true, y_pred)
    class_names = val_dataset.get_class_names()
    plot_confusion_matrix(
        cm, class_names, save_path=str(plots_dir / "confusion_matrix.png")
    )


if __name__ == "__main__":
    main()  # type: ignore
