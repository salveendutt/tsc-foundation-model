"""Training loop for TSC Foundation Model.

Supports:
- Linear probing with pre-computed embeddings (efficient, backbone frozen)
- End-to-end training with full forward passes (fine-tuning)
- Early stopping, gradient clipping, LR scheduling
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from ..evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Train and evaluate the TSC model.

    For frozen backbones (linear probing), embeddings are pre-extracted once
    and cached, making training very fast even with large foundation models.

    Args:
        model: The TSCFoundationModel instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation/test data.
        config: Training configuration dictionary.
        device: Target device.
        output_dir: Directory for saving checkpoints and results.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cpu",
        output_dir: str = "outputs",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer with differential learning rates."""
        lr = self.config.get("learning_rate", 1e-3)
        weight_decay = self.config.get("weight_decay", 1e-4)
        backbone_lr = self.config.get("backbone_lr", 1e-5)

        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": lr})

        if not param_groups:
            logger.warning("No trainable parameters found!")
            param_groups = [{"params": self.model.parameters(), "lr": lr}]

        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        sched_type = self.config.get("scheduler", "cosine")
        epochs = self.config.get("epochs", 100)

        if sched_type == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=epochs)
        return None

    def train_epoch(self) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).long()

            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate on a dataset. Returns metrics dictionary."""
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).long()

            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)

            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

            total_loss += loss.item()
            num_batches += 1

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        metrics = compute_metrics(all_labels, all_preds)
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    def train(self) -> Dict[str, float]:
        """Full training loop with early stopping.

        Returns:
            Final evaluation metrics on the validation set.
        """
        epochs = self.config.get("epochs", 100)
        patience = self.config.get("early_stopping_patience", 15)

        logger.info(f"Training for up to {epochs} epochs (patience={patience})")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable:,}")

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss = self.train_epoch()
            val_metrics = self.evaluate()

            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1_weighted"])

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:3d}/{epochs} ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1_weighted']:.4f}"
            )

            # Early stopping
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        # Load best checkpoint and do final evaluation
        self._load_best_checkpoint()
        final_metrics = self.evaluate()
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        return final_metrics

    def _save_checkpoint(self, epoch: int, metrics: dict):
        """Save model checkpoint (classifier head only for frozen backbone)."""
        path = self.output_dir / "best_model.pt"

        # Save only the trainable parameters (skip frozen TimesFM weights)
        state_dict = {}
        for k, v in self.model.state_dict().items():
            # Skip the heavy TimesFM internals
            if "backbone.tfm" not in k and "backbone._core" not in k:
                state_dict[k] = v

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "best_val_acc": self.best_val_acc,
            },
            path,
        )

    def _load_best_checkpoint(self):
        """Load the best model checkpoint."""
        path = self.output_dir / "best_model.pt"
        if not path.exists():
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model_dict = self.model.state_dict()
        saved_dict = {
            k: v for k, v in checkpoint["model_state_dict"].items() if k in model_dict
        }
        model_dict.update(saved_dict)
        self.model.load_state_dict(model_dict, strict=False)
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

    def save_history(self, path: Optional[str] = None):
        """Save training history to a JSON file."""
        import json

        if path is None:
            path = self.output_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
