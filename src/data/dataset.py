"""Dataset classes for time series classification.

Provides:
- UCRDataset: Loads datasets from the UCR Time Series Archive via aeon.
- EmbeddingDataset: Wraps pre-extracted embeddings for efficient training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class UCRDataset(Dataset):
    """PyTorch Dataset for UCR Time Series Archive datasets.

    Uses the `aeon` library to download and load datasets automatically.
    Handles z-normalization, NaN values, and label encoding.

    Args:
        name: UCR dataset name (e.g., 'ECG200', 'GunPoint', 'FordA').
        split: 'train' or 'test'.
        normalize: Whether to z-normalize each time series.
        max_len: Maximum sequence length (truncates if longer).
    """

    def __init__(
        self,
        name: str,
        split: str = "train",
        normalize: bool = True,
        max_len: Optional[int] = None,
    ):
        super().__init__()
        self.name = name
        self.split = split
        self.normalize = normalize
        self.max_len = max_len

        self.X, self.y, self.label_map = self._load_data()
        self.num_classes = len(self.label_map)
        self.seq_len = self.X.shape[1]

        logger.info(
            f"UCRDataset '{name}' ({split}): "
            f"{len(self)} samples, seq_len={self.seq_len}, "
            f"{self.num_classes} classes"
        )

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Load and preprocess the dataset."""
        from aeon.datasets import load_classification

        X, y = load_classification(self.name, split=self.split)

        # X: (n_instances, n_channels, n_timepoints) from aeon
        # Squeeze univariate to (n_instances, n_timepoints)
        if X.shape[1] == 1:
            X = X.squeeze(1)
        else:
            # For multivariate: concatenate channels
            n, c, t = X.shape
            X = X.reshape(n, c * t)

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        # Truncate to max_len if needed
        if self.max_len and X.shape[1] > self.max_len:
            X = X[:, : self.max_len]

        # Encode string labels to integers
        unique_labels = sorted(set(y))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_int = np.array([label_map[label] for label in y])

        # Z-normalize each time series
        if self.normalize:
            mean = X.mean(axis=1, keepdims=True)
            std = X.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            X = (X - mean) / std

        return X.astype(np.float32), y_int, label_map

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return torch.from_numpy(self.X[idx]), self.y[idx]

    def get_info(self) -> dict:
        """Return dataset statistics."""
        return {
            "name": self.name,
            "split": self.split,
            "num_samples": len(self),
            "seq_len": self.seq_len,
            "num_classes": self.num_classes,
            "label_map": self.label_map,
            "class_distribution": {
                str(label): int((self.y == idx).sum())
                for label, idx in self.label_map.items()
            },
        }


class EmbeddingDataset(Dataset):
    """Dataset of pre-extracted embeddings for efficient linear probing.

    When the backbone is frozen, we can extract all embeddings once and
    train the classifier on the cached embeddings. This avoids redundant
    forward passes through the backbone each epoch.

    Args:
        embeddings: [N, D] tensor of pre-extracted embeddings.
        labels: [N] array of integer labels.
    """

    def __init__(self, embeddings: torch.Tensor, labels: np.ndarray):
        self.embeddings = embeddings
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]
