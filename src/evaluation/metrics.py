"""Evaluation metrics for time series classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from typing import Optional, List


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary of metric name → value.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def detailed_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
) -> str:
    """Generate a detailed classification report string.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        label_names: Optional list of class names.

    Returns:
        Formatted classification report.
    """
    return classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0
    )


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Confusion matrix as numpy array.
    """
    return confusion_matrix(y_true, y_pred)
