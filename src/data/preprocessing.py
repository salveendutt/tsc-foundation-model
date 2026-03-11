"""Preprocessing utilities for time series data."""

import numpy as np


def normalize_zscore(X: np.ndarray) -> np.ndarray:
    """Z-normalize each time series independently (zero mean, unit variance).

    Args:
        X: [N, T] array of time series.

    Returns:
        Normalized array of same shape.
    """
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def normalize_minmax(X: np.ndarray) -> np.ndarray:
    """Min-max normalize each time series to [0, 1].

    Args:
        X: [N, T] array of time series.

    Returns:
        Normalized array of same shape.
    """
    xmin = X.min(axis=-1, keepdims=True)
    xmax = X.max(axis=-1, keepdims=True)
    denom = xmax - xmin
    denom[denom == 0] = 1.0
    return (X - xmin) / denom


def pad_or_truncate(
    X: np.ndarray, target_len: int, pad_value: float = 0.0
) -> np.ndarray:
    """Pad (right) or truncate time series to a fixed target length.

    Args:
        X: [N, T] array of time series.
        target_len: Desired sequence length.
        pad_value: Value to use for padding.

    Returns:
        [N, target_len] array.
    """
    current_len = X.shape[-1]

    if current_len == target_len:
        return X
    elif current_len > target_len:
        return X[..., :target_len]
    else:
        pad_width = [(0, 0)] * (X.ndim - 1) + [(0, target_len - current_len)]
        return np.pad(X, pad_width, constant_values=pad_value)
