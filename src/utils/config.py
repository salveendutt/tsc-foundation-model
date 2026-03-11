"""Configuration management.

Loads YAML config files and merges with sensible defaults.
CLI arguments override config file values.
"""

import yaml
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "model": {
        "backbone": "google/timesfm-1.0-200m-pytorch",
        "context_len": 512,
        "horizon_len": 128,
        "pooling": "mean",
        "freeze_backbone": True,
        "extraction_mode": "hook",
    },
    "classifier": {
        "type": "linear",
        "hidden_dims": [256],
        "dropout": 0.1,
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "backbone_lr": 1e-5,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "early_stopping_patience": 15,
    },
    "data": {
        "dataset": "ECG200",
        "normalize": True,
        "seed": 42,
    },
    "device": "cpu",
}


def load_config(path: str = None) -> dict:
    """Load configuration from YAML file, merged with defaults.

    Args:
        path: Path to YAML config file. If None, returns defaults.

    Returns:
        Complete configuration dictionary.
    """
    config = _deep_copy(DEFAULT_CONFIG)

    if path is not None:
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
            if user_config:
                config = _deep_merge(config, user_config)

    return config


def save_config(config: dict, path: str):
    """Save configuration to YAML file."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base dictionary."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _deep_copy(d: dict) -> dict:
    """Simple deep copy for nested dicts with basic types."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy(v)
        elif isinstance(v, list):
            result[k] = v.copy()
        else:
            result[k] = v
    return result
