"""Unit tests for the TSC Foundation Model pipeline.

Tests the pipeline components that don't require TimesFM to be installed.
Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.backbone import SimpleCNNBackbone
from src.model.classifier import LinearClassifier, MLPClassifier, AttentionPooling
from src.model.tsc_model import TSCFoundationModel
from src.data.dataset import EmbeddingDataset
from src.data.preprocessing import normalize_zscore, normalize_minmax, pad_or_truncate
from src.evaluation.metrics import compute_metrics
from src.utils.config import load_config, _deep_merge


# ── Preprocessing tests ──────────────────────────────────────


class TestPreprocessing:
    def test_zscore_normalization(self):
        X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        X_norm = normalize_zscore(X)
        np.testing.assert_almost_equal(X_norm.mean(axis=1), [0.0])
        np.testing.assert_almost_equal(X_norm.std(axis=1), [1.0])

    def test_zscore_constant_series(self):
        X = np.array([[5.0, 5.0, 5.0]])
        X_norm = normalize_zscore(X)
        np.testing.assert_array_equal(X_norm, [[0.0, 0.0, 0.0]])

    def test_minmax_normalization(self):
        X = np.array([[0.0, 5.0, 10.0]])
        X_norm = normalize_minmax(X)
        np.testing.assert_almost_equal(X_norm, [[0.0, 0.5, 1.0]])

    def test_pad_or_truncate(self):
        X = np.array([[1.0, 2.0, 3.0]])
        padded = pad_or_truncate(X, 5)
        assert padded.shape == (1, 5)
        np.testing.assert_array_equal(padded, [[1.0, 2.0, 3.0, 0.0, 0.0]])

        truncated = pad_or_truncate(X, 2)
        assert truncated.shape == (1, 2)
        np.testing.assert_array_equal(truncated, [[1.0, 2.0]])


# ── Model component tests ────────────────────────────────────


class TestCNNBackbone:
    def test_forward_shape(self):
        backbone = SimpleCNNBackbone(hidden_dim=32, num_layers=2)
        x = torch.randn(4, 96)  # batch=4, seq_len=96
        out = backbone(x)
        assert out.dim() == 2
        assert out.shape[0] == 4
        assert out.shape[1] == backbone.hidden_dim

    def test_freeze_unfreeze(self):
        backbone = SimpleCNNBackbone()
        backbone.freeze()
        assert all(not p.requires_grad for p in backbone.parameters())
        backbone.unfreeze()
        assert all(p.requires_grad for p in backbone.parameters())


class TestClassifiers:
    def test_linear_classifier(self):
        clf = LinearClassifier(128, 5)
        x = torch.randn(4, 128)
        out = clf(x)
        assert out.shape == (4, 5)

    def test_mlp_classifier(self):
        clf = MLPClassifier(128, 5, hidden_dims=[64, 32])
        x = torch.randn(4, 128)
        out = clf(x)
        assert out.shape == (4, 5)

    def test_attention_pooling(self):
        pool = AttentionPooling(64)
        x = torch.randn(4, 10, 64)  # batch=4, seq=10, dim=64
        out = pool(x)
        assert out.shape == (4, 64)


class TestTSCModel:
    def test_cnn_model_forward(self):
        model = TSCFoundationModel(
            num_classes=3,
            backbone_repo="cnn",
            freeze_backbone=False,
            device="cpu",
        )
        x = torch.randn(4, 96)
        logits = model(x)
        assert logits.shape == (4, 3)

    def test_cnn_model_embeddings(self):
        model = TSCFoundationModel(
            num_classes=3,
            backbone_repo="cnn",
            freeze_backbone=False,
            device="cpu",
        )
        x = torch.randn(4, 96)
        emb = model.get_embeddings(x)
        assert emb.dim() == 2
        assert emb.shape[0] == 4

    def test_cnn_model_backward(self):
        model = TSCFoundationModel(
            num_classes=2,
            backbone_repo="cnn",
            freeze_backbone=False,
            device="cpu",
        )
        x = torch.randn(4, 96)
        y = torch.tensor([0, 1, 0, 1])
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        # Check gradients exist
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad


# ── Dataset tests ─────────────────────────────────────────────


class TestEmbeddingDataset:
    def test_basic(self):
        emb = torch.randn(10, 64)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        ds = EmbeddingDataset(emb, labels)
        assert len(ds) == 10
        x, y = ds[0]
        assert x.shape == (64,)
        assert isinstance(y, torch.Tensor)


# ── Metrics tests ─────────────────────────────────────────────


class TestMetrics:
    def test_perfect_accuracy(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        metrics = compute_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_weighted"] == 1.0

    def test_random_accuracy(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.5


# ── Config tests ──────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        config = load_config()
        assert "model" in config
        assert "training" in config
        assert config["model"]["backbone"] == "google/timesfm-1.0-200m-pytorch"

    def test_deep_merge(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 5}
        merged = _deep_merge(base, override)
        assert merged["a"]["b"] == 10
        assert merged["a"]["c"] == 2
        assert merged["d"] == 3
        assert merged["e"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
