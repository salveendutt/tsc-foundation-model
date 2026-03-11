"""Combined TSC Foundation Model.

Assembles backbone + pooling + classification head into a single model
for end-to-end time series classification.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

from .backbone import TimesFMBackbone, SimpleCNNBackbone
from .classifier import LinearClassifier, MLPClassifier, AttentionPooling

logger = logging.getLogger(__name__)


class TSCFoundationModel(nn.Module):
    """Foundation model for time series classification.

    Architecture:
        Input TS [B, T] → Backbone → [B, N, D] → Pooling → [B, D] → Head → [B, C]

    Args:
        num_classes: Number of output classes.
        backbone_repo: HuggingFace repo ID for TimesFM, or 'cnn' for baseline.
        context_len: Maximum input context length.
        horizon_len: Forecast horizon (TimesFM parameter).
        pooling: Pooling strategy ('mean', 'max', 'last', 'attention', 'none').
        classifier_type: 'linear' or 'mlp'.
        classifier_hidden_dims: Hidden layer sizes for MLP classifier.
        dropout: Dropout rate for classifier.
        freeze_backbone: Whether to freeze the backbone weights.
        device: Target device.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_repo: str = "google/timesfm-1.0-200m-pytorch",
        context_len: int = 512,
        horizon_len: int = 128,
        pooling: str = "mean",
        classifier_type: str = "linear",
        classifier_hidden_dims: list = None,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [256]

        self.num_classes = num_classes
        self._pooling_type = pooling

        # Build backbone
        if backbone_repo == "cnn":
            self.backbone = SimpleCNNBackbone()
            if freeze_backbone:
                self.backbone.freeze()
        else:
            self.backbone = TimesFMBackbone(
                repo_id=backbone_repo,
                context_len=context_len,
                horizon_len=horizon_len,
                device=device,
            )
            if freeze_backbone:
                self.backbone.freeze()

        # Determine feature dimension
        feature_dim = self.backbone.hidden_dim
        if hasattr(self.backbone, "extraction_mode"):
            if self.backbone.extraction_mode == "forecast":
                feature_dim = horizon_len
                self._pooling_type = "none"
            elif self.backbone.extraction_mode == "direct":
                # CNN backbone outputs [B, D] directly, no pooling needed
                self._pooling_type = "none"

        self.feature_dim = feature_dim

        # Attention pooling layer (if selected)
        self.attention_pool = None
        if self._pooling_type == "attention":
            self.attention_pool = AttentionPooling(feature_dim)

        # Classification head
        if classifier_type == "linear":
            self.classifier = LinearClassifier(feature_dim, num_classes, dropout)
        elif classifier_type == "mlp":
            self.classifier = MLPClassifier(
                feature_dim, num_classes, classifier_hidden_dims, dropout
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"TSCFoundationModel: {num_classes} classes, "
            f"feature_dim={feature_dim}, pooling={self._pooling_type}, "
            f"trainable={trainable:,}/{total:,} parameters"
        )

    def pool_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Pool variable-length embeddings to a fixed-size representation.

        Args:
            embeddings: [B, N, D] patch embeddings or [B, D] already pooled.

        Returns:
            [B, D] fixed-size representation.
        """
        if self._pooling_type == "none" or embeddings.dim() == 2:
            return embeddings

        if self._pooling_type == "mean":
            return embeddings.mean(dim=1)
        elif self._pooling_type == "max":
            return embeddings.max(dim=1).values
        elif self._pooling_type == "last":
            return embeddings[:, -1, :]
        elif self._pooling_type == "attention":
            return self.attention_pool(embeddings)
        else:
            return embeddings.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: extract embeddings, pool, classify.

        Args:
            x: [batch_size, seq_len] input time series.

        Returns:
            [batch_size, num_classes] class logits.
        """
        embeddings = self.backbone(x)
        pooled = self.pool_embeddings(embeddings)
        logits = self.classifier(pooled)
        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled embeddings without classification.

        Useful for visualization, analysis, and downstream tasks.
        """
        embeddings = self.backbone(x)
        return self.pool_embeddings(embeddings)
