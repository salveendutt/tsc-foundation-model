"""Classification heads for time series classification.

Provides:
- AttentionPooling: Learnable attention-weighted pooling over sequence dim.
- LinearClassifier: Single linear layer with LayerNorm (for linear probing).
- MLPClassifier: Multi-layer perceptron with GELU and dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Learnable attention-weighted pooling over the sequence dimension.

    Computes a weighted average of token embeddings where the weights
    are learned via a single-head attention mechanism.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool sequence embeddings.

        Args:
            x: [batch_size, seq_len, hidden_dim] token embeddings.

        Returns:
            [batch_size, hidden_dim] pooled representation.
        """
        scores = self.attention(x).squeeze(-1)  # [B, T]
        weights = F.softmax(scores, dim=-1)  # [B, T]
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # [B, D]


class LinearClassifier(nn.Module):
    """Linear classification head with LayerNorm.

    Suitable for linear probing experiments where the backbone is frozen.
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MLPClassifier(nn.Module):
    """Multi-layer perceptron classification head.

    Uses GELU activation and dropout for regularization.
    Suitable for more expressive classification when fine-tuning.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256]

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend(
                [
                    nn.LayerNorm(prev_dim),
                    nn.Linear(prev_dim, dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
