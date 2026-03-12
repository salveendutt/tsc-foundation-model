"""Backbone models for time series embedding extraction.

Provides two backbone options:
1. TimesFMBackbone: Wraps Google's TimesFM 2.5 pretrained transformer
2. SimpleCNNBackbone: Lightweight 1D CNN baseline (no external dependencies)
"""

import math
import torch
import torch.nn as nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TimesFMBackbone(nn.Module):
    """Extract embeddings from a pretrained TimesFM 2.5 model.

    Calls the core transformer's forward() directly to extract
    output_embeddings, bypassing the forecast/decode pipeline entirely.
    This is faster and avoids numpy round-trips.

    Args:
        repo_id: HuggingFace repo ID for the TimesFM checkpoint.
        context_len: Maximum input context length.
        horizon_len: Forecast horizon (required for model loading).
        device: Target device ('cpu', 'cuda', 'mps').
    """

    def __init__(
        self,
        repo_id: str = "google/timesfm-2.5-200m-pytorch",
        context_len: int = 512,
        horizon_len: int = 128,
        device: str = "cpu",
    ):
        super().__init__()
        self.repo_id = repo_id
        self.context_len = context_len
        self.horizon_len = horizon_len
        self._device = device

        self._load_model()

    def _load_model(self):
        """Load the TimesFM checkpoint."""
        try:
            import timesfm
        except ImportError:
            raise ImportError(
                "timesfm is required for TimesFMBackbone. "
                "Install it with: pip install timesfm"
            )

        self._load_model_2p5(timesfm)

        # Find the core PyTorch model
        self._core_model = self._find_core_model()
        self.hidden_dim = self._find_hidden_dim()
        logger.info(f"TimesFM loaded: hidden_dim={self.hidden_dim}")

    def _load_model_2p5(self, timesfm):
        """Load a TimesFM 2.5 checkpoint using the new API."""
        if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            raise RuntimeError(
                "TimesFM 2.5 requires the latest timesfm package. "
                "Install from GitHub: pip install git+https://github.com/google-research/timesfm.git#egg=timesfm[torch]"
            )

        # Disable torch.compile on macOS / CPU — it's not supported
        use_compile = self._device not in ("cpu", "mps")

        try:
            self.tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.repo_id, torch_compile=use_compile,
            )
        except TypeError:
            # Workaround: newer huggingface_hub passes extra kwargs (e.g. proxies)
            # that the timesfm constructor doesn't accept. Load manually.
            from huggingface_hub import hf_hub_download

            self.tfm = timesfm.TimesFM_2p5_200M_torch(torch_compile=use_compile)
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename="model.safetensors",
            )
            self.tfm.model.load_checkpoint(model_path, torch_compile=use_compile)

        logger.info("Loaded TimesFM 2.5 via from_pretrained")

        # Move the internal PyTorch model to the target device (e.g. MPS).
        # TimesFM hardcodes device to cuda-or-cpu; override it.
        if self._device not in ("cpu",):
            target = torch.device(self._device)
            core = self.tfm.model  # TimesFM_2p5_200M_torch_module
            core.to(target)
            core.device = target
            logger.info(f"Moved TimesFM 2.5 core model to {self._device}")

    def _find_core_model(self) -> nn.Module:
        """Locate the internal PyTorch nn.Module."""
        for attr in ["_model", "model", "torch_model", "backbone"]:
            obj = getattr(self.tfm, attr, None)
            if isinstance(obj, nn.Module):
                return obj

        # Search all attributes
        for attr in dir(self.tfm):
            if attr.startswith("_"):
                continue
            try:
                obj = getattr(self.tfm, attr)
                if isinstance(obj, nn.Module):
                    return obj
            except Exception:
                continue

        raise RuntimeError(
            "Cannot locate PyTorch model inside TimesFM. "
            "Run `python scripts/inspect_model.py` to examine the structure."
        )

    def _find_hidden_dim(self) -> int:
        """Determine hidden dimension from model structure."""
        # Check common config attributes
        for obj in [self._core_model, self.tfm, getattr(self.tfm, "hparams", None)]:
            if obj is None:
                continue
            for attr in ["md", "model_dim", "d_model", "hidden_size", "embed_dim"]:
                val = getattr(obj, attr, None)
                if isinstance(val, int) and val > 0:
                    return val

        # Infer from LayerNorm parameters
        for name, module in self._core_model.named_modules():
            if isinstance(module, nn.LayerNorm) and len(module.normalized_shape) == 1:
                dim = module.normalized_shape[0]
                if dim >= 64:
                    logger.info(f"Inferred hidden_dim={dim} from LayerNorm '{name}'")
                    return dim

        logger.warning("Could not determine hidden_dim, defaulting to 1280")
        return 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from input time series.

        Calls the core transformer's forward() directly, replicating the
        input preprocessing pipeline (left-pad → global normalize → patch →
        RevIN) and grabbing the output_embeddings. This bypasses the
        autoregressive decoding loop and force_flip_invariance double pass.

        Args:
            x: [batch_size, seq_len] input time series.

        Returns:
            [batch_size, num_real_patches, hidden_dim] patch embeddings.
        """
        from timesfm.torch.util import revin, update_running_stats

        model = self._core_model
        device = next(model.parameters()).device
        patch_len = model.p  # 32

        batch_size, seq_len = x.shape
        x = x.to(device).float()

        # Left-pad to context_len (matching forecast() base class)
        context = self.context_len
        if seq_len >= context:
            x = x[:, -context:]
            masks = torch.zeros(batch_size, context, dtype=torch.bool, device=device)
        else:
            pad_len = context - seq_len
            x = torch.nn.functional.pad(x, (pad_len, 0), value=0.0)
            masks = torch.zeros(batch_size, context, dtype=torch.bool, device=device)
            masks[:, :pad_len] = True

        # Step 1: Global normalization (matching _compiled_decode's normalize_inputs)
        mu_global = torch.mean(x, dim=-1, keepdim=True)
        sigma_global = torch.std(x, dim=-1, keepdim=True)
        x = revin(x, mu_global, sigma_global, reverse=False)

        # Step 2: Patch into (B, num_patches, patch_len)
        num_patches = context // patch_len
        patched_inputs = x.reshape(batch_size, num_patches, patch_len)
        patched_masks = masks.reshape(batch_size, num_patches, patch_len)

        # Step 3: Per-patch running mean/sigma → RevIN (matching decode())
        n = torch.zeros(batch_size, device=device)
        mu = torch.zeros(batch_size, device=device)
        sigma = torch.zeros(batch_size, device=device)
        patch_mu, patch_sigma = [], []
        for i in range(num_patches):
            (n, mu, sigma), _ = update_running_stats(
                n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
            )
            patch_mu.append(mu)
            patch_sigma.append(sigma)
        context_mu = torch.stack(patch_mu, dim=1)
        context_sigma = torch.stack(patch_sigma, dim=1)

        normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
        normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

        # Step 4: Forward through the model — grab output_embeddings (index [1])
        with torch.no_grad():
            (_, output_embeddings, _, _), _ = model(normed_inputs, patched_masks)

        # Strip padded patches — keep only the real ones
        num_real_patches = math.ceil(seq_len / patch_len)
        if num_real_patches < output_embeddings.shape[1]:
            output_embeddings = output_embeddings[:, -num_real_patches:]

        return output_embeddings

    def _apply(self, fn):
        """Prevent moving TimesFM internals via model.to(device).

        The core model is already placed on the target device during
        _load_model_2p5(). The _apply override prevents Trainer's
        model.to(device) from interfering with TimesFM's internal state.
        """
        return self

    def freeze(self):
        """Freeze all backbone parameters."""
        if self._core_model is not None:
            for p in self._core_model.parameters():
                p.requires_grad = False
            logger.info("TimesFM backbone frozen")

    def unfreeze(self, num_layers: Optional[int] = None):
        """Unfreeze backbone parameters.

        Args:
            num_layers: If set, only unfreeze the last N transformer layers.
                        If None, unfreeze all parameters.
        """
        if self._core_model is None:
            return

        if num_layers is None:
            for p in self._core_model.parameters():
                p.requires_grad = True
            logger.info("TimesFM backbone fully unfrozen")
            return

        # Find transformer layer lists and unfreeze last N
        for name, module in self._core_model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 3:
                total = len(module)
                start = max(0, total - num_layers)
                for i, layer in enumerate(module):
                    for p in layer.parameters():
                        p.requires_grad = i >= start
                logger.info(f"Unfroze last {num_layers}/{total} transformer layers")
                return

        # Fallback: unfreeze everything
        self.unfreeze(None)


class SimpleCNNBackbone(nn.Module):
    """Simple 1D CNN backbone for baseline comparison.

    A lightweight feature extractor that doesn't require any pretrained
    checkpoint. Useful for testing the pipeline and as a baseline.

    Args:
        hidden_dim: Base number of channels (doubled each layer).
        num_layers: Number of convolutional blocks.
        kernel_size: Convolution kernel size.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        kernel_size: int = 7,
    ):
        super().__init__()
        layers = []
        in_ch = 1  # Univariate time series

        for i in range(num_layers):
            out_ch = hidden_dim * (2**i)
            padding = kernel_size // 2
            layers.extend(
                [
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                    nn.MaxPool1d(2),
                ]
            )
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.hidden_dim = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from time series.

        Args:
            x: [batch_size, seq_len] input time series.

        Returns:
            [batch_size, hidden_dim] feature vector.
        """
        x = x.unsqueeze(1)  # [B, 1, T]
        features = self.features(x)  # [B, C, T']
        pooled = self.pool(features).squeeze(-1)  # [B, C]
        return pooled

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self, num_layers=None):
        for p in self.parameters():
            p.requires_grad = True

    def cleanup(self):
        pass
