"""Backbone models for time series embedding extraction.

Provides two backbone options:
1. TimesFMBackbone: Wraps Google's TimesFM pretrained transformer
2. SimpleCNNBackbone: Lightweight 1D CNN baseline (no external dependencies)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TimesFMBackbone(nn.Module):
    """Extract embeddings from a pretrained TimesFM model.

    Uses forward hooks on the transformer layers to capture intermediate
    representations. Falls back to using forecast outputs as features if
    hook-based extraction fails.

    Args:
        repo_id: HuggingFace repo ID for the TimesFM checkpoint.
        context_len: Maximum input context length.
        horizon_len: Forecast horizon (required for model loading).
        device: Target device ('cpu', 'cuda', 'mps').
        extraction_mode: 'hook' for hidden states, 'forecast' for forecast features.
    """

    def __init__(
        self,
        repo_id: str = "google/timesfm-1.0-200m-pytorch",
        context_len: int = 512,
        horizon_len: int = 128,
        device: str = "cpu",
        extraction_mode: str = "hook",
    ):
        super().__init__()
        self.repo_id = repo_id
        self.context_len = context_len
        self.horizon_len = horizon_len
        self._device = device
        self.extraction_mode = extraction_mode

        self._load_model()

        if extraction_mode == "hook":
            self._captured = {}
            self._hooks = []
            self._setup_hooks()

    def _load_model(self):
        """Load the TimesFM checkpoint."""
        try:
            import timesfm
        except ImportError:
            raise ImportError(
                "timesfm is required for TimesFMBackbone. "
                "Install it with: pip install timesfm"
            )

        backend = "cpu" if self._device in ("cpu", "mps") else "gpu"

        # Try the newer API first (timesfm >= 1.2)
        try:
            self.tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend=backend,
                    per_core_batch_size=32,
                    horizon_len=self.horizon_len,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.repo_id,
                ),
            )
        except (TypeError, AttributeError):
            # Older API fallback
            try:
                self.tfm = timesfm.TimesFm(
                    context_len=self.context_len,
                    horizon_len=self.horizon_len,
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=20,
                    model_dims=1280,
                    backend=backend,
                )
                self.tfm.load_from_checkpoint(repo_id=self.repo_id)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load TimesFM with both API versions. "
                    f"Please check your timesfm package version. Error: {e}"
                )

        # Find the core PyTorch model
        self._core_model = self._find_core_model()
        self.hidden_dim = self._find_hidden_dim()
        logger.info(f"TimesFM loaded: hidden_dim={self.hidden_dim}")

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
            for attr in ["model_dim", "d_model", "hidden_size", "embed_dim"]:
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

    def _setup_hooks(self):
        """Register forward hooks on transformer layers."""
        if self._core_model is None:
            logger.warning("No core model found, falling back to forecast mode")
            self.extraction_mode = "forecast"
            return

        target = None

        # Priority 1: The StackedDecoder / transformer container module itself
        # (captures the full transformer output = final hidden states)
        for name, module in self._core_model.named_modules():
            name_l = name.lower()
            type_l = type(module).__name__.lower()
            if (
                ("stacked" in name_l or "stacked" in type_l)
                and ("transformer" in name_l or "decoder" in type_l or "encoder" in type_l)
                and not isinstance(module, nn.ModuleList)
            ):
                target = (name, module)
                break

        # Priority 2: Any module whose type contains 'decoder' or 'encoder' (not leaf layers)
        if target is None:
            for name, module in self._core_model.named_modules():
                type_l = type(module).__name__.lower()
                if ("decoder" in type_l or "encoder" in type_l) and list(module.children()):
                    # Skip individual decoder layers, prefer container modules
                    if "layer" not in name.split(".")[-1]:
                        target = (name, module)
                        break

        # Priority 3: Final LayerNorm / RMSNorm (exists right before output projection)
        if target is None:
            norms = [
                (n, m)
                for n, m in self._core_model.named_modules()
                if isinstance(m, (nn.LayerNorm,)) or "rmsnorm" in type(m).__name__.lower()
            ]
            if norms:
                target = norms[-1]

        if target:
            name, module = target
            hook = module.register_forward_hook(self._hook_fn)
            self._hooks.append(hook)
            logger.info(f"Embedding hook registered on: {name} ({type(module).__name__})")
        else:
            logger.warning(
                "No suitable module found for hooks, falling back to forecast mode"
            )
            self.extraction_mode = "forecast"

    def _hook_fn(self, module, input, output):
        """Capture module output during forward pass.

        Handles various output formats from different model architectures.
        Prefers 3D tensors [batch, seq, hidden] over 4D attention weights.
        """
        if isinstance(output, tuple):
            # Pick the best tensor: prefer 3D [batch, seq, hidden_dim] over
            # 4D [batch, heads, seq, seq] (attention weights)
            best = None
            for item in output:
                if isinstance(item, torch.Tensor):
                    if item.dim() == 3 and item.shape[-1] == self.hidden_dim:
                        best = item
                        break
                    elif item.dim() == 3 and best is None:
                        best = item
                    elif item.dim() == 2 and best is None:
                        best = item
            if best is not None:
                self._captured["emb"] = best
            else:
                # Fallback: take first tensor
                for item in output:
                    if isinstance(item, torch.Tensor):
                        self._captured["emb"] = item
                        break
        elif isinstance(output, dict):
            for key in ["last_hidden_state", "hidden_states", "output"]:
                if key in output and isinstance(output[key], torch.Tensor):
                    self._captured["emb"] = output[key]
                    return
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    self._captured["emb"] = v
                    return
        elif isinstance(output, torch.Tensor):
            self._captured["emb"] = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from input time series.

        Args:
            x: [batch_size, seq_len] input time series.

        Returns:
            Hook mode: [batch_size, num_tokens, hidden_dim] patch embeddings.
            Forecast mode: [batch_size, horizon_len] forecast features.
        """
        x_np = x.detach().cpu().numpy()

        if self.extraction_mode == "hook":
            return self._extract_via_hooks(x_np, x.device)
        else:
            return self._extract_via_forecast(x_np, x.device)

    def _extract_via_hooks(self, x_np: np.ndarray, device) -> torch.Tensor:
        """Extract embeddings by running forecast and capturing hook outputs."""
        self._captured.clear()
        batch_size = x_np.shape[0]
        freq = [0] * batch_size

        _ = self.tfm.forecast(list(x_np), freq)

        if "emb" in self._captured:
            emb = self._captured["emb"]
            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb).float()
            elif not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)

            # TimesFM pads batch internally to per_core_batch_size;
            # slice to keep only our actual samples
            if emb.shape[0] > batch_size:
                emb = emb[:batch_size]

            return emb.to(device)

        # Fallback to forecast mode if hooks didn't capture anything
        logger.warning("Hooks did not capture embeddings, falling back to forecast mode")
        return self._extract_via_forecast(x_np, device)

    def _extract_via_forecast(self, x_np: np.ndarray, device) -> torch.Tensor:
        """Use forecast outputs as features."""
        batch_size = x_np.shape[0]
        freq = [0] * batch_size

        result = self.tfm.forecast(list(x_np), freq)

        # forecast() returns (point_forecasts, quantile_forecasts) or just forecasts
        if isinstance(result, tuple):
            forecasts = result[0]
        else:
            forecasts = result

        forecasts = np.array(forecasts)
        return torch.from_numpy(forecasts).float().to(device)

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

    def cleanup(self):
        """Remove registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def print_structure(self):
        """Print model structure for debugging."""
        print(f"\n{'='*60}")
        print("TimesFM Model Structure")
        print(f"{'='*60}")
        print(f"Core model type: {type(self._core_model).__name__}")
        print(f"Hidden dim: {self.hidden_dim}")
        print(f"Extraction mode: {self.extraction_mode}")
        print(f"\nModules:")
        for name, module in self._core_model.named_modules():
            if name:
                depth = name.count(".")
                prefix = "  " * depth + "├─ "
                params = sum(p.numel() for p in module.parameters(recurse=False))
                param_str = f" ({params:,} params)" if params > 0 else ""
                print(f"{prefix}{name}: {type(module).__name__}{param_str}")
        total = sum(p.numel() for p in self._core_model.parameters())
        print(f"\nTotal parameters: {total:,}")
        print(f"{'='*60}\n")


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
        self.extraction_mode = "direct"

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
