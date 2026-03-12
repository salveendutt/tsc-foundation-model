"""Check embedding quality: padded vs real patches."""
import sys
sys.path.insert(0, ".")

import torch
import math
from src.model.backbone import TimesFMBackbone

backbone = TimesFMBackbone(
    repo_id="google/timesfm-2.5-200m-pytorch",
    context_len=512, horizon_len=128,
    device="cpu", extraction_mode="hook",
)

patch_len = backbone._core_model.p  # 32
print(f"patch_len: {patch_len}")
print(f"max_context: {backbone.context_len}")

x1 = torch.randn(2, 96)
x2 = torch.randn(2, 96)
e1 = backbone(x1)  # [2, 16, 1280]
e2 = backbone(x2)

num_real_patches = math.ceil(96 / patch_len)  # 3
num_total = e1.shape[1]  # 16
print(f"Total patches: {num_total}, Real patches: {num_real_patches}")
print(f"Padded patches at front: {num_total - num_real_patches}")

# Compare using only real patches (last N)
real_e1 = e1[:, -num_real_patches:]
real_e2 = e2[:, -num_real_patches:]
F = torch.nn.functional
print(f"\nCosine sim (ALL patches mean): {F.cosine_similarity(e1.mean(1), e2.mean(1)).mean():.4f}")
print(f"Cosine sim (REAL patches mean): {F.cosine_similarity(real_e1.mean(1), real_e2.mean(1)).mean():.4f}")

# Per-sample variance
print(f"\nPadded patch std (mean across dims): {e1[:, :-num_real_patches].std(0).mean():.6f}")
print(f"Real patch std (mean across dims):   {e1[:, -num_real_patches:].std(0).mean():.6f}")

# Are padded patches identical across samples?
print(f"\nPadded patches diff across samples: {(e1[:, :-num_real_patches] - e2[:, :-num_real_patches]).abs().mean():.6f}")
print(f"Real patches diff across samples:   {(e1[:, -num_real_patches:] - e2[:, -num_real_patches:]).abs().mean():.6f}")

backbone.cleanup()
