"""Verify the padding-strip fix produces discriminative embeddings."""
import sys
sys.path.insert(0, ".")

import torch
from src.model.backbone import TimesFMBackbone

backbone = TimesFMBackbone(
    repo_id="google/timesfm-2.5-200m-pytorch",
    context_len=512, horizon_len=128,
    device="cpu", extraction_mode="hook",
)

x1 = torch.randn(2, 96)
x2 = torch.randn(2, 96)
e1 = backbone(x1)
e2 = backbone(x2)

F = torch.nn.functional
print(f"Output shape: {e1.shape}")
print(f"Cosine sim (mean pooled): {F.cosine_similarity(e1.mean(1), e2.mean(1)).mean():.4f}")
print(f"Same input reproduces:    {torch.allclose(backbone(x1), e1)}")

# Full pipeline test
from src.model.tsc_model import TSCFoundationModel

model = TSCFoundationModel(
    backbone_repo="google/timesfm-2.5-200m-pytorch",
    pooling="attention", classifier_type="linear",
    num_classes=2, context_len=512, horizon_len=128,
    device="cpu", freeze_backbone=True, extraction_mode="hook",
)

logits = model(x1)
print(f"\nFull model: {x1.shape} -> {logits.shape}")
print("Done!")
backbone.cleanup()
