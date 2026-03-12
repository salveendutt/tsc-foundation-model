"""Debug hook capture for TimesFM 2.5."""
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
from src.model.backbone import TimesFMBackbone

backbone = TimesFMBackbone(
    repo_id="google/timesfm-2.5-200m-pytorch",
    context_len=512,
    horizon_len=128,
    device="cpu",
    extraction_mode="hook",
)

call_count = [0]
original_hook = backbone._hook_fn

def debug_hook(module, inp, output):
    call_count[0] += 1
    if isinstance(output, tuple):
        shapes = []
        for o in output:
            if isinstance(o, torch.Tensor):
                shapes.append(str(o.shape))
            elif isinstance(o, list):
                shapes.append(f"list[{len(o)}]")
            else:
                shapes.append(type(o).__name__)
        print(f"  Hook #{call_count[0]}: tuple({', '.join(shapes)})")
    elif isinstance(output, torch.Tensor):
        print(f"  Hook #{call_count[0]}: tensor {output.shape}")
    original_hook(module, inp, output)
    if "emb" in backbone._captured:
        print(f"    -> captured: {backbone._captured['emb'].shape}")

# Replace hook with debug version
for h in backbone._hooks:
    h.remove()
backbone._hooks.clear()

stacked_xf = backbone._core_model.stacked_xf
last_layer = stacked_xf[-1]
hook = last_layer.register_forward_hook(debug_hook)
backbone._hooks.append(hook)

print("Running forecast on batch=2, length=96...")
x = torch.randn(2, 96)
backbone._captured.clear()
call_count[0] = 0
emb = backbone(x)
print(f"\nTotal hook calls: {call_count[0]}")
print(f"Final output shape: {emb.shape}")
backbone.cleanup()
