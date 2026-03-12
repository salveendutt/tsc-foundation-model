"""Inspect the TimesFM model structure.

Prints the full module tree, parameter counts, and hook targets.
Run this to verify the model structure before training.

Usage:
    python scripts/inspect_model.py
    python scripts/inspect_model.py --repo google/timesfm-1.0-200m-pytorch
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Inspect TimesFM model structure")
    parser.add_argument(
        "--repo",
        type=str,
        default="google/timesfm-2.5-200m-pytorch",
        help="HuggingFace repo ID",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"Loading TimesFM from: {args.repo}")
    print("This may take a moment on first run (downloads checkpoint)...\n")

    from src.model.backbone import TimesFMBackbone

    backbone = TimesFMBackbone(
        repo_id=args.repo,
        context_len=512,
        horizon_len=128,
        device="mps" if torch.backends.mps.is_available() else "cpu",
    )

    backbone.print_structure()

    # Test with dummy data
    import torch
    import numpy as np

    print("\nTesting embedding extraction with dummy data...")
    dummy = torch.randn(2, 96)  # 2 samples, length 96
    try:
        emb = backbone(dummy)
        print(f"  Input shape:     {dummy.shape}")
        print(f"  Embedding shape: {emb.shape}")
        print(f"  Embedding dtype: {emb.dtype}")
        print(f"  Extraction mode: {backbone.extraction_mode}")
        print("\n✓ Embedding extraction works!")
    except Exception as e:
        print(f"\n✗ Embedding extraction failed: {e}")
        print("  You may need to adjust the hook targets in backbone.py")

    backbone.cleanup()


if __name__ == "__main__":
    main()
