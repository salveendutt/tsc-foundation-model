"""Evaluate a trained TSC model on a UCR dataset.

Usage:
    python scripts/evaluate.py --checkpoint outputs/best_model.pt --dataset ECG200
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.data.dataset import UCRDataset
from src.model.tsc_model import TSCFoundationModel
from src.evaluation.metrics import compute_metrics, detailed_report, compute_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate TSC Foundation Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config (from checkpoint directory or CLI)
    checkpoint_dir = Path(args.checkpoint).parent
    config_path = args.config or checkpoint_dir / "config.yaml"
    config = load_config(str(config_path) if Path(config_path).exists() else None)

    if args.dataset:
        config["data"]["dataset"] = args.dataset

    dataset_name = config["data"]["dataset"]

    # Load test data
    test_dataset = UCRDataset(
        name=dataset_name,
        split="test",
        normalize=config["data"]["normalize"],
        max_len=config["model"]["context_len"],
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Build model
    model = TSCFoundationModel(
        num_classes=test_dataset.num_classes,
        backbone_repo=config["model"]["backbone"],
        context_len=config["model"]["context_len"],
        horizon_len=config["model"]["horizon_len"],
        pooling=config["model"]["pooling"],
        classifier_type=config["classifier"]["type"],
        classifier_hidden_dims=config["classifier"]["hidden_dims"],
        dropout=config["classifier"]["dropout"],
        freeze_backbone=True,
        device=args.device,
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model_dict = model.state_dict()
    saved_dict = {
        k: v for k, v in checkpoint["model_state_dict"].items() if k in model_dict
    }
    model_dict.update(saved_dict)
    model.load_state_dict(model_dict, strict=False)
    model = model.to(args.device)
    model.eval()

    # Evaluate
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(args.device)
            logits = model(batch_x)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Metrics
    metrics = compute_metrics(all_labels, all_preds)
    label_names = [str(k) for k in sorted(test_dataset.label_map.keys())]
    report = detailed_report(all_labels, all_preds, label_names)
    cm = compute_confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*55}")
    print(f"  Evaluation Results: {dataset_name}")
    print(f"{'='*55}")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  F1 (weighted):  {metrics['f1_weighted']:.4f}")
    print(f"  F1 (macro):     {metrics['f1_macro']:.4f}")
    print(f"  Precision (w):  {metrics['precision_weighted']:.4f}")
    print(f"  Recall (w):     {metrics['recall_weighted']:.4f}")
    print(f"\n  Classification Report:")
    print(report)
    print(f"  Confusion Matrix:")
    print(cm)
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
