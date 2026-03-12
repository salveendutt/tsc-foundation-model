"""Training entry point for TSC Foundation Model.

Usage:
    # Default: TimesFM + linear probing on ECG200
    python scripts/train.py

    # CNN baseline (no TimesFM needed)
    python scripts/train.py --config configs/baseline_cnn.yaml

    # Fine-tuning
    python scripts/train.py --config configs/finetune.yaml --dataset FordA

    # Custom settings
    python scripts/train.py --dataset GunPoint --epochs 50 --lr 5e-4 --batch-size 16
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, save_config
from src.data.dataset import UCRDataset
from src.model.tsc_model import TSCFoundationModel
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train TSC Foundation Model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default=None, help="UCR dataset name")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Setup logging
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.output_dir) / "train.log", mode="w"),
        ],
    )
    log = logging.getLogger(__name__)

    # Load and override config
    config = load_config(args.config)
    if args.dataset:
        config["data"]["dataset"] = args.dataset
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.device:
        config["device"] = args.device
    if args.seed:
        config["data"]["seed"] = args.seed

    # Set random seed
    seed = config["data"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = config["device"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    save_config(config, output_dir / "config.yaml")

    log.info(f"Config: {config}")

    # ── Load data ──────────────────────────────────────────────
    dataset_name = config["data"]["dataset"]
    log.info(f"Loading dataset: {dataset_name}")

    train_dataset = UCRDataset(
        name=dataset_name,
        split="train",
        normalize=config["data"]["normalize"],
        max_len=config["model"]["context_len"],
    )
    test_dataset = UCRDataset(
        name=dataset_name,
        split="test",
        normalize=config["data"]["normalize"],
        max_len=config["model"]["context_len"],
    )

    log.info(f"Train: {train_dataset.get_info()}")
    log.info(f"Test:  {test_dataset.get_info()}")

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # ── Build model ────────────────────────────────────────────
    log.info("Building model...")
    model = TSCFoundationModel(
        num_classes=train_dataset.num_classes,
        backbone_repo=config["model"]["backbone"],
        context_len=config["model"]["context_len"],
        horizon_len=config["model"]["horizon_len"],
        pooling=config["model"]["pooling"],
        classifier_type=config["classifier"]["type"],
        classifier_hidden_dims=config["classifier"]["hidden_dims"],
        dropout=config["classifier"]["dropout"],
        freeze_backbone=config["model"]["freeze_backbone"],
        device=device,
    )

    # ── Train ──────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        config=config["training"],
        device=device,
        output_dir=args.output_dir,
    )

    results = trainer.train()
    trainer.save_history()

    # ── Report ─────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  Dataset:       {dataset_name}")
    print(f"  Backbone:      {config['model']['backbone']}")
    print(f"  Pooling:       {config['model']['pooling']}")
    print(f"  Freeze:        {config['model']['freeze_backbone']}")
    print(f"  Classifier:    {config['classifier']['type']}")
    print(f"  ─────────────────────────────────")
    print(f"  Test Accuracy: {results['accuracy']:.4f}")
    print(f"  Test F1 (w):   {results['f1_weighted']:.4f}")
    print(f"  Test F1 (m):   {results['f1_macro']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
