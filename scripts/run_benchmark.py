"""Benchmark TSC Foundation Model across multiple UCR datasets.

Runs training and evaluation for each specified dataset, collects results,
and outputs a summary table. Results are saved to CSV for paper inclusion.

Usage:
    python scripts/run_benchmark.py --datasets ECG200 GunPoint FordA Wafer
    python scripts/run_benchmark.py --datasets ECG200 GunPoint --config configs/finetune.yaml
"""

import argparse
import logging
import sys
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config, save_config
from src.data.dataset import UCRDataset
from src.model.tsc_model import TSCFoundationModel
from src.training.trainer import Trainer

# Popular UCR datasets for quick benchmarking
DEFAULT_DATASETS = [
    "ECG200",
    "GunPoint",
    "Wafer",
    "Coffee",
    "BeetleFly",
    "BirdChicken",
    "TwoLeadECG",
    "ItalyPowerDemand",
]


def run_single_dataset(dataset_name, config, output_base):
    """Train and evaluate on a single dataset. Returns metrics dict."""
    log = logging.getLogger(dataset_name)
    output_dir = output_base / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = config["data"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        # Load data
        train_ds = UCRDataset(
            name=dataset_name,
            split="train",
            normalize=config["data"]["normalize"],
            max_len=config["model"]["context_len"],
        )
        test_ds = UCRDataset(
            name=dataset_name,
            split="test",
            normalize=config["data"]["normalize"],
            max_len=config["model"]["context_len"],
        )

        bs = config["training"]["batch_size"]
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

        # Build model
        model = TSCFoundationModel(
            num_classes=train_ds.num_classes,
            backbone_repo=config["model"]["backbone"],
            context_len=config["model"]["context_len"],
            horizon_len=config["model"]["horizon_len"],
            pooling=config["model"]["pooling"],
            classifier_type=config["classifier"]["type"],
            classifier_hidden_dims=config["classifier"]["hidden_dims"],
            dropout=config["classifier"]["dropout"],
            freeze_backbone=config["model"]["freeze_backbone"],
            device=config["device"],
        )

        # Train
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            config=config["training"],
            device=config["device"],
            output_dir=str(output_dir),
        )

        results = trainer.train()
        results["dataset"] = dataset_name
        results["num_classes"] = train_ds.num_classes
        results["train_size"] = len(train_ds)
        results["test_size"] = len(test_ds)
        results["seq_len"] = train_ds.seq_len

        # Save per-dataset results
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        log.info(f"  {dataset_name}: Acc={results['accuracy']:.4f}, F1={results['f1_weighted']:.4f}")
        return results

    except Exception as e:
        log.error(f"  {dataset_name}: FAILED — {e}")
        return {"dataset": dataset_name, "accuracy": -1, "f1_weighted": -1, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Benchmark TSC Foundation Model")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="UCR dataset names (default: small selection)"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device

    datasets = args.datasets or DEFAULT_DATASETS
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    save_config(config, output_base / "config.yaml")

    print(f"\n{'='*70}")
    print(f"  TSC Foundation Model Benchmark")
    print(f"  Backbone: {config['model']['backbone']}")
    print(f"  Datasets: {len(datasets)}")
    print(f"{'='*70}\n")

    all_results = []
    for i, dataset_name in enumerate(datasets, 1):
        print(f"[{i}/{len(datasets)}] {dataset_name}")
        result = run_single_dataset(dataset_name, config, output_base)
        all_results.append(result)

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"{'Dataset':<22} {'Classes':>7} {'Train':>6} {'Test':>6} {'Len':>5} {'Accuracy':>9} {'F1 (w)':>9}")
    print("-" * 70)

    accs = []
    for r in all_results:
        if r.get("accuracy", -1) >= 0:
            print(
                f"{r['dataset']:<22} {r.get('num_classes','?'):>7} "
                f"{r.get('train_size','?'):>6} {r.get('test_size','?'):>6} "
                f"{r.get('seq_len','?'):>5} "
                f"{r['accuracy']:>9.4f} {r['f1_weighted']:>9.4f}"
            )
            accs.append(r["accuracy"])
        else:
            print(f"{r['dataset']:<22} {'FAILED':>50}")

    if accs:
        print("-" * 70)
        print(f"{'AVERAGE':<22} {'':>26} {np.mean(accs):>9.4f} {np.mean([r['f1_weighted'] for r in all_results if r.get('accuracy',-1) >= 0]):>9.4f}")
    print(f"{'='*70}\n")

    # Save CSV
    import csv
    csv_path = output_base / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "accuracy", "f1_weighted", "f1_macro",
                                                "num_classes", "train_size", "test_size", "seq_len"])
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
