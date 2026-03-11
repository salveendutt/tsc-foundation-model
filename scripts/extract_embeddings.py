"""Extract embeddings from TimesFM and optionally visualize with t-SNE.

Useful for:
- Analyzing the quality of pretrained representations
- Visualizing class separability in embedding space
- Saving embeddings for use in other classifiers (e.g., sklearn)

Usage:
    python scripts/extract_embeddings.py --dataset ECG200 --visualize
    python scripts/extract_embeddings.py --dataset GunPoint --output-dir embeddings/
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
from src.model.backbone import TimesFMBackbone, SimpleCNNBackbone


def extract_embeddings(backbone, loader, device, pooling="mean"):
    """Extract embeddings for all samples in a dataloader."""
    backbone.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            emb = backbone(batch_x)

            # Pool if 3D (patch-level embeddings)
            if emb.dim() == 3:
                if pooling == "mean":
                    emb = emb.mean(dim=1)
                elif pooling == "max":
                    emb = emb.max(dim=1).values
                elif pooling == "last":
                    emb = emb[:, -1, :]

            all_embeddings.append(emb.cpu())
            all_labels.append(batch_y)

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = np.concatenate(all_labels)
    return embeddings, labels


def visualize_embeddings(embeddings, labels, title, save_path=None):
    """Visualize embeddings using t-SNE."""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    emb_np = embeddings.numpy()

    # Reduce to 2D with t-SNE
    perplexity = min(30, len(emb_np) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    emb_2d = tsne.fit_transform(emb_np)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=[colors[i]],
            label=f"Class {label}",
            alpha=0.7,
            s=30,
        )

    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Extract TimesFM embeddings")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="ECG200")
    parser.add_argument("--backbone", type=str, default=None,
                        help="'cnn' or HuggingFace repo ID")
    parser.add_argument("--output-dir", type=str, default="embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "last"])
    parser.add_argument("--visualize", action="store_true",
                        help="Generate t-SNE visualization")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = load_config(args.config)
    if args.backbone:
        config["model"]["backbone"] = args.backbone

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_ds = UCRDataset(args.dataset, split="train", normalize=True,
                          max_len=config["model"]["context_len"])
    test_ds = UCRDataset(args.dataset, split="test", normalize=True,
                         max_len=config["model"]["context_len"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Build backbone
    backbone_name = config["model"]["backbone"]
    if backbone_name == "cnn":
        backbone = SimpleCNNBackbone()
    else:
        backbone = TimesFMBackbone(
            repo_id=backbone_name,
            context_len=config["model"]["context_len"],
            horizon_len=config["model"]["horizon_len"],
            device=args.device,
        )
    backbone = backbone.to(args.device)

    # Extract
    print(f"Extracting embeddings for {args.dataset}...")
    train_emb, train_labels = extract_embeddings(
        backbone, train_loader, args.device, args.pooling
    )
    test_emb, test_labels = extract_embeddings(
        backbone, test_loader, args.device, args.pooling
    )

    print(f"  Train embeddings: {train_emb.shape}")
    print(f"  Test embeddings:  {test_emb.shape}")

    # Save
    np.savez(
        output_dir / f"{args.dataset}_embeddings.npz",
        train_embeddings=train_emb.numpy(),
        train_labels=train_labels,
        test_embeddings=test_emb.numpy(),
        test_labels=test_labels,
    )
    print(f"Saved to {output_dir / f'{args.dataset}_embeddings.npz'}")

    # Visualize
    if args.visualize:
        # Combine train + test for visualization
        all_emb = torch.cat([train_emb, test_emb], dim=0)
        all_labels = np.concatenate([train_labels, test_labels])

        title = f"{args.dataset} — {backbone_name} Embeddings (t-SNE)"
        save_path = output_dir / f"{args.dataset}_tsne.png"
        visualize_embeddings(all_emb, all_labels, title, save_path)

    # Quick linear probe with sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_emb.numpy(), train_labels)
    preds = clf.predict(test_emb.numpy())
    acc = accuracy_score(test_labels, preds)
    print(f"\n  sklearn LogisticRegression accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
