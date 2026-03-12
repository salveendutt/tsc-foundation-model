"""Compare embedding quality across context lengths on FordA (seq_len=500)."""
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.data.dataset import UCRDataset
from src.model.backbone import TimesFMBackbone


def extract_all(backbone, dataset, max_samples=200):
    """Extract embeddings one sample at a time (safe for large context)."""
    embeddings, labels = [], []
    n = min(len(dataset), max_samples)
    with torch.no_grad():
        for i in range(n):
            x, y = dataset[i]
            emb = backbone(x.unsqueeze(0))  # [1, N, D]
            if emb.dim() == 3:
                emb = emb.mean(dim=1)
            embeddings.append(emb.squeeze(0).cpu().numpy())
            labels.append(y.item())
            if (i + 1) % 50 == 0:
                print(f"    extracted {i+1}/{n}")
    return np.stack(embeddings), np.array(labels)


def try_context(context_len, train_ds, test_ds, max_samples=200):
    print(f"\n  context_len={context_len}")
    backbone = TimesFMBackbone(
        repo_id="google/timesfm-2.5-200m-pytorch",
        context_len=context_len,
        horizon_len=128,
        device="cpu",
        extraction_mode="hook",
    )

    X_train, y_train = extract_all(backbone, train_ds, max_samples)
    X_test, y_test = extract_all(backbone, test_ds, max_samples)

    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"  Train emb std:  {X_train.std(axis=0).mean():.4f}")
    print(f"  Train emb mean: {X_train.mean(axis=0).mean():.4f}")

    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"  LogReg C={C:5.2f}: train={train_acc:.3f}  test={test_acc:.3f}")

    # Also check if embeddings are numerically identical to context_len=512
    backbone.cleanup()
    return X_train, X_test


# FordA: seq_len=500, so context_len must be >= 500
train_ds = UCRDataset("FordA", split="train", normalize=True)
test_ds = UCRDataset("FordA", split="test", normalize=True)
print(f"\n{'#'*60}")
print(f"# Dataset: FordA  (seq_len={train_ds.seq_len})")
print(f"# Using first 200 train / 200 test samples")
print(f"{'#'*60}")

results = {}
for ctx in [512, 1024]:
    X_tr, X_te = try_context(ctx, train_ds, test_ds, max_samples=200)
    results[ctx] = (X_tr, X_te)

# Check if embeddings differ between context lengths
if len(results) == 2:
    X512, _ = results[512]
    X1024, _ = results[1024]
    diff = np.abs(X512 - X1024).max()
    print(f"\n  Max abs difference between ctx=512 and ctx=1024: {diff:.6e}")
    print(f"  Embeddings identical: {diff < 1e-5}")
