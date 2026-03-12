# TSC Foundation Model

**Adapting Time Series Foundation Models for Classification**

> Leveraging Google's TimesFM pretrained representations for time series classification (TSC) tasks.

---

## Research Motivation

Foundation models have revolutionized NLP (BERT, GPT) and computer vision (CLIP, DINOv2) through transfer learning — pretrain on large-scale data, then adapt to downstream tasks. The same paradigm is emerging for time series:

- **Google's TimesFM** (2024) is a decoder-only transformer pretrained on 100B+ time points for forecasting
- Its learned representations capture rich temporal patterns: trends, seasonality, level shifts, and complex dynamics
- **Key question**: Can these forecasting-oriented representations transfer to *classification* tasks?

This project adapts TimesFM as a feature extractor for time series classification, evaluated on standard UCR/UEA benchmarks.

## Method

### Architecture

```
Input Time Series [B, T]
        │
        ▼
┌──────────────────────┐
│   Preprocessing      │  z-normalization, pad/truncate to context_len
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  TimesFM Backbone    │  Pretrained 200M-param transformer (frozen or fine-tuned)
│  (Patched Decoder)   │  Patches input → transformer layers → hidden states
└──────────┬───────────┘
           │  [B, num_patches, 1280]
           ▼
┌──────────────────────┐
│     Pooling          │  mean / max / last / attention pooling
└──────────┬───────────┘
           │  [B, 1280]
           ▼
┌──────────────────────┐
│  Classification Head │  Linear probe or MLP with dropout
└──────────┬───────────┘
           │  [B, num_classes]
           ▼
       Predictions
```

### Embedding Extraction: Direct Forward Pass

TimesFM was designed **only for forecasting** — its public API is `forecast()`, which takes a time series and returns predicted future values. It has no built-in way to output internal representations (embeddings). But classification needs those internal representations, not forecasts.

**The problem**: We need the rich hidden states that TimesFM computes *internally* during a forward pass, but the `forecast()` API discards them.

**The solution**: Call the **core transformer's `forward()` directly**, bypassing the autoregressive decoding loop, `force_flip_invariance` double pass, and numpy round-trips. We replicate TimesFM's input preprocessing pipeline and grab `output_embeddings` straight from the model.

Concretely, here's what happens:

```
1. Left-pad the input time series to context_len (matching forecast()'s behavior)
2. Global mean/std normalization
3. Patch the input into fixed-size chunks (patch_len=32)
4. Per-patch running-stats → RevIN normalization
5. Forward through all 20 transformer layers
6. Extract output_embeddings: [B, num_real_patches, 1280]
7. Strip padded patches, keeping only those corresponding to real input data
```

This is significantly faster than calling `forecast()` and intercepting hidden states via hooks, because it skips the autoregressive decoding loop entirely.

### Pooling Strategies

When using the TimesFM backbone, the model outputs a 3D tensor `[B, num_patches, 1280]` — one 1280-dim vector per input patch. The classifier needs a single fixed-size vector per sample. Pooling collapses the patch dimension:

| Strategy | Config Value | Operation | When to Use |
|---|---|---|---|
| **Mean Pooling** | `"mean"` | Average across all patches | Default. Treats all patches equally. Good general-purpose choice. |
| **Max Pooling** | `"max"` | Element-wise maximum across patches | Emphasizes the most activated features. Good when discriminative info is localized. |
| **Last Token** | `"last"` | Takes the final patch's embedding | Mirrors how decoder-only models (like GPT) work — the last token attends to all previous tokens. |
| **Attention Pooling** | `"attention"` | Learnable weighted average (trained end-to-end) | Learns which patches matter most for classification. Has extra trainable parameters. |
| **None** | `"none"` | No pooling (used for CNN backbone) | When backbone already outputs `[B, D]`. |

### Classification Heads

After pooling, the fixed-size embedding `[B, D]` is fed to a classification head:

| Head | Config Value | Architecture | Trainable Params (D=1280, C=2) | When to Use |
|---|---|---|---|---|
| **Linear** | `"linear"` | LayerNorm → Dropout → Linear(D, C) | ~2,562 | Linear probing (frozen backbone). Tests if embeddings are linearly separable. |
| **MLP** | `"mlp"` | LayerNorm → Linear(D, 256) → GELU → Dropout → Linear(256, C) | ~328,450 | When you want a nonlinear decision boundary. More expressive. |
| **sklearn LogReg** | *(via `extract_embeddings.py`)* | L2-regularized Logistic Regression (L-BFGS solver) | N/A | Quick evaluation without training a neural net. Often best for small datasets. |

### Training Strategies

| Strategy | Description | When to Use |
|---|---|---|
| **Linear Probing** | Freeze TimesFM, train only classification head | Small datasets, quick baseline |
| **Fine-tuning** | Train all parameters with differential LR (backbone: 1e-5, head: 1e-3) | Larger datasets, maximize performance |
| **Gradual Unfreezing** *(planned)* | Progressively unfreeze transformer layers epoch-by-epoch | Medium datasets, avoid catastrophic forgetting |

> **Note on fine-tuning**: The embedding extraction calls the core transformer's `forward()` directly with `torch.no_grad()`. To enable true fine-tuning with gradient flow, this `no_grad` guard would need to be removed and the backbone unfrozen. The `backbone.unfreeze(num_layers=N)` method exists in preparation for this.

### Baseline

A simple 1D CNN backbone is included for comparison — no external model needed. This helps isolate the benefit of TimesFM's pretrained representations.

## Installation

### Requirements

- Python 3.10+
- macOS (Apple Silicon supported), Linux, or Windows
- ~2GB disk space for TimesFM checkpoint

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/tsc-foundation-model.git
cd tsc-foundation-model

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### TimesFM Installation

TimesFM requires separate installation:

```bash
pip install timesfm

# The checkpoint (~925MB) downloads automatically on first use from:
# https://huggingface.co/google/timesfm-2.5-200m-pytorch
```

> **Note for macOS**: Apple Silicon (MPS) is supported. The model is automatically placed on the MPS device when configured with `device: mps`. CPU also works as a fallback.

## Quick Start

### 1. Test the Pipeline (no TimesFM needed)

Use the built-in CNN baseline to verify everything works:

```bash
python scripts/train.py --dataset ECG200 --config configs/baseline_cnn.yaml
```

### 2. Run with TimesFM (Linear Probing)

```bash
python scripts/train.py --dataset ECG200
```

### 3. Run with Fine-tuning

```bash
python scripts/train.py --dataset ECG200 --config configs/finetune.yaml
```

### 4. Extract & Visualize Embeddings

```bash
python scripts/extract_embeddings.py --dataset ECG200 --visualize
```

### 5. Benchmark Multiple Datasets

```bash
python scripts/run_benchmark.py --datasets ECG200 GunPoint FordA Wafer
```

## Configuration

Configuration is managed via YAML files in `configs/`. Key settings:

```yaml
model:
  backbone: "google/timesfm-2.5-200m-pytorch"  # or "cnn" for baseline
  context_len: 512          # Max input length (TimesFM context window)
  pooling: "mean"           # mean, max, last, attention
  freeze_backbone: true     # true for linear probing, false for fine-tuning

classifier:
  type: "linear"            # linear or mlp
  hidden_dims: [256]        # MLP hidden dimensions (if type=mlp)
  dropout: 0.1

training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-3       # Classification head LR
  backbone_lr: 1e-5         # Backbone LR (only used when fine-tuning)
  early_stopping_patience: 15
```

Override any setting via CLI:
```bash
python scripts/train.py --dataset GunPoint --epochs 50 --batch-size 16 --lr 5e-4
```

## Project Structure

```
tsc-foundation-model/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── default.yaml              # Default config (TimesFM + linear probing)
│   ├── finetune.yaml             # Fine-tuning config
│   └── baseline_cnn.yaml         # CNN baseline config
├── src/
│   ├── model/
│   │   ├── backbone.py           # TimesFM wrapper + CNN baseline
│   │   ├── classifier.py         # Classification heads (Linear, MLP)
│   │   └── tsc_model.py          # Combined TSC model
│   ├── data/
│   │   ├── dataset.py            # UCR dataset loading (via aeon)
│   │   └── preprocessing.py      # Normalization, padding
│   ├── training/
│   │   └── trainer.py            # Training loop with early stopping
│   ├── evaluation/
│   │   └── metrics.py            # Accuracy, F1, confusion matrix
│   └── utils/
│       └── config.py             # Configuration management
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation entry point
│   ├── extract_embeddings.py     # Extract & visualize embeddings
│   ├── inspect_model.py          # Inspect TimesFM model structure
│   └── run_benchmark.py          # Benchmark on multiple UCR datasets
└── tests/
    └── test_model.py             # Unit tests
```

## Evaluation

This project evaluates on the [UCR Time Series Archive](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) — the standard benchmark for univariate TSC with 128 datasets.

### Suggested Datasets for Initial Experiments

| Dataset | Train/Test | Length | Classes | Domain |
|---|---|---|---|---|
| ECG200 | 100/100 | 96 | 2 | Medical |
| GunPoint | 50/150 | 150 | 2 | Motion |
| FordA | 3601/1320 | 500 | 2 | Sensor |
| Wafer | 1000/6164 | 152 | 2 | Manufacturing |
| ElectricDevices | 8926/7711 | 96 | 7 | Energy |

## References

- **TimesFM**: Das et al., "A decoder-only foundation model for time-series forecasting," *ICML 2024*. [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)
- **UCR Archive**: Dau et al., "The UCR time series archive," *IEEE/CAA Journal of Automatica Sinica*, 2019.
- **Linear Probing**: Alain & Bengio, "Understanding intermediate layers using linear classifier probes," *ICLR Workshop 2017*.

## License

This project is for academic research purposes.

---

*PhD Research Project — Time Series Classification with Foundation Models*

### SOTA Reference Results (UCR Archive)

| Dataset | Best Model(s) | Accuracy |
|---|---|---|
| ECG200 | MultiRocket, MultiRocketHydra, ROCKET | 0.92 |
| GunPoint | HIVECOTEV2, InceptionTime, MultiRocket, Hydra, MultiRocketHydra, Arsenal, ROCKET | 1.00 |
| FordA | DrCIF | 0.9682 |
| Wafer | HIVECOTEV2, Hydra | 1.00 |
| ElectricDevices | HIVECOTEV2 | 0.7613 |

