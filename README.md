# SASRec Sequential Recommendation

Sequential recommendation model based on **SASRec** (Kang & McAuley, 2018),
trained and evaluated on **MovieLens-1M**.

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch numpy
```

Python ≥ 3.10 and PyTorch ≥ 2.0 recommended.

### 2. Place the dataset

Download [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) and put
`ratings.dat` in the same directory as `main.py`.

```
.
├── ratings.dat        ← required
├── dataset.py
├── model.py
├── train.py
├── eval.py
└── main.py
```

### 3. Run

```bash
python main.py
```

That's it. All required-setting experiment configurations will train
sequentially and a comparison table is printed at the end.

---

## What the run does

1. Loads and preprocesses `ratings.dat` (filters rating ≥ 4, min 5 interactions per user)
2. Builds chronological sequences and applies leave-one-out splits
3. Trains `config_base` (64-dim, 2 blocks, 2 heads, len=50)
4. Trains one-setting-at-a-time variants:
   - `setting_num_blocks` (3 blocks)
   - `setting_hidden_dim` (128 hidden dim)
   - `setting_num_heads` (4 heads)
   - `setting_max_seq_len` (len=100)
5. Trains `config_deeper` (128-dim, 3 blocks, 4 heads, len=100)
6. Evaluates all configurations on test set with **full ranking** (as required by the assignment)
7. Prints a comparison table across all four architectural dimensions

---

## Configuration

All knobs live in `main.py` inside `base_config` and `experiment_configs`.

| Key | Default | Description |
|-----|---------|-------------|
| `hidden_dim` | 64 | Embedding / model width H |
| `num_blocks` | 2 | Number of stacked SASRec blocks |
| `num_heads` | 2 | Attention heads (must divide hidden_dim) |
| `max_seq_len` | 50 | Input sequence length (truncated/padded to this) |
| `dropout` | 0.2 | Dropout rate throughout |
| `lr` | 1e-3 | Adam learning rate |
| `weight_decay` | 1e-5 | L2 regularisation |
| `epochs` | 50 | Max training epochs |
| `early_stop_patience` | 5 | Stop if val NDCG@10 does not improve for N epochs |
| `grad_clip` | 5.0 | Gradient norm clipping |
| `eval_mode` | `"full"` | `"full"` = rank among all items; `"sampled"` = fast approx |

### Adding a new experiment

```python
"config_wide": {
    **base_config,
    "hidden_dim":  256,
    "num_blocks":  2,
    "num_heads":   8,
    "max_seq_len": 50,
},
```

> **Constraint**: `hidden_dim` must be divisible by `num_heads`.

---

## Bug fixes vs original code

| File | Fix |
|------|-----|
| `dataset.py` | `set_seed` now calls `torch.cuda.manual_seed_all` (GPU reproducibility) |
| `model.py` | FFN expanded to 4×hidden_dim (standard transformer capacity) |
| `model.py` | Padding positions zeroed **before** the first block (prevents positional embedding leakage) |
| `model.py` | Activation changed from ReLU → GELU (smoother, better for transformers) |
| `model.py` | Padding embedding explicitly zeroed after `_reset_parameters` |
| `train.py` | `weight_decay` default changed from `0.0` → `1e-5` (mild L2 regularisation) |
| `main.py` | `eval_mode` changed from `"sampled"` → `"full"` (assignment requires full ranking) |

---

## Evaluation metrics

| Metric | Description |
|--------|-------------|
| Recall@10 | 1 if target is in top-10, else 0 |
| Recall@20 | 1 if target is in top-20, else 0 |
| NDCG@10 | 1/log₂(rank+2) if rank < 10, else 0 |
| NDCG@20 | 1/log₂(rank+2) if rank < 20, else 0 |

Full-ranking mode ranks the target against **all unseen items**, which is the
stricter and unbiased evaluation required by the assignment.

---

## Expected runtime (CPU)

| Config | Approx. time per epoch |
|--------|------------------------|
| config_small (full eval) | ~3–5 min |
| config_deeper (full eval) | ~5–8 min |

Use `eval_mode: "sampled"` during development to iterate faster, then switch
to `"full"` for final results.
