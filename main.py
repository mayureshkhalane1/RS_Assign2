"""
main.py — Experiment orchestration for SASRec on MovieLens-1M.

Usage
-----
    python main.py

Two configurations are run by default and compared in a summary table:

    config_small   — hidden_dim=64,  num_blocks=2, num_heads=2, max_len=50
    config_deeper  — hidden_dim=128, num_blocks=3, num_heads=4, max_len=100

The comparison covers all four dimensions required by the assignment rubric:
    1. number of self-attention blocks
    2. hidden size
    3. number of attention heads
    4. maximum sequence length
"""

import os
import pprint
from copy import deepcopy
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from dataset import (
    SequenceDataBundle,
    SASRecEvalDataset,
    SASRecTrainDataset,
    build_data_bundle,
    set_seed,
)
from eval import evaluate_model
from model import SASRec
from train import run_train_loop


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_loaders(
    data_bundle: SequenceDataBundle,
    config: Dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, valid_loader, test_loader) for the given config."""
    train_ds = SASRecTrainDataset(
        user_train=data_bundle.user_train,
        num_items=data_bundle.num_items,
        max_len=config["max_seq_len"],
        seed=config["seed"],
    )
    valid_ds = SASRecEvalDataset(
        data_bundle=data_bundle,
        max_len=config["max_seq_len"],
        mode="valid",
    )
    test_ds = SASRecEvalDataset(
        data_bundle=data_bundle,
        max_len=config["max_seq_len"],
        mode="test",
    )

    loader_kwargs = dict(
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, valid_loader, test_loader


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_one_experiment(
    exp_name: str,
    config: Dict[str, Any],
    data_bundle: SequenceDataBundle,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Train and evaluate one SASRec configuration.

    Returns a result dict suitable for ``print_comparison_table``.
    """
    print("=" * 100)
    print(f"Running experiment: {exp_name}")
    pprint.pprint(config)

    train_loader, valid_loader, test_loader = build_loaders(data_bundle, config)

    model = SASRec(
        num_items=data_bundle.num_items,
        max_len=config["max_seq_len"],
        hidden_dim=config["hidden_dim"],
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    train_result = run_train_loop(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        data_bundle=data_bundle,
        config=config,
        device=device,
    )

    test_metrics = evaluate_model(
        model=train_result["model"],
        eval_loader=test_loader,
        data_bundle=data_bundle,
        device=device,
        k_list=(10, 20),
        eval_mode=config["eval_mode"],
        num_negatives=config["eval_num_negatives"],
        seed=config["seed"],
    )

    result = {
        "experiment":          exp_name,
        "best_epoch":          train_result["best_epoch"],
        "best_valid_NDCG@10":  train_result["best_valid_ndcg10"],
        **test_metrics,
        "config":              deepcopy(config),
    }

    print(f"\nFinished: {exp_name}")
    print(result)
    return result


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table of all experiment results."""
    headers = [
        "Experiment",
        "BestEpoch",
        "ValNDCG@10",
        "TestRecall@10",
        "TestRecall@20",
        "TestNDCG@10",
        "TestNDCG@20",
    ]

    rows = [
        [
            r["experiment"],
            r["best_epoch"],
            f"{r['best_valid_NDCG@10']:.4f}",
            f"{r['Recall@10']:.4f}",
            f"{r['Recall@20']:.4f}",
            f"{r['NDCG@10']:.4f}",
            f"{r['NDCG@20']:.4f}",
        ]
        for r in results
    ]

    col_widths = [
        max(len(str(x)) for x in col) for col in zip(headers, *rows)
    ]

    def fmt_row(row: list) -> str:
        return " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))

    sep = "-" * (sum(col_widths) + 3 * (len(headers) - 1))

    print("\n" + "=" * 100)
    print("Comparison Table")
    print("=" * 100)
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print("=" * 100 + "\n")

    print("Assignment comparison dimensions:")
    print("  1. number of self-attention blocks  (num_blocks)")
    print("  2. hidden size                      (hidden_dim)")
    print("  3. number of attention heads        (num_heads)")
    print("  4. maximum sequence length          (max_seq_len)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    ratings_file = os.path.join(base_dir, "ratings.dat")

    # ------------------------------------------------------------------
    # Base configuration — shared by all experiments
    # eval_mode="full" is required by the assignment rubric
    # ------------------------------------------------------------------
    base_config: Dict[str, Any] = {
        "seed":                42,
        "batch_size":          256,
        "eval_batch_size":     256,
        "num_workers":         0,
        "pin_memory":          torch.cuda.is_available(),

        "max_seq_len":         50,
        "hidden_dim":          64,
        "num_blocks":          2,
        "num_heads":           2,
        "dropout":             0.2,

        "lr":                  1e-3,
        "weight_decay":        1e-5,    # FIX: was 0.0 — small L2 regularisation
        "epochs":              50,
        "early_stop_patience": 5,
        "grad_clip":           5.0,

        # FIX: was "sampled" — assignment requires full ranking evaluation
        "eval_mode":           "full",
        "eval_num_negatives":  100,     # only used when eval_mode="sampled"
    }

    # ------------------------------------------------------------------
    # Experiment grid — varies one or more architectural dimensions
    # ------------------------------------------------------------------
    experiment_configs: Dict[str, Dict[str, Any]] = {
        "config_small": {
            **base_config,
            "hidden_dim":  64,
            "num_blocks":  2,
            "num_heads":   2,
            "max_seq_len": 50,
        },
        "config_deeper": {
            **base_config,
            "hidden_dim":  128,
            "num_blocks":  3,
            "num_heads":   4,
            "max_seq_len": 100,
        },
    }

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_seed(base_config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_bundle = build_data_bundle(
        ratings_file=ratings_file,
        min_rating=4.0,
        min_user_interactions=5,
    )
    print(
        f"Data loaded — users: {data_bundle.num_users}, "
        f"items: {data_bundle.num_items}, "
        f"train users: {len(data_bundle.user_train)}"
    )

    # ------------------------------------------------------------------
    # Run experiments
    # ------------------------------------------------------------------
    all_results: List[Dict[str, Any]] = []
    for exp_name, config in experiment_configs.items():
        set_seed(config["seed"])
        result = run_one_experiment(
            exp_name=exp_name,
            config=config,
            data_bundle=data_bundle,
            device=device,
        )
        all_results.append(result)

    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
