import os
import pprint
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from dataset import (
    build_data_bundle,
    SASRecTrainDataset,
    SASRecEvalDataset,
    set_seed,
)
from model import SASRec
from train import run_train_loop
from eval import evaluate_model


def build_loaders(data_bundle, config):
    train_dataset = SASRecTrainDataset(
        user_train=data_bundle.user_train,
        num_items=data_bundle.num_items,
        max_len=config["max_seq_len"],
        seed=config["seed"],
    )
    valid_dataset = SASRecEvalDataset(
        data_bundle=data_bundle,
        max_len=config["max_seq_len"],
        mode="valid",
    )
    test_dataset = SASRecEvalDataset(
        data_bundle=data_bundle,
        max_len=config["max_seq_len"],
        mode="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", False),
    )
    return train_loader, valid_loader, test_loader


def run_one_experiment(exp_name, config, data_bundle, device):
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

    train_result = run_train_loop(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        data_bundle=data_bundle,
        config=config,
        device=device,
    )

    best_model = train_result["model"]

    test_metrics = evaluate_model(
        model=best_model,
        eval_loader=test_loader,
        data_bundle=data_bundle,
        device=device,
        k_list=(10, 20),
        eval_mode=config["eval_mode"],
        num_negatives=config["eval_num_negatives"],
        seed=config["seed"],
    )

    result = {
        "experiment": exp_name,
        "best_epoch": train_result["best_epoch"],
        "best_valid_NDCG@10": train_result["best_valid_ndcg10"],
        **test_metrics,
        "config": deepcopy(config),
    }

    print(f"Finished experiment: {exp_name}")
    print(result)
    return result


def print_comparison_table(results):
    headers = [
        "Experiment",
        "BestEpoch",
        "ValNDCG@10",
        "TestRecall@10",
        "TestRecall@20",
        "TestNDCG@10",
        "TestNDCG@20",
    ]

    rows = []
    for r in results:
        rows.append([
            r["experiment"],
            r["best_epoch"],
            f"{r['best_valid_NDCG@10']:.4f}",
            f"{r['Recall@10']:.4f}",
            f"{r['Recall@20']:.4f}",
            f"{r['NDCG@10']:.4f}",
            f"{r['NDCG@20']:.4f}",
        ])

    col_widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

    def fmt_row(row):
        return " | ".join(str(v).ljust(w) for v, w in zip(row, col_widths))

    print("\n" + "=" * 100)
    print("Comparison Table")
    print("=" * 100)
    print(fmt_row(headers))
    print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))
    for row in rows:
        print(fmt_row(row))
    print("=" * 100 + "\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ratings_file = os.path.join(base_dir, "ratings.dat")

    base_config = {
        "seed": 42,
        "batch_size": 256,
        "eval_batch_size": 256,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),

        "max_seq_len": 50,
        "hidden_dim": 64,
        "num_blocks": 2,
        "num_heads": 2,
        "dropout": 0.2,

        "lr": 1e-3,
        "weight_decay": 0.0,
        "epochs": 50,
        "early_stop_patience": 5,
        "grad_clip": 5.0,

        "eval_mode": "sampled",      # change to "full" if needed
        "eval_num_negatives": 100,
    }

    experiment_configs = {
        "config_small": {
            **base_config,
            "hidden_dim": 64,
            "num_blocks": 2,
            "num_heads": 2,
            "max_seq_len": 50,
            "dropout": 0.2,
        },
        "config_deeper": {
            **base_config,
            "hidden_dim": 128,
            "num_blocks": 3,
            "num_heads": 4,
            "max_seq_len": 100,
            "dropout": 0.2,
        },
    }

    set_seed(base_config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_bundle = build_data_bundle(
        ratings_file=ratings_file,
        min_rating=4.0,
        min_user_interactions=5,
    )

    print(
        f"Loaded data: users={data_bundle.num_users}, "
        f"items={data_bundle.num_items}, "
        f"train_users={len(data_bundle.user_train)}"
    )

    all_results = []
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

    print("Required comparison dimensions to mention in report:")
    print("1. number of self-attention blocks")
    print("2. hidden size")
    print("3. number of attention heads")
    print("4. maximum sequence length")


if __name__ == "__main__":
    main()