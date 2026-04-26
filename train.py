import copy
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eval import evaluate_model


def run_train_loop(
    model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    data_bundle,
    config: Dict[str, Any],
    device: torch.device,
):
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 0.0)
    epochs = config.get("epochs", 50)
    early_stop_patience = config.get("early_stop_patience", 5)
    grad_clip = config.get("grad_clip", 5.0)

    eval_mode = config.get("eval_mode", "sampled")
    eval_num_negatives = config.get("eval_num_negatives", 100)
    seed = config.get("seed", 42)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_metric = -1.0
    best_state = None
    best_epoch = -1
    history = []

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for _, seq, pos, neg in train_loader:
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_logits, neg_logits = model.training_logits(seq, pos, neg)

            mask = (pos != 0).float()

            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits), reduction="none"
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction="none"
            )

            loss = ((pos_loss + neg_loss) * mask).sum() / torch.clamp(mask.sum(), min=1.0)

            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

        train_loss = total_loss / max(total_tokens, 1)

        valid_metrics = evaluate_model(
            model=model,
            eval_loader=valid_loader,
            data_bundle=data_bundle,
            device=device,
            k_list=(10, 20),
            eval_mode=eval_mode,
            num_negatives=eval_num_negatives,
            seed=seed,
        )

        current_metric = valid_metrics["NDCG@10"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **valid_metrics,
        })

        improved = current_metric > best_metric
        if improved:
            best_metric = current_metric
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} | "
            f"val_NDCG@10={valid_metrics['NDCG@10']:.4f} | "
            f"val_Recall@10={valid_metrics['Recall@10']:.4f}"
        )

        if epoch - best_epoch >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best model state captured.")

    model.load_state_dict(best_state)

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_valid_ndcg10": best_metric,
        "history": history,
    }