"""
train.py — Training loop for SASRec.

Objective
---------
Binary Cross-Entropy with negative sampling at every sequence position:

    L = -∑_{t: pos_t ≠ 0} [ log σ(h_t · e_{pos_t}) + log(1 − σ(h_t · e_{neg_t})) ]

Optimisation
------------
* Adam with optional L2 weight decay and gradient clipping.
* Early stopping monitored on validation NDCG@10.
* Best model state is deep-copied and restored before returning.
"""

import copy
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eval import evaluate_model


def run_train_loop(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    data_bundle,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Train ``model`` and return the best checkpoint together with metrics.

    Parameters
    ----------
    model        : SASRec instance (uninitialised — will be moved to device)
    train_loader : DataLoader yielding (user, seq, pos, neg)
    valid_loader : DataLoader yielding (user, seq, target)
    data_bundle  : SequenceDataBundle with train_matrix and num_items
    config       : experiment configuration dict (see main.py for keys)
    device       : torch device to train on

    Returns
    -------
    dict with keys:
        "model"              — best model (state loaded)
        "best_epoch"         — epoch index of best validation NDCG@10
        "best_valid_ndcg10"  — best validation NDCG@10 value
        "history"            — list of per-epoch metric dicts
    """
    lr                  = config.get("lr",                   1e-3)
    weight_decay        = config.get("weight_decay",         1e-5)  # FIX: was 0.0
    epochs              = config.get("epochs",               50)
    early_stop_patience = config.get("early_stop_patience",  5)
    grad_clip           = config.get("grad_clip",            5.0)
    eval_mode           = config.get("eval_mode",            "full")
    eval_num_negatives  = config.get("eval_num_negatives",   100)
    seed                = config.get("seed",                 42)

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay,
    )

    best_metric = -1.0
    best_state  = None
    best_epoch  = -1
    history     = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss   = 0.0
        total_tokens = 0

        for _, seq, pos, neg in train_loader:
            seq = seq.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            pos_logits, neg_logits = model.training_logits(seq, pos, neg)

            # Mask: only compute loss at non-padding positions
            mask = (pos != 0).float()

            pos_loss = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits), reduction="none",
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction="none",
            )

            # Average over real (non-padding) tokens
            n_tokens = torch.clamp(mask.sum(), min=1.0)
            loss     = ((pos_loss + neg_loss) * mask).sum() / n_tokens

            optimizer.zero_grad()
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_loss   += loss.item() * mask.sum().item()
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
        epoch_log = {
            "epoch":      epoch,
            "train_loss": train_loss,
            **valid_metrics,
        }
        history.append(epoch_log)

        improved = current_metric > best_metric
        if improved:
            best_metric = current_metric
            best_state  = copy.deepcopy(model.state_dict())
            best_epoch  = epoch

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={train_loss:.4f} | "
            f"val_NDCG@10={valid_metrics['NDCG@10']:.4f} | "
            f"val_Recall@10={valid_metrics['Recall@10']:.4f}"
            + (" ✓ best" if improved else "")
        )

        if epoch - best_epoch >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} — best was epoch {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training produced no valid model state.")

    model.load_state_dict(best_state)

    return {
        "model":             model,
        "best_epoch":        best_epoch,
        "best_valid_ndcg10": best_metric,
        "history":           history,
    }
