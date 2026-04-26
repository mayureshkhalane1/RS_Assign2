"""Ranking-based evaluation for SASRec.

Two modes are supported:
    full     -- rank the target against all items the user has not seen.
    sampled  -- rank the target against ``num_negatives`` random unseen items.

All ranks are 0-based (rank 0 means the model placed the target first).
"""

import math
import random
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader


def recall_at_k(rank: int, k: int) -> float:
    """Return 1.0 if ``rank`` < k, else 0.0.

    Parameters
    ----------
    rank : int
        0-based rank of the target item.
    k : int
        Cutoff.

    Returns
    -------
    float
    """
    return 1.0 if rank < k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    """Return the normalised discounted cumulative gain for a single query.

    Parameters
    ----------
    rank : int
        0-based rank of the target item.
    k : int
        Cutoff.

    Returns
    -------
    float
    """
    return 1.0 / math.log2(rank + 2.0) if rank < k else 0.0


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    eval_loader: DataLoader,
    data_bundle,
    device: torch.device,
    k_list: Tuple[int, ...] = (10, 20),
    eval_mode: str = "full",
    num_negatives: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate ``model`` and return average Recall@K and NDCG@K per user.

    Parameters
    ----------
    model : torch.nn.Module
        Trained SASRec model.
    eval_loader : DataLoader
        Yields ``(user, seq, target)`` batches.
    data_bundle : SequenceDataBundle
        Provides ``train_matrix`` for seen-item masking and ``num_items``.
    device : torch.device
    k_list : tuple of int
        Cutoffs, e.g. ``(10, 20)``.
    eval_mode : {'full', 'sampled'}
        ``full`` ranks against all unseen items (required for the assignment).
        ``sampled`` ranks against ``num_negatives`` randomly drawn unseen items.
    num_negatives : int
        Number of negatives per user in sampled mode.
    seed : int
        RNG seed for reproducible negative sampling.

    Returns
    -------
    dict[str, float]
        Keys are ``Recall@K`` and ``NDCG@K`` for each k in ``k_list``.
    """
    model.eval()
    rng = random.Random(seed)

    metric_sums: Dict[str, float] = {
        f"{m}@{k}": 0.0 for m in ("Recall", "NDCG") for k in k_list
    }
    user_count = 0

    for batch in eval_loader:
        users, seqs, targets = batch
        users = users.to(device)
        seqs = seqs.to(device)
        targets = targets.to(device)
        B = seqs.size(0)

        if eval_mode == "sampled":
            candidate_list = []
            for i in range(B):
                user = users[i].item()
                target = targets[i].item()
                seen = data_bundle.train_matrix[user]
                used = {target}
                negatives = []
                while len(negatives) < num_negatives:
                    neg = rng.randint(1, data_bundle.num_items)
                    if neg not in seen and neg not in used:
                        negatives.append(neg)
                        used.add(neg)
                candidate_list.append([target] + negatives)

            candidates = torch.tensor(candidate_list, dtype=torch.long, device=device)
            scores = model.score_items(seqs, candidates)
            target_score = scores[:, 0].unsqueeze(1)
            rank = (scores > target_score).sum(dim=1)

        elif eval_mode == "full":
            all_items = (
                torch.arange(1, data_bundle.num_items + 1, device=device)
                .unsqueeze(0)
                .expand(B, -1)
            )
            scores = model.score_items(seqs, all_items)

            for i in range(B):
                user = users[i].item()
                target = targets[i].item()
                seen = data_bundle.train_matrix[user]
                to_mask = [it for it in seen if it != target]
                if to_mask:
                    # Items are 1-indexed; shift to 0-indexed for the score tensor.
                    idx = torch.tensor(
                        [it - 1 for it in to_mask], device=device, dtype=torch.long
                    )
                    scores[i, idx] = -1e9

            target_idx = (targets - 1).unsqueeze(1)
            target_score = scores.gather(1, target_idx)
            rank = (scores > target_score).sum(dim=1)

        else:
            raise ValueError(
                f"Unknown eval_mode '{eval_mode}'. Choose 'full' or 'sampled'."
            )

        for r in rank.tolist():
            for k in k_list:
                metric_sums[f"Recall@{k}"] += recall_at_k(r, k)
                metric_sums[f"NDCG@{k}"] += ndcg_at_k(r, k)
            user_count += 1

    if user_count == 0:
        raise RuntimeError("No users were evaluated. Check your DataLoader.")

    return {name: val / user_count for name, val in metric_sums.items()}
