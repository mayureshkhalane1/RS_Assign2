"""
eval.py — Ranking-based evaluation for SASRec.

Metrics
-------
Recall@K  = 1  if target rank < K, else 0
NDCG@K    = 1 / log₂(rank + 2)  if rank < K, else 0

Both metrics use 0-based ranking (rank 0 = top-1).

Evaluation modes
----------------
"full"     Rank target against ALL items the user has not yet seen.
           Slower but required by the assignment rubric.

"sampled"  Rank target against ``num_negatives`` randomly sampled unseen items.
           Fast approximation; biased but common in the literature.
"""

import math
import random
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def recall_at_k(rank: int, k: int) -> float:
    """Return 1.0 if ``rank`` is in the top-K, else 0.0.  Rank is 0-based."""
    return 1.0 if rank < k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    """Normalised discounted cumulative gain for a single query.  Rank is 0-based."""
    return 1.0 / math.log2(rank + 2.0) if rank < k else 0.0


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

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
    """
    Evaluate ``model`` on validation or test data.

    Parameters
    ----------
    model          : trained SASRec model
    eval_loader    : DataLoader yielding (user, seq, target)
    data_bundle    : SequenceDataBundle containing train_matrix and num_items
    device         : torch device
    k_list         : cutoffs for Recall@K and NDCG@K
    eval_mode      : "full" (assignment requirement) or "sampled" (fast approx)
    num_negatives  : number of negatives per user (sampled mode only)
    seed           : RNG seed for reproducible negative sampling

    Returns
    -------
    dict  e.g. {"Recall@10": 0.12, "NDCG@10": 0.08, "Recall@20": ..., ...}
    """
    model.eval()
    rng = random.Random(seed)

    metric_sums: Dict[str, float] = {}
    for k in k_list:
        metric_sums[f"Recall@{k}"] = 0.0
        metric_sums[f"NDCG@{k}"]   = 0.0
    user_count = 0

    for batch in eval_loader:
        users, seqs, targets = batch
        users   = users.to(device)
        seqs    = seqs.to(device)
        targets = targets.to(device)
        B       = seqs.size(0)

        if eval_mode == "sampled":
            # Build per-user candidate list: [target] + [num_negatives negatives]
            candidate_list = []
            for i in range(B):
                user   = users[i].item()
                target = targets[i].item()
                seen   = data_bundle.train_matrix[user]

                negatives: list = []
                used = {target}
                while len(negatives) < num_negatives:
                    neg = rng.randint(1, data_bundle.num_items)
                    if neg not in seen and neg not in used:
                        negatives.append(neg)
                        used.add(neg)

                candidate_list.append([target] + negatives)

            candidates   = torch.tensor(candidate_list, dtype=torch.long, device=device)
            scores       = model.score_items(seqs, candidates)          # [B, 1+neg]
            target_score = scores[:, 0].unsqueeze(1)                    # [B, 1]
            # 0-based rank = number of candidates scored higher than target
            rank = (scores > target_score).sum(dim=1)                   # [B]

        elif eval_mode == "full":
            # Score every item (1 … num_items)
            all_items = (
                torch.arange(1, data_bundle.num_items + 1, device=device)
                .unsqueeze(0).expand(B, -1)
            )
            scores = model.score_items(seqs, all_items)                 # [B, N]

            # Mask seen items (except the target) with −∞
            for i in range(B):
                user   = users[i].item()
                target = targets[i].item()
                seen   = data_bundle.train_matrix[user]

                # BUG FIX: explicitly exclude target from masking
                to_mask = [it for it in seen if it != target]
                if to_mask:
                    idx = torch.tensor(
                        [it - 1 for it in to_mask],   # 1-indexed → 0-indexed
                        device=device,
                        dtype=torch.long,
                    )
                    scores[i, idx] = -1e9

            # Gather scores of the target items
            target_idx   = (targets - 1).unsqueeze(1)                  # [B, 1]
            target_score = scores.gather(1, target_idx)                 # [B, 1]
            rank         = (scores > target_score).sum(dim=1)          # [B]

        else:
            raise ValueError(
                f"Unknown eval_mode '{eval_mode}'. Choose 'full' or 'sampled'."
            )

        for r in rank.tolist():
            for k in k_list:
                metric_sums[f"Recall@{k}"] += recall_at_k(r, k)
                metric_sums[f"NDCG@{k}"]   += ndcg_at_k(r, k)
            user_count += 1

    if user_count == 0:
        raise RuntimeError("No users were evaluated. Check your DataLoader.")

    return {name: val / user_count for name, val in metric_sums.items()}
