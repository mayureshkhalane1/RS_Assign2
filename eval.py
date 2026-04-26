import math
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader


def recall_at_k(rank: int, k: int) -> float:
    return 1.0 if rank < k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    if rank < k:
        return 1.0 / math.log2(rank + 2.0)
    return 0.0


@torch.no_grad()
def evaluate_model(
    model,
    eval_loader: DataLoader,
    data_bundle,
    device: torch.device,
    k_list=(10, 20),
    eval_mode: str = "sampled",   # "sampled" or "full"
    num_negatives: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Validation/test evaluation.

    sampled:
      rank target among sampled negatives + target

    full:
      rank target among all unseen items
    """
    model.eval()
    rng = random.Random(seed)

    metric_sums = {f"Recall@{k}": 0.0 for k in k_list}
    metric_sums.update({f"NDCG@{k}": 0.0 for k in k_list})
    user_count = 0

    for batch in eval_loader:
        users, seqs, targets = batch
        users = users.to(device)
        seqs = seqs.to(device)
        targets = targets.to(device)

        batch_size = seqs.size(0)

        if eval_mode == "sampled":
            candidate_items = []
            for i in range(batch_size):
                user = users[i].item()
                seen = data_bundle.train_matrix[user]
                target = targets[i].item()

                negatives = []
                used = {target}
                while len(negatives) < num_negatives:
                    neg = rng.randint(1, data_bundle.num_items)
                    if neg not in seen and neg not in used:
                        negatives.append(neg)
                        used.add(neg)

                candidates = [target] + negatives
                candidate_items.append(candidates)

            candidate_items = torch.tensor(candidate_items, dtype=torch.long, device=device)
            scores = model.score_items(seqs, candidate_items)  # [B, 1+neg]
            # target is at index 0
            target_scores = scores[:, 0].unsqueeze(1)
            rank = (scores > target_scores).sum(dim=1)  # 0-based rank

        elif eval_mode == "full":
            all_items = torch.arange(1, data_bundle.num_items + 1, device=device).unsqueeze(0).expand(batch_size, -1)
            scores = model.score_items(seqs, all_items)  # [B, num_items]

            # mask seen items except target
            for i in range(batch_size):
                user = users[i].item()
                seen = data_bundle.train_matrix[user]
                target = targets[i].item()
                seen_to_mask = [it for it in seen if it != target]
                if seen_to_mask:
                    item_idx = torch.tensor([x - 1 for x in seen_to_mask], device=device, dtype=torch.long)
                    scores[i, item_idx] = -1e9

            target_positions = targets - 1
            target_scores = scores.gather(1, target_positions.unsqueeze(1))
            rank = (scores > target_scores).sum(dim=1)

        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")

        for r in rank.tolist():
            for k in k_list:
                metric_sums[f"Recall@{k}"] += recall_at_k(r, k)
                metric_sums[f"NDCG@{k}"] += ndcg_at_k(r, k)
            user_count += 1

    if user_count == 0:
        raise RuntimeError("No users evaluated. Check dataset or DataLoader setup.")

    metrics = {name: value / user_count for name, value in metric_sums.items()}
    return metrics