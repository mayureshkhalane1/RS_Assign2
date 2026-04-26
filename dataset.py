"""
dataset.py — Data loading, preprocessing, and PyTorch datasets for SASRec.

Key design decisions
---------------------
* Item ID 0 is reserved for padding throughout.
* leave-one-out split: train = seq[:-2], valid = seq[-2], test = seq[-1].
* train_matrix stores the *full* interaction set per user so evaluation can
  mask already-seen items (except the evaluation target) when computing ranks.
"""

import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SequenceDataBundle:
    """Holds all processed splits and metadata for one dataset."""
    user_train:   Dict[int, List[int]]   # user -> training item sequence
    user_valid:   Dict[int, int]          # user -> single validation item
    user_test:    Dict[int, int]          # user -> single test item
    num_users:    int
    num_items:    int
    # full history per user (train+valid+test) used to mask seen items at eval
    train_matrix: Dict[int, set] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility (CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # BUG FIX: was missing, broke GPU repro


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_movielens_1m(
    file_path: str,
    min_rating: float = 4.0,
    min_user_interactions: int = 5,
) -> Tuple[Dict[int, List[int]], int, int]:
    """
    Load MovieLens-1M ratings.dat and convert to chronological item sequences.

    Format expected: userId::movieId::rating::timestamp

    Steps
    -----
    1. Filter out ratings below ``min_rating``.
    2. Sort each user's interactions by timestamp.
    3. Remove users with fewer than ``min_user_interactions`` items.
    4. Remap user/item IDs to contiguous integers starting from 1
       (0 is reserved for padding).

    Returns
    -------
    remapped_sequences : dict[int, list[int]]
    num_users : int
    num_items : int
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    raw: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue
            user_id, item_id, rating, timestamp = parts
            if float(rating) < min_rating:
                continue
            raw[int(user_id)].append((int(timestamp), int(item_id)))

    # Sort by time, filter short sequences
    filtered: Dict[int, List[int]] = {}
    for uid, events in raw.items():
        events.sort(key=lambda x: x[0])
        items = [iid for _, iid in events]
        if len(items) >= min_user_interactions:
            filtered[uid] = items

    # Remap to contiguous IDs (1-indexed; 0 = padding)
    user_map: Dict[int, int] = {}
    item_map: Dict[int, int] = {}
    next_uid = 1
    next_iid = 1
    remapped: Dict[int, List[int]] = {}

    for raw_uid in sorted(filtered):
        user_map[raw_uid] = next_uid
        next_uid += 1
        seq: List[int] = []
        for raw_iid in filtered[raw_uid]:
            if raw_iid not in item_map:
                item_map[raw_iid] = next_iid
                next_iid += 1
            seq.append(item_map[raw_iid])
        remapped[user_map[raw_uid]] = seq

    return remapped, next_uid - 1, next_iid - 1


# ---------------------------------------------------------------------------
# Train/valid/test split
# ---------------------------------------------------------------------------

def leave_one_out_split(
    user_sequences: Dict[int, List[int]],
) -> SequenceDataBundle:
    """
    Apply leave-one-out split to each user's chronological item sequence.

    Split rule (requires len >= 3):
        train = seq[:-2]   (all but last two items)
        valid = seq[-2]    (second-to-last)
        test  = seq[-1]    (last)

    train_matrix stores the *complete* interaction history per user.
    This is used during evaluation to mask all items the user has seen
    (except the evaluation target) so we never credit the model for
    re-ranking already-consumed content.
    """
    user_train:   Dict[int, List[int]] = {}
    user_valid:   Dict[int, int]       = {}
    user_test:    Dict[int, int]       = {}
    train_matrix: Dict[int, set]       = {}

    for user, seq in user_sequences.items():
        if len(seq) < 3:
            continue
        user_train[user]   = seq[:-2]
        user_valid[user]   = seq[-2]
        user_test[user]    = seq[-1]
        train_matrix[user] = set(seq)   # full history for seen-item masking

    num_users = max(user_sequences.keys())
    num_items = max(max(seq) for seq in user_sequences.values())

    return SequenceDataBundle(
        user_train=user_train,
        user_valid=user_valid,
        user_test=user_test,
        num_users=num_users,
        num_items=num_items,
        train_matrix=train_matrix,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pad_or_truncate(seq: List[int], max_len: int) -> np.ndarray:
    """
    Left-pad with 0s and right-truncate so the output is exactly max_len.

    The most recent items are kept (right-aligned), which is what the causal
    attention mask expects — position -1 is always the prediction target.
    """
    seq = seq[-max_len:]                       # keep most recent max_len items
    out = np.zeros(max_len, dtype=np.int64)
    out[-len(seq):] = seq
    return out


def sample_negative(
    exclude_items: set,
    num_items: int,
    rng: random.Random,
) -> int:
    """Sample a uniform random item not in ``exclude_items``."""
    while True:
        neg = rng.randint(1, num_items)
        if neg not in exclude_items:
            return neg


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class SASRecTrainDataset(Dataset):
    """
    Token-level next-item prediction dataset for training.

    For a user with training sequence [i1, i2, i3, ..., iN]:

        input  = [0, ..., i1,   i2,   ..., i(N-1)]   ← left-padded prefix
        pos    = [0, ..., i2,   i3,   ..., iN    ]   ← next item at each pos
        neg    = [0, ..., n1,   n2,   ..., nN    ]   ← random negatives

    Padding positions (pos == 0) are masked out during the loss computation.
    """

    def __init__(
        self,
        user_train: Dict[int, List[int]],
        num_items: int,
        max_len: int,
        seed: int = 42,
    ) -> None:
        self.user_train = user_train
        self.users      = list(user_train.keys())
        self.num_items  = num_items
        self.max_len    = max_len
        self.rng        = random.Random(seed)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int):
        user = self.users[index]
        seq  = self.user_train[user]

        input_seq = np.zeros(self.max_len, dtype=np.int64)
        pos_seq   = np.zeros(self.max_len, dtype=np.int64)
        neg_seq   = np.zeros(self.max_len, dtype=np.int64)

        seen = set(seq)
        nxt  = seq[-1]
        idx  = self.max_len - 1

        for item in reversed(seq[:-1]):
            input_seq[idx] = item
            pos_seq[idx]   = nxt
            if nxt != 0:
                neg_seq[idx] = sample_negative(seen, self.num_items, self.rng)
            nxt  = item
            idx -= 1
            if idx < 0:
                break

        return (
            torch.tensor(user,      dtype=torch.long),
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(pos_seq,   dtype=torch.long),
            torch.tensor(neg_seq,   dtype=torch.long),
        )


class SASRecEvalDataset(Dataset):
    """
    One sample per user for validation or test evaluation.

    mode='valid'
        sequence = train items
        target   = validation item

    mode='test'
        sequence = train + validation items
        target   = test item
    """

    def __init__(
        self,
        data_bundle: SequenceDataBundle,
        max_len: int,
        mode: str = "valid",
    ) -> None:
        assert mode in ("valid", "test"), f"Unknown mode: {mode}"
        self.data_bundle = data_bundle
        self.max_len     = max_len
        self.mode        = mode
        self.users       = list(data_bundle.user_train.keys())

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int):
        user      = self.users[index]
        train_seq = self.data_bundle.user_train[user]

        if self.mode == "valid":
            seq    = train_seq
            target = self.data_bundle.user_valid[user]
        else:
            seq    = train_seq + [self.data_bundle.user_valid[user]]
            target = self.data_bundle.user_test[user]

        seq_arr = pad_or_truncate(seq, self.max_len)

        return (
            torch.tensor(user,    dtype=torch.long),
            torch.tensor(seq_arr, dtype=torch.long),
            torch.tensor(target,  dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def build_data_bundle(
    ratings_file: str,
    min_rating: float = 4.0,
    min_user_interactions: int = 5,
) -> SequenceDataBundle:
    """Load MovieLens-1M and return a fully processed SequenceDataBundle."""
    user_sequences, num_users, num_items = load_movielens_1m(
        ratings_file,
        min_rating=min_rating,
        min_user_interactions=min_user_interactions,
    )
    bundle = leave_one_out_split(user_sequences)
    # Override with the remapped counts from the loader (authoritative source)
    bundle.num_users = num_users
    bundle.num_items = num_items
    return bundle
