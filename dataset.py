"""Data loading, preprocessing, and PyTorch datasets for SASRec on MovieLens-1M."""

import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceDataBundle:
    """All processed splits and metadata for one dataset.

    Attributes
    ----------
    user_train : dict[int, list[int]]
        Chronological training sequence per user.
    user_valid : dict[int, int]
        Single held-out validation item per user.
    user_test : dict[int, int]
        Single held-out test item per user.
    num_users : int
        Total number of users after filtering.
    num_items : int
        Total number of items after filtering. Item 0 is padding.
    train_matrix : dict[int, set]
        Full interaction history per user (train + valid + test).
        Used at evaluation to mask already-seen items.
    """

    user_train: Dict[int, List[int]]
    user_valid: Dict[int, int]
    user_test: Dict[int, int]
    num_users: int
    num_items: int
    train_matrix: Dict[int, set] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    """Set random seeds on CPU and GPU for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value applied to random, numpy, and torch (including CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_movielens_1m(
    file_path: str,
    min_rating: float = 4.0,
    min_user_interactions: int = 5,
) -> Tuple[Dict[int, List[int]], int, int]:
    """Load MovieLens-1M and return chronological per-user item sequences.

    Ratings below ``min_rating`` are discarded. Each user's interactions are
    sorted by timestamp, users with fewer than ``min_user_interactions`` items
    are dropped, and all IDs are remapped to contiguous integers starting from 1.
    Index 0 is reserved for padding throughout.

    Parameters
    ----------
    file_path : str
        Path to ``ratings.dat`` in ``userId::movieId::rating::timestamp`` format.
    min_rating : float
        Minimum rating treated as a positive interaction.
    min_user_interactions : int
        Minimum sequence length required to keep a user.

    Returns
    -------
    sequences : dict[int, list[int]]
        Remapped user ID to list of remapped item IDs in chronological order.
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

    filtered: Dict[int, List[int]] = {}
    for uid, events in raw.items():
        events.sort(key=lambda x: x[0])
        items = [iid for _, iid in events]
        if len(items) >= min_user_interactions:
            filtered[uid] = items

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


def leave_one_out_split(
    user_sequences: Dict[int, List[int]],
) -> SequenceDataBundle:
    """Split each user sequence into train/valid/test using leave-one-out.

    For a sequence of length n:
        train = seq[:-2]  (all but the last two items)
        valid = seq[-2]   (second-to-last)
        test  = seq[-1]   (last)

    ``train_matrix`` stores the complete per-user history so that evaluation
    can exclude all seen items when ranking, except the evaluation target.

    Parameters
    ----------
    user_sequences : dict[int, list[int]]
        Chronological item sequences keyed by remapped user ID.

    Returns
    -------
    SequenceDataBundle
    """
    user_train: Dict[int, List[int]] = {}
    user_valid: Dict[int, int] = {}
    user_test: Dict[int, int] = {}
    train_matrix: Dict[int, set] = {}

    for user, seq in user_sequences.items():
        if len(seq) < 3:
            continue
        user_train[user] = seq[:-2]
        user_valid[user] = seq[-2]
        user_test[user] = seq[-1]
        train_matrix[user] = set(seq)

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


def pad_or_truncate(seq: List[int], max_len: int) -> np.ndarray:
    """Return a left-zero-padded, right-truncated copy of ``seq`` of length ``max_len``.

    The most recent items are right-aligned so position ``-1`` always
    corresponds to the last observed item, which the model predicts from.

    Parameters
    ----------
    seq : list[int]
    max_len : int

    Returns
    -------
    np.ndarray of shape (max_len,) with dtype int64
    """
    seq = seq[-max_len:]
    out = np.zeros(max_len, dtype=np.int64)
    out[-len(seq):] = seq
    return out


def sample_negative(exclude_items: set, num_items: int, rng: random.Random) -> int:
    """Sample a random item ID not in ``exclude_items``.

    Parameters
    ----------
    exclude_items : set
        Item IDs to avoid (typically the user's full interaction history).
    num_items : int
        Upper bound of valid item IDs (items are in [1, num_items]).
    rng : random.Random
        Caller-controlled RNG for reproducibility.

    Returns
    -------
    int
    """
    while True:
        neg = rng.randint(1, num_items)
        if neg not in exclude_items:
            return neg


class SASRecTrainDataset(Dataset):
    """Token-level next-item prediction dataset used during training.

    For a user with training sequence [i1, i2, ..., iN], each sample contains:

        input  = [PAD, ..., i1,   i2,   ..., i(N-1)]
        pos    = [PAD, ..., i2,   i3,   ..., iN    ]
        neg    = [PAD, ..., n1,   n2,   ..., nN    ]

    where PAD = 0. Loss is computed only at positions where ``pos != 0``.

    Parameters
    ----------
    user_train : dict[int, list[int]]
    num_items : int
    max_len : int
    seed : int
    """

    def __init__(
        self,
        user_train: Dict[int, List[int]],
        num_items: int,
        max_len: int,
        seed: int = 42,
    ) -> None:
        self.user_train = user_train
        self.users = list(user_train.keys())
        self.num_items = num_items
        self.max_len = max_len
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int):
        user = self.users[index]
        seq = self.user_train[user]

        input_seq = np.zeros(self.max_len, dtype=np.int64)
        pos_seq = np.zeros(self.max_len, dtype=np.int64)
        neg_seq = np.zeros(self.max_len, dtype=np.int64)

        seen = set(seq)
        nxt = seq[-1]
        idx = self.max_len - 1

        for item in reversed(seq[:-1]):
            input_seq[idx] = item
            pos_seq[idx] = nxt
            if nxt != 0:
                neg_seq[idx] = sample_negative(seen, self.num_items, self.rng)
            nxt = item
            idx -= 1
            if idx < 0:
                break

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(pos_seq, dtype=torch.long),
            torch.tensor(neg_seq, dtype=torch.long),
        )


class SASRecEvalDataset(Dataset):
    """Per-user evaluation dataset for validation or test.

    Parameters
    ----------
    data_bundle : SequenceDataBundle
    max_len : int
    mode : {'valid', 'test'}
        ``valid`` uses the training sequence to predict the validation item.
        ``test`` appends the validation item to the training sequence and
        predicts the test item.
    """

    def __init__(
        self,
        data_bundle: SequenceDataBundle,
        max_len: int,
        mode: str = "valid",
    ) -> None:
        assert mode in ("valid", "test"), f"Unknown mode: {mode}"
        self.data_bundle = data_bundle
        self.max_len = max_len
        self.mode = mode
        self.users = list(data_bundle.user_train.keys())

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int):
        user = self.users[index]
        train_seq = self.data_bundle.user_train[user]

        if self.mode == "valid":
            seq = train_seq
            target = self.data_bundle.user_valid[user]
        else:
            seq = train_seq + [self.data_bundle.user_valid[user]]
            target = self.data_bundle.user_test[user]

        seq_arr = pad_or_truncate(seq, self.max_len)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(seq_arr, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def build_data_bundle(
    ratings_file: str,
    min_rating: float = 4.0,
    min_user_interactions: int = 5,
) -> SequenceDataBundle:
    """Load MovieLens-1M and return a ready-to-use ``SequenceDataBundle``.

    Parameters
    ----------
    ratings_file : str
        Path to ``ratings.dat``.
    min_rating : float
    min_user_interactions : int

    Returns
    -------
    SequenceDataBundle
    """
    user_sequences, num_users, num_items = load_movielens_1m(
        ratings_file,
        min_rating=min_rating,
        min_user_interactions=min_user_interactions,
    )
    bundle = leave_one_out_split(user_sequences)
    # num_users and num_items from the loader are authoritative; the bundle's
    # own values (derived inside leave_one_out_split) are overridden here.
    bundle.num_users = num_users
    bundle.num_items = num_items
    return bundle
