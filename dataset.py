import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SequenceDataBundle:
    user_train: Dict[int, List[int]]
    user_valid: Dict[int, int]
    user_test: Dict[int, int]
    num_users: int
    num_items: int
    train_matrix: Dict[int, set]  # user -> set(items seen in train/val/test or train only, depending on need)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_movielens_1m(
    file_path: str,
    min_rating: float = 4.0,
    min_user_interactions: int = 5,
) -> Tuple[Dict[int, List[int]], int, int]:
    """
    Load ratings.dat with format:
    userId::movieId::rating::timestamp

    Keeps only rating >= min_rating.
    Returns remapped user/item sequences sorted by timestamp.
    User IDs and item IDs are remapped to contiguous integers starting from 1.
    Index 0 is reserved for padding.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find dataset file: {file_path}")

    raw_interactions = defaultdict(list)

    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) != 4:
                continue

            user_id, item_id, rating, timestamp = parts
            rating = float(rating)
            if rating < min_rating:
                continue

            raw_interactions[int(user_id)].append((int(timestamp), int(item_id)))

    # Sort each user's interactions by time
    filtered_users = {}
    for user_id, events in raw_interactions.items():
        events.sort(key=lambda x: x[0])
        items = [item_id for _, item_id in events]
        if len(items) >= min_user_interactions:
            filtered_users[user_id] = items

    # Remap users and items to contiguous ids
    user_mapping = {}
    item_mapping = {}
    next_user_id = 1
    next_item_id = 1

    remapped_sequences = {}

    for raw_user_id in sorted(filtered_users.keys()):
        if raw_user_id not in user_mapping:
            user_mapping[raw_user_id] = next_user_id
            next_user_id += 1

        mapped_user = user_mapping[raw_user_id]
        remapped_sequences[mapped_user] = []

        for raw_item_id in filtered_users[raw_user_id]:
            if raw_item_id not in item_mapping:
                item_mapping[raw_item_id] = next_item_id
                next_item_id += 1
            remapped_sequences[mapped_user].append(item_mapping[raw_item_id])

    num_users = next_user_id - 1
    num_items = next_item_id - 1
    return remapped_sequences, num_users, num_items


def leave_one_out_split(user_sequences: Dict[int, List[int]]) -> SequenceDataBundle:
    """
    For each user sequence:
      train = all but last two
      valid = second last
      test  = last
    """
    user_train = {}
    user_valid = {}
    user_test = {}
    train_matrix = {}

    for user, seq in user_sequences.items():
        if len(seq) < 3:
            continue
        user_train[user] = seq[:-2]
        user_valid[user] = seq[-2]
        user_test[user] = seq[-1]
        train_matrix[user] = set(seq)  # useful for filtering seen items in eval

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
    """
    Left-pad with 0, truncate from the left if too long.
    """
    seq = seq[-max_len:]
    out = np.zeros(max_len, dtype=np.int64)
    out[-len(seq):] = seq
    return out


def sample_negative(exclude_items: set, num_items: int, rng: random.Random) -> int:
    while True:
        neg = rng.randint(1, num_items)
        if neg not in exclude_items:
            return neg


class SASRecTrainDataset(Dataset):
    """
    Generates training samples for next-item prediction.
    For each user train sequence [i1, i2, i3, ..., in]
    create token-level training:
      input  = [0, 0, ..., i1, i2, ..., i(n-1)]
      pos    = [0, 0, ..., i2, i3, ..., in]
      neg    = sampled per positive position
    """
    def __init__(
        self,
        user_train: Dict[int, List[int]],
        num_items: int,
        max_len: int,
        seed: int = 42,
    ):
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

        nxt = seq[-1]
        idx = self.max_len - 1
        seen = set(seq)

        # fill from right to left
        for i in reversed(seq[:-1]):
            input_seq[idx] = i
            pos_seq[idx] = nxt
            if nxt != 0:
                neg_seq[idx] = sample_negative(seen, self.num_items, self.rng)
            nxt = i
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
    """
    One sample per user for validation/test.
    Produces the sequence prefix and target item.

    mode='valid':
      sequence = train
      target   = valid
    mode='test':
      sequence = train + valid
      target   = test
    """
    def __init__(
        self,
        data_bundle: SequenceDataBundle,
        max_len: int,
        mode: str = "valid",
    ):
        assert mode in ("valid", "test")
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

        seq = pad_or_truncate(seq, self.max_len)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def build_data_bundle(
    ratings_file: str,
    min_rating: float = 4.0,
    min_user_interactions: int = 5,
) -> SequenceDataBundle:
    user_sequences, num_users, num_items = load_movielens_1m(
        ratings_file,
        min_rating=min_rating,
        min_user_interactions=min_user_interactions,
    )
    data = leave_one_out_split(user_sequences)
    data.num_users = num_users
    data.num_items = num_items
    return data