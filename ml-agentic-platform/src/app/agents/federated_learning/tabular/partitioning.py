from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClientPartition:
    client_id: str
    owned_columns: list[str]


def build_vertical_partitions(feature_names: list[str], num_clients: int = 3) -> list[ClientPartition]:
    if num_clients < 2:
        raise ValueError("num_clients must be >= 2")
    if len(feature_names) < num_clients:
        raise ValueError("Number of features must be >= num_clients for disjoint partitioning")

    column_indexes = np.array_split(np.arange(len(feature_names)), num_clients)
    partitions: list[ClientPartition] = []

    for i, index_block in enumerate(column_indexes):
        owned = [feature_names[j] for j in index_block.tolist()]
        partitions.append(ClientPartition(client_id=str(i), owned_columns=owned))

    validate_vertical_partitions(partitions=partitions, feature_names=feature_names)
    return partitions


def apply_vertical_mask(frame: pd.DataFrame, owned_columns: list[str]) -> pd.DataFrame:
    missing_columns = [col for col in owned_columns if col not in frame.columns]
    if missing_columns:
        raise ValueError(f"Owned columns missing from frame: {missing_columns}")

    masked = pd.DataFrame(
        data=np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32),
        columns=frame.columns,
        index=frame.index,
    )
    masked.loc[:, owned_columns] = frame.loc[:, owned_columns].to_numpy(dtype=np.float32)
    return masked


def partition_map(partitions: list[ClientPartition]) -> dict[str, list[str]]:
    return {f"client_{part.client_id}": part.owned_columns for part in partitions}


def validate_vertical_partitions(
    *,
    partitions: list[ClientPartition],
    feature_names: list[str],
) -> None:
    expected = set(feature_names)
    seen: set[str] = set()

    for partition in partitions:
        owned = set(partition.owned_columns)
        overlap = owned.intersection(seen)
        if overlap:
            raise ValueError(f"Partition overlap detected for features: {sorted(overlap)}")
        seen.update(owned)

    missing = expected.difference(seen)
    extra = seen.difference(expected)

    if missing:
        raise ValueError(f"Missing features from partitions: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected features in partitions: {sorted(extra)}")


def build_partition_metadata(
    *,
    partitions: list[ClientPartition],
    feature_names: list[str],
    num_train_rows: int,
    num_eval_rows: int,
) -> dict[str, Any]:
    validate_vertical_partitions(partitions=partitions, feature_names=feature_names)

    client_to_columns = partition_map(partitions)
    feature_owner_map = {
        column: client_key
        for client_key, columns in client_to_columns.items()
        for column in columns
    }

    return {
        "num_clients": len(partitions),
        "num_features": len(feature_names),
        "row_index_policy": "shared_all_rows",
        "shared_row_indices": {
            "train": {"start": 0, "end": max(num_train_rows - 1, -1), "count": num_train_rows},
            "eval": {"start": 0, "end": max(num_eval_rows - 1, -1), "count": num_eval_rows},
        },
        "client_feature_map": client_to_columns,
        "feature_owner_map": feature_owner_map,
    }


# ─────────────────────────── Horizontal partitioning ─────────────────────────


def build_horizontal_partitions(
    num_train_rows: int,
    num_clients: int = 3,
    *,
    random_state: int = 42,
    stratify_labels: np.ndarray | None = None,
) -> list[list[int]]:
    """Split training row indices into disjoint subsets — one per client.

    In horizontal FL each client owns a distinct set of *rows* but sees *all*
    features.  The returned lists of integer indices can be used to slice
    ``x_train`` / ``y_train`` inside a Flower client.

    Args:
        num_train_rows: Total number of training examples.
        num_clients: Number of FL clients (must be >= 2).
        random_state: RNG seed for reproducibility.
        stratify_labels: Optional 1-D label array.  When provided, positives and
            negatives are split independently before concatenation so that each
            client's shard has a similar class ratio.

    Returns:
        List of *num_clients* lists of row indices (ints), one per client.
    """
    if num_clients < 2:
        raise ValueError("num_clients must be >= 2")
    if num_train_rows < num_clients:
        raise ValueError(
            f"num_train_rows ({num_train_rows}) must be >= num_clients ({num_clients})"
        )

    rng = np.random.default_rng(random_state)
    indices = np.arange(num_train_rows)

    if stratify_labels is not None and len(np.unique(stratify_labels)) > 1:
        pos_idx = indices[stratify_labels > 0]
        neg_idx = indices[stratify_labels == 0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        pos_splits = np.array_split(pos_idx, num_clients)
        neg_splits = np.array_split(neg_idx, num_clients)
        return [
            sorted(list(pos_splits[i].tolist()) + list(neg_splits[i].tolist()))
            for i in range(num_clients)
        ]

    rng.shuffle(indices)
    return [split.tolist() for split in np.array_split(indices, num_clients)]


def build_horizontal_partition_metadata(
    *,
    num_clients: int,
    row_index_splits: list[list[int]],
    feature_names: list[str],
    num_eval_rows: int,
) -> dict[str, Any]:
    """Build metadata dict describing a horizontal partition assignment."""
    return {
        "num_clients": num_clients,
        "num_features": len(feature_names),
        "row_index_policy": "horizontal_disjoint_rows",
        "client_row_counts": {
            f"client_{i}": len(row_index_splits[i]) for i in range(num_clients)
        },
        "eval_row_count": num_eval_rows,
        "feature_names": feature_names,
    }
