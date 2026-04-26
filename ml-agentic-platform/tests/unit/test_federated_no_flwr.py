"""Tests for federated learning modules that do NOT require flwr.

Covers:
- Horizontal partitioning (build_horizontal_partitions, build_horizontal_partition_metadata)
- PaySim dataset loader (load_paysim_dataset)

These tests run in any environment where torch + pandas are available, making
them suitable for fast CI runs where flwr is not installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.agents.federated_learning.tabular.data import load_paysim_dataset
from app.agents.federated_learning.tabular.partitioning import (
    build_horizontal_partition_metadata,
    build_horizontal_partitions,
)


# ─────────────────────── Horizontal partitioning ─────────────────────────────


def test_horizontal_partitions_are_disjoint_and_cover_all_rows() -> None:
    num_rows = 30
    num_clients = 3
    splits = build_horizontal_partitions(num_rows, num_clients, random_state=7)

    assert len(splits) == num_clients
    all_indices = sorted(idx for split in splits for idx in split)
    assert all_indices == list(range(num_rows))

    for i, split_i in enumerate(splits):
        for j, split_j in enumerate(splits):
            if i != j:
                assert not set(split_i).intersection(split_j), "Splits must be disjoint"


def test_horizontal_partitions_are_stratified_when_labels_provided() -> None:
    num_rows = 60
    labels = np.array([1] * 6 + [0] * 54, dtype=np.float32)  # 10 % positive

    splits = build_horizontal_partitions(
        num_rows, num_clients=3, random_state=0, stratify_labels=labels
    )
    for split in splits:
        split_labels = labels[split]
        positive_fraction = float(split_labels.mean())
        # Each shard should have roughly ~10% positives (allow ±10 pp tolerance)
        assert 0.0 <= positive_fraction <= 0.20, (
            f"Expected ~10% positives, got {positive_fraction:.2%}"
        )


def test_horizontal_partition_metadata_structure() -> None:
    splits = build_horizontal_partitions(20, num_clients=2, random_state=1)
    feature_names = ["a", "b", "c"]
    metadata = build_horizontal_partition_metadata(
        num_clients=2,
        row_index_splits=splits,
        feature_names=feature_names,
        num_eval_rows=5,
    )
    assert metadata["row_index_policy"] == "horizontal_disjoint_rows"
    assert metadata["num_clients"] == 2
    assert metadata["num_features"] == 3
    assert metadata["eval_row_count"] == 5
    assert sum(metadata["client_row_counts"].values()) == 20


def test_horizontal_partitions_raise_on_too_few_rows() -> None:
    with pytest.raises(ValueError, match="num_train_rows"):
        build_horizontal_partitions(num_train_rows=1, num_clients=3)


def test_horizontal_partitions_raise_on_single_client() -> None:
    with pytest.raises(ValueError, match="num_clients"):
        build_horizontal_partitions(num_train_rows=10, num_clients=1)


# ─────────────────────────── PaySim data loader ───────────────────────────────


def test_load_paysim_dataset_from_features_dir(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n = 40
    features = pd.DataFrame(
        rng.normal(size=(n, 4)).astype(np.float32),
        columns=["f0", "f1", "f2", "f3"],
    )
    labels = pd.DataFrame({"isFraud": (rng.random(n) > 0.8).astype(int)})
    features.to_csv(tmp_path / "X_train.csv", index=False)
    labels.to_csv(tmp_path / "y_train.csv", index=False)

    dataset = load_paysim_dataset(features_dir=tmp_path, random_state=0)

    assert dataset.x_train.shape[1] == 4
    assert len(dataset.y_train) + len(dataset.y_eval) == n
    assert set(dataset.y_train.unique()).issubset({0, 1})
    assert dataset.feature_names == ["f0", "f1", "f2", "f3"]


def test_load_paysim_dataset_from_processed_dir(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    n = 50
    raw = pd.DataFrame(
        {
            "f0": rng.normal(size=n).astype(np.float32),
            "f1": rng.normal(size=n).astype(np.float32),
            "isFraud": (rng.random(n) > 0.9).astype(int),
        }
    )
    raw.to_csv(tmp_path / "train_processed.csv", index=False)

    dataset = load_paysim_dataset(processed_dir=tmp_path, random_state=0)

    assert "isFraud" not in dataset.x_train.columns
    assert len(dataset.y_train) + len(dataset.y_eval) == n


def test_load_paysim_dataset_raises_when_no_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PaySim"):
        load_paysim_dataset(features_dir=tmp_path, processed_dir=tmp_path)


def test_load_paysim_dataset_train_eval_shapes_consistent(tmp_path: Path) -> None:
    rng = np.random.default_rng(99)
    n = 100
    features = pd.DataFrame(
        rng.normal(size=(n, 5)).astype(np.float32),
        columns=[f"feat_{i}" for i in range(5)],
    )
    labels = pd.DataFrame({"isFraud": (rng.random(n) > 0.85).astype(int)})
    features.to_csv(tmp_path / "X_train.csv", index=False)
    labels.to_csv(tmp_path / "y_train.csv", index=False)

    dataset = load_paysim_dataset(features_dir=tmp_path, eval_size=0.25, random_state=0)

    # Eval should be ~25% of total rows
    total = len(dataset.y_train) + len(dataset.y_eval)
    assert total == n
    assert dataset.x_train.shape[0] == len(dataset.y_train)
    assert dataset.x_eval.shape[0] == len(dataset.y_eval)
    assert dataset.x_train.shape[1] == dataset.x_eval.shape[1]
