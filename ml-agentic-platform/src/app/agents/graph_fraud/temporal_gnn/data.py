from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from app.agents.graph_fraud.temporal_gnn.config import TemporalDatasetConfig
from app.data_engineering.loaders import (
    load_ieee_cis_transactions,
    load_paysim_transactions,
    preprocess_transactions,
)


def build_temporal_node_classification_data(
    dataset_name: str,
    csv_path: Path,
    *,
    random_seed: int = 42,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> tuple[Data, TemporalDatasetConfig]:
    normalized_name = dataset_name.strip().lower()
    if normalized_name == "ieee_cis":
        config = TemporalDatasetConfig(
            dataset_name="ieee_cis",
            numeric_edge_cols=["TransactionAmt"],
            random_seed=random_seed,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
        raw_frame = load_ieee_cis_transactions(csv_path)
    elif normalized_name == "paysim":
        config = TemporalDatasetConfig(
            dataset_name="paysim",
            numeric_edge_cols=[
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
            ],
            random_seed=random_seed,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
        raw_frame = load_paysim_transactions(csv_path)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    frame = preprocess_transactions(
        raw_frame,
        source_col=config.source_col,
        target_col=config.target_col,
        time_col=config.time_col,
        numeric_cols=config.numeric_edge_cols,
    )
    data = _frame_to_temporal_node_data(frame, config)
    return data, config


def _frame_to_temporal_node_data(frame: pd.DataFrame, config: TemporalDatasetConfig) -> Data:
    required_cols = [
        config.source_col,
        config.target_col,
        config.time_col,
        config.label_col,
        *config.numeric_edge_cols,
    ]
    missing_cols = [col for col in required_cols if col not in frame.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for temporal graph: {missing_cols}")

    work_frame = frame.copy()
    work_frame[config.label_col] = pd.to_numeric(work_frame[config.label_col], errors="coerce").fillna(0)
    work_frame[config.label_col] = (work_frame[config.label_col] > 0).astype(np.int64)
    work_frame[config.time_col] = pd.to_numeric(work_frame[config.time_col], errors="coerce")
    work_frame[config.time_col] = work_frame[config.time_col].fillna(work_frame[config.time_col].median())

    unique_nodes = pd.Index(work_frame[config.source_col]).append(
        pd.Index(work_frame[config.target_col])
    ).unique()
    node_to_id = {node: index for index, node in enumerate(unique_nodes)}
    num_nodes = len(unique_nodes)

    src_ids = work_frame[config.source_col].map(node_to_id).to_numpy(dtype=np.int64)
    dst_ids = work_frame[config.target_col].map(node_to_id).to_numpy(dtype=np.int64)
    edge_index_np = np.vstack([src_ids, dst_ids])

    edge_time_raw = work_frame[config.time_col].to_numpy(dtype=np.float32)
    edge_time = _min_max_normalize(edge_time_raw)
    edge_attr = work_frame[config.numeric_edge_cols].to_numpy(dtype=np.float32)

    labels = np.zeros(num_nodes, dtype=np.int64)
    fraud_edges = work_frame[config.label_col].to_numpy(dtype=np.int64) == 1
    labels[src_ids[fraud_edges]] = 1
    labels[dst_ids[fraud_edges]] = 1

    node_first_seen = np.full(num_nodes, fill_value=np.inf, dtype=np.float64)
    np.minimum.at(node_first_seen, src_ids, edge_time_raw)
    np.minimum.at(node_first_seen, dst_ids, edge_time_raw)
    valid_times = node_first_seen[np.isfinite(node_first_seen)]
    median_time = float(np.median(valid_times)) if valid_times.size else 0.0
    node_first_seen[~np.isfinite(node_first_seen)] = median_time

    in_degree = np.bincount(dst_ids, minlength=num_nodes).astype(np.float32)
    out_degree = np.bincount(src_ids, minlength=num_nodes).astype(np.float32)
    avg_incoming_time = _aggregate_mean_per_node(dst_ids, edge_time, num_nodes)
    avg_outgoing_time = _aggregate_mean_per_node(src_ids, edge_time, num_nodes)
    first_seen_norm = _min_max_normalize(node_first_seen.astype(np.float32))
    total_degree = in_degree + out_degree

    node_features = np.stack(
        [
            in_degree,
            out_degree,
            total_degree,
            avg_incoming_time,
            avg_outgoing_time,
            first_seen_norm,
        ],
        axis=1,
    ).astype(np.float32)

    train_mask, validation_mask, test_mask = _build_temporal_masks(
        node_first_seen=node_first_seen,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        random_seed=config.random_seed,
    )

    return Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.float32),
        edge_index=torch.tensor(edge_index_np, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        edge_time=torch.tensor(edge_time, dtype=torch.float32),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(validation_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
        num_nodes=num_nodes,
    )


def _build_temporal_masks(
    *,
    node_first_seen: np.ndarray,
    train_ratio: float,
    validation_ratio: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0.0 <= validation_ratio < 1.0):
        raise ValueError("validation_ratio must be in [0, 1)")
    if (train_ratio + validation_ratio) >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be < 1")

    order = np.argsort(node_first_seen, kind="stable")
    num_nodes = node_first_seen.shape[0]
    train_end = max(1, int(num_nodes * train_ratio))
    validation_end = min(num_nodes, train_end + max(1, int(num_nodes * validation_ratio)))

    train_mask = np.zeros(num_nodes, dtype=bool)
    validation_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)

    train_mask[order[:train_end]] = True
    validation_mask[order[train_end:validation_end]] = True
    test_mask[order[validation_end:]] = True

    # Ensure all splits are non-empty for robust training/evaluation.
    if not np.any(validation_mask) or not np.any(test_mask):
        generator = np.random.default_rng(random_seed)
        shuffled = np.arange(num_nodes)
        generator.shuffle(shuffled)
        train_end = max(1, int(num_nodes * train_ratio))
        validation_end = min(num_nodes, train_end + max(1, int(num_nodes * validation_ratio)))
        train_mask[:] = False
        validation_mask[:] = False
        test_mask[:] = False
        train_mask[shuffled[:train_end]] = True
        validation_mask[shuffled[train_end:validation_end]] = True
        test_mask[shuffled[validation_end:]] = True

    return train_mask, validation_mask, test_mask


def _aggregate_mean_per_node(node_ids: np.ndarray, values: np.ndarray, num_nodes: int) -> np.ndarray:
    sums = np.zeros(num_nodes, dtype=np.float32)
    counts = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(sums, node_ids, values)
    np.add.at(counts, node_ids, 1.0)
    return np.divide(sums, np.maximum(counts, 1.0), dtype=np.float32)


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    denominator = max_value - min_value
    if denominator <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_value) / denominator).astype(np.float32)
