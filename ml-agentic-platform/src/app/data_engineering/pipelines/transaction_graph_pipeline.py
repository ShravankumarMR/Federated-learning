from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.data_engineering.features.transaction_graph import (
    GraphBuildConfig,
    build_pyg_data,
    load_ieee_cis_transactions,
    load_paysim_transactions,
    preprocess_transactions,
)

if TYPE_CHECKING:
    from torch_geometric.data import Data
else:
    Data = Any


def build_ieee_cis_graph(csv_path: Path) -> Data:
    """Build a PyG transaction graph from the IEEE-CIS transaction CSV."""
    frame = load_ieee_cis_transactions(csv_path)
    processed = preprocess_transactions(
        frame,
        source_col="source_account",
        target_col="target_account",
        time_col="timestamp",
        numeric_cols=["TransactionAmt"],
    )
    config = GraphBuildConfig(
        source_col="source_account",
        target_col="target_account",
        time_col="timestamp",
        numeric_edge_cols=["TransactionAmt"],
        label_col="isFraud",
    )
    return build_pyg_data(processed, config)


def build_paysim_graph(csv_path: Path) -> Data:
    """Build a PyG transaction graph from the PaySim transaction CSV."""
    frame = load_paysim_transactions(csv_path)
    processed = preprocess_transactions(
        frame,
        source_col="source_account",
        target_col="target_account",
        time_col="timestamp",
        numeric_cols=[
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ],
    )
    config = GraphBuildConfig(
        source_col="source_account",
        target_col="target_account",
        time_col="timestamp",
        numeric_edge_cols=[
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
        ],
        label_col="isFraud",
    )
    return build_pyg_data(processed, config)


def build_transaction_graphs(ieee_csv_path: Path, paysim_csv_path: Path) -> dict[str, Data]:
    """Build one transaction graph per dataset and return them keyed by dataset name."""
    return {
        "ieee_cis": build_ieee_cis_graph(ieee_csv_path),
        "paysim": build_paysim_graph(paysim_csv_path),
    }
