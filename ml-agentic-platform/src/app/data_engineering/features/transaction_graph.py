from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch_geometric.data import Data
except ImportError as exc:  # pragma: no cover - exercised only when deps are missing.
    torch = None
    Data = Any
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class GraphBuildConfig:
    """Configuration for converting transaction tables into graph objects."""

    source_col: str
    target_col: str
    time_col: str
    numeric_edge_cols: list[str]
    label_col: str | None = None


def _require_torch_geometric() -> None:
    """Ensure torch and torch_geometric are available before graph construction."""
    if _TORCH_IMPORT_ERROR is not None:
        raise ImportError(
            "torch and torch-geometric are required for graph construction. "
            "Install project dependencies before running this pipeline."
        ) from _TORCH_IMPORT_ERROR


def load_ieee_cis_transactions(csv_path: Path) -> pd.DataFrame:
    """Load IEEE-CIS transaction CSV and derive graph endpoint account identifiers."""
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError("IEEE-CIS transaction file is empty.")

    source_parts = [
        _safe_text_col(frame, "card1"),
        _safe_text_col(frame, "card2"),
        _safe_text_col(frame, "card3"),
        _safe_text_col(frame, "card5"),
        _safe_text_col(frame, "addr1"),
        _safe_text_col(frame, "P_emaildomain"),
    ]
    source_account = _combine_parts(source_parts, prefix="src")

    target_parts = [
        _safe_text_col(frame, "card4"),
        _safe_text_col(frame, "card6"),
        _safe_text_col(frame, "addr2"),
        _safe_text_col(frame, "R_emaildomain"),
        _safe_text_col(frame, "M4"),
    ]
    target_account = _combine_parts(target_parts, prefix="dst")

    product_cd = _safe_text_col(frame, "ProductCD")
    fallback_target = "merchant_" + product_cd
    target_account = np.where(target_account == "dst_unknown", fallback_target, target_account)

    transformed = frame.copy()
    transformed["source_account"] = source_account
    transformed["target_account"] = target_account
    transformed["timestamp"] = transformed.get("TransactionDT", pd.Series([np.nan] * len(transformed)))
    return transformed


def load_paysim_transactions(csv_path: Path) -> pd.DataFrame:
    """Load PaySim transaction CSV and map endpoint columns to a common schema."""
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError("PaySim transaction file is empty.")

    transformed = frame.copy()
    transformed["source_account"] = _safe_text_col(transformed, "nameOrig")
    transformed["target_account"] = _safe_text_col(transformed, "nameDest")
    transformed["timestamp"] = transformed.get("step", pd.Series([np.nan] * len(transformed)))
    return transformed


def preprocess_transactions(
    frame: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    time_col: str,
    numeric_cols: list[str],
) -> pd.DataFrame:
    """Handle missing values and normalize numeric columns for transaction records."""
    _validate_required_columns(frame, [source_col, target_col, time_col])

    transformed = frame.copy()
    transformed[source_col] = transformed[source_col].fillna("unknown_account").astype(str)
    transformed[target_col] = transformed[target_col].fillna("unknown_account").astype(str)

    transformed[time_col] = pd.to_numeric(transformed[time_col], errors="coerce")
    transformed[time_col] = transformed[time_col].fillna(transformed[time_col].median())

    usable_numeric = [col for col in numeric_cols if col in transformed.columns]
    if usable_numeric:
        numeric_frame = transformed[usable_numeric].apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        imputed = imputer.fit_transform(numeric_frame)
        transformed[usable_numeric] = scaler.fit_transform(imputed)

    return transformed


def build_pyg_data(frame: pd.DataFrame, config: GraphBuildConfig) -> Data:
    """Create a torch_geometric Data object from preprocessed transaction records."""
    _require_torch_geometric()
    _validate_required_columns(
        frame,
        [config.source_col, config.target_col, config.time_col, *config.numeric_edge_cols],
    )

    unique_accounts = pd.Index(frame[config.source_col]).append(pd.Index(frame[config.target_col])).unique()
    account_to_id = {account: idx for idx, account in enumerate(unique_accounts)}

    src_ids = frame[config.source_col].map(account_to_id).to_numpy()
    dst_ids = frame[config.target_col].map(account_to_id).to_numpy()

    edge_index = torch.tensor(np.vstack([src_ids, dst_ids]), dtype=torch.long)
    edge_attr = torch.tensor(frame[config.numeric_edge_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    edge_time = torch.tensor(frame[config.time_col].to_numpy(dtype=np.int64), dtype=torch.long)

    in_degree = np.bincount(dst_ids, minlength=len(unique_accounts)).astype(np.float32)
    out_degree = np.bincount(src_ids, minlength=len(unique_accounts)).astype(np.float32)
    node_features = np.stack([in_degree, out_degree], axis=1)
    x = torch.tensor(node_features, dtype=torch.float32)

    data_kwargs: dict[str, Any] = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_time": edge_time,
        "num_nodes": len(unique_accounts),
    }

    if config.label_col and config.label_col in frame.columns:
        labels = pd.to_numeric(frame[config.label_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        data_kwargs["y"] = torch.tensor(labels, dtype=torch.long)

    return Data(**data_kwargs)


def _validate_required_columns(frame: pd.DataFrame, required_cols: list[str]) -> None:
    """Validate that all required columns are available in the input frame."""
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_text_col(frame: pd.DataFrame, col_name: str) -> pd.Series:
    """Return a string series for a column, defaulting to 'unknown' when absent."""
    if col_name not in frame.columns:
        return pd.Series(["unknown"] * len(frame), index=frame.index)
    return frame[col_name].fillna("unknown").astype(str)


def _combine_parts(parts: list[pd.Series], *, prefix: str) -> pd.Series:
    """Combine multiple string columns into a deterministic account identifier."""
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + "_" + part
    empty_mask = combined.str.replace("_", "", regex=False).str.len() == 0
    return combined.mask(empty_mask, f"{prefix}_unknown")
