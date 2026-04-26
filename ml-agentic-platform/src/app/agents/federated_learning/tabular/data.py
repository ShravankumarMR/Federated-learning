from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class TabularFederatedDataset:
    x_train: pd.DataFrame
    y_train: pd.Series
    x_eval: pd.DataFrame
    y_eval: pd.Series
    feature_names: list[str]


def load_ieee_cis_dataset(
    *,
    features_dir: Path | None = None,
    processed_dir: Path | None = None,
    random_state: int = 42,
    eval_size: float = 0.2,
) -> TabularFederatedDataset:
    feature_root = features_dir or Path("data/features/ieee_cis_fraud")
    processed_root = processed_dir or Path("data/processed/ieee_cis_fraud")

    x_train_path = feature_root / "X_train.csv"
    y_train_path = feature_root / "y_train.csv"
    x_test_path = feature_root / "X_test.csv"
    y_test_path = feature_root / "y_test.csv"

    train_processed_path = processed_root / "train_processed.csv"
    test_processed_path = processed_root / "test_processed.csv"

    if not x_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError(
            "Required training files not found under "
            f"{feature_root}. Expected X_train.csv and y_train.csv."
        )

    x_train_frame = pd.read_csv(x_train_path)
    y_train_frame = pd.read_csv(y_train_path)

    if y_train_frame.empty:
        raise ValueError("y_train.csv is empty")

    y_column = y_train_frame.columns[0]
    y_train_series = pd.to_numeric(y_train_frame[y_column], errors="coerce").fillna(0)
    y_train_series = (y_train_series > 0).astype(int)

    x_train_frame = x_train_frame.reset_index(drop=True)
    y_train_series = y_train_series.reset_index(drop=True)

    x_eval_frame, y_eval_series = _try_load_external_eval_split(
        x_test_path=x_test_path,
        y_test_path=y_test_path,
        test_processed_path=test_processed_path,
    )

    if x_eval_frame is None or y_eval_series is None:
        x_train_frame, x_eval_frame, y_train_series, y_eval_series = train_test_split(
            x_train_frame,
            y_train_series,
            test_size=eval_size,
            random_state=random_state,
            stratify=y_train_series if y_train_series.nunique() > 1 else None,
        )
        x_train_frame = x_train_frame.reset_index(drop=True)
        x_eval_frame = x_eval_frame.reset_index(drop=True)
        y_train_series = y_train_series.reset_index(drop=True)
        y_eval_series = y_eval_series.reset_index(drop=True)

    x_train_frame, x_eval_frame = _prepare_feature_matrices(
        x_train_frame=x_train_frame,
        x_eval_frame=x_eval_frame,
    )

    if list(x_train_frame.columns) != list(x_eval_frame.columns):
        x_eval_frame = x_eval_frame.reindex(columns=x_train_frame.columns)

    return TabularFederatedDataset(
        x_train=x_train_frame,
        y_train=y_train_series,
        x_eval=x_eval_frame,
        y_eval=y_eval_series,
        feature_names=list(x_train_frame.columns),
    )


def _try_load_external_eval_split(
    *,
    x_test_path: Path,
    y_test_path: Path,
    test_processed_path: Path,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    if not x_test_path.exists():
        return None, None

    x_eval = pd.read_csv(x_test_path).reset_index(drop=True)

    if y_test_path.exists():
        y_eval_frame = pd.read_csv(y_test_path)
        if y_eval_frame.empty:
            raise ValueError("y_test.csv exists but is empty")
        y_column = y_eval_frame.columns[0]
        y_eval = pd.to_numeric(y_eval_frame[y_column], errors="coerce").fillna(0)
        return x_eval, (y_eval > 0).astype(int).reset_index(drop=True)

    if not test_processed_path.exists():
        return None, None

    processed_test = pd.read_csv(test_processed_path)
    if "isFraud" not in processed_test.columns:
        return None, None

    y_eval = pd.to_numeric(processed_test["isFraud"], errors="coerce").fillna(0)
    y_eval = (y_eval > 0).astype(int).reset_index(drop=True)

    if len(y_eval) == len(x_eval):
        return x_eval, y_eval

    if "TransactionID" in processed_test.columns and "TransactionID" in x_eval.columns:
        id_to_label = dict(zip(processed_test["TransactionID"], y_eval, strict=False))
        aligned = x_eval["TransactionID"].map(id_to_label)
        if aligned.notna().all():
            return x_eval, aligned.astype(int).reset_index(drop=True)

    return None, None


def _prepare_feature_matrices(
    *,
    x_train_frame: pd.DataFrame,
    x_eval_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_count = x_train_frame.shape[0]

    merged = pd.concat([x_train_frame, x_eval_frame], axis=0, ignore_index=True)
    merged = pd.get_dummies(merged, dummy_na=True)
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.fillna(0.0).astype(np.float32)

    train_encoded = merged.iloc[:train_count].reset_index(drop=True)
    eval_encoded = merged.iloc[train_count:].reset_index(drop=True)
    return train_encoded, eval_encoded


def load_paysim_dataset(
    *,
    features_dir: Path | None = None,
    processed_dir: Path | None = None,
    random_state: int = 42,
    eval_size: float = 0.2,
) -> TabularFederatedDataset:
    """Load PaySim tabular federated dataset.

    Resolution order:

    1. ``features_dir/X_train.csv`` + ``features_dir/y_train.csv``
       (same layout as the IEEE-CIS feature pipeline output).
    2. ``processed_dir/train_processed.csv`` with an ``isFraud`` / ``is_fraud`` label
       column — useful when the feature-engineering step has not been run yet.

    An internal 80/20 stratified train/eval split is always applied because
    PaySim does not have a separate held-out test file in the current pipeline.
    """
    feature_root = features_dir or Path("data/features/paysim")
    processed_root = processed_dir or Path("data/processed/paysim")

    x_train_path = feature_root / "X_train.csv"
    y_train_path = feature_root / "y_train.csv"

    if x_train_path.exists() and y_train_path.exists():
        x_train_frame = pd.read_csv(x_train_path).reset_index(drop=True)
        y_train_frame = pd.read_csv(y_train_path)
        if y_train_frame.empty:
            raise ValueError(f"y_train.csv is empty: {y_train_path}")
        y_column = y_train_frame.columns[0]
        y_train_series = (
            pd.to_numeric(y_train_frame[y_column], errors="coerce").fillna(0) > 0
        ).astype(int).reset_index(drop=True)
    else:
        processed_train_path = processed_root / "train_processed.csv"
        if not processed_train_path.exists():
            raise FileNotFoundError(
                "PaySim training data not found. "
                f"Expected {x_train_path} + {y_train_path} "
                f"or {processed_train_path}."
            )
        raw_frame = pd.read_csv(processed_train_path)
        label_col = next(
            (c for c in ("isFraud", "is_fraud") if c in raw_frame.columns), None
        )
        if label_col is None:
            raise ValueError(
                f"No label column (isFraud / is_fraud) found in {processed_train_path}"
            )
        y_train_series = (
            pd.to_numeric(raw_frame[label_col], errors="coerce").fillna(0) > 0
        ).astype(int).reset_index(drop=True)
        x_train_frame = raw_frame.drop(columns=[label_col]).reset_index(drop=True)

    x_train_frame, x_eval_frame, y_train_series, y_eval_series = train_test_split(
        x_train_frame,
        y_train_series,
        test_size=eval_size,
        random_state=random_state,
        stratify=y_train_series if y_train_series.nunique() > 1 else None,
    )
    x_train_frame = x_train_frame.reset_index(drop=True)
    x_eval_frame = x_eval_frame.reset_index(drop=True)
    y_train_series = y_train_series.reset_index(drop=True)
    y_eval_series = y_eval_series.reset_index(drop=True)

    x_train_frame, x_eval_frame = _prepare_feature_matrices(
        x_train_frame=x_train_frame,
        x_eval_frame=x_eval_frame,
    )

    return TabularFederatedDataset(
        x_train=x_train_frame,
        y_train=y_train_series,
        x_eval=x_eval_frame,
        y_eval=y_eval_series,
        feature_names=list(x_train_frame.columns),
    )
