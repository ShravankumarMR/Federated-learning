from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from app.agents.biometric.mouse_authentication.config import MouseSequenceFeatureConfig


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


@dataclass(frozen=True)
class MouseAuthenticationDataBundle:
    train_loader: DataLoader
    validation_loader: DataLoader
    train_size: int
    validation_size: int
    timing_scaler: FeatureScaler
    movement_scaler: FeatureScaler


class MouseAuthenticationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        timing_sequences: np.ndarray,
        movement_sequences: np.ndarray,
        lengths: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self._timing_sequences = torch.as_tensor(timing_sequences, dtype=torch.float32)
        self._movement_sequences = torch.as_tensor(movement_sequences, dtype=torch.float32)
        self._lengths = torch.as_tensor(lengths, dtype=torch.int64)
        self._labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "timing_sequence": self._timing_sequences[index],
            "movement_sequence": self._movement_sequences[index],
            "length": self._lengths[index],
            "label": self._labels[index],
        }


def load_mouse_events_frame(source: Path | str | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()

    source_path = Path(source)
    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source_path)
    if suffix == ".parquet":
        return pd.read_parquet(source_path)
    raise ValueError("Mouse events source must be a DataFrame, .csv, or .parquet file")


def create_mouse_authentication_dataloaders(
    events_source: Path | str | pd.DataFrame,
    target_user_id: str,
    feature_config: MouseSequenceFeatureConfig | None = None,
    *,
    batch_size: int = 32,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
    allowed_splits: tuple[str, ...] | None = None,
) -> MouseAuthenticationDataBundle:
    feature_config = feature_config or MouseSequenceFeatureConfig()
    events = load_mouse_events_frame(events_source)
    timing_sequences, movement_sequences, lengths, labels = prepare_mouse_authentication_arrays(
        events,
        target_user_id=target_user_id,
        feature_config=feature_config,
        allowed_splits=allowed_splits,
    )

    if len(labels) < 2:
        raise ValueError("At least two session samples are required to build dataloaders")
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("Authentication dataset must contain both legitimate and impostor sessions")

    indices = np.arange(len(labels))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=random_seed,
        stratify=labels,
    )

    timing_scaler = _fit_feature_scaler(timing_sequences[train_indices], lengths[train_indices])
    movement_scaler = _fit_feature_scaler(movement_sequences[train_indices], lengths[train_indices])

    scaled_timing = timing_sequences.copy()
    scaled_movement = movement_sequences.copy()
    _apply_scaler_in_place(scaled_timing, lengths, timing_scaler)
    _apply_scaler_in_place(scaled_movement, lengths, movement_scaler)

    train_dataset = MouseAuthenticationDataset(
        scaled_timing[train_indices],
        scaled_movement[train_indices],
        lengths[train_indices],
        labels[train_indices],
    )
    validation_dataset = MouseAuthenticationDataset(
        scaled_timing[validation_indices],
        scaled_movement[validation_indices],
        lengths[validation_indices],
        labels[validation_indices],
    )

    return MouseAuthenticationDataBundle(
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        validation_loader=DataLoader(validation_dataset, batch_size=batch_size, shuffle=False),
        train_size=len(train_dataset),
        validation_size=len(validation_dataset),
        timing_scaler=timing_scaler,
        movement_scaler=movement_scaler,
    )


def prepare_mouse_authentication_arrays(
    events: pd.DataFrame,
    *,
    target_user_id: str,
    feature_config: MouseSequenceFeatureConfig | None = None,
    allowed_splits: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_config = feature_config or MouseSequenceFeatureConfig()
    _validate_required_columns(events, feature_config)

    clean_events = events.copy()
    for column in feature_config.all_features + ("record_timestamp", "is_illegal"):
        if column in clean_events.columns:
            clean_events[column] = pd.to_numeric(clean_events[column], errors="coerce")

    if allowed_splits is not None:
        split_filter = {str(split).strip().lower() for split in allowed_splits}
        clean_events = clean_events[
            clean_events["split"].astype(str).str.lower().isin(split_filter)
        ].copy()
        if clean_events.empty:
            raise ValueError(
                f"No events left after applying split filter: {sorted(split_filter)}"
            )

    timing_sequences: list[np.ndarray] = []
    movement_sequences: list[np.ndarray] = []
    lengths: list[int] = []
    labels: list[int] = []

    for (_, user_id, _), session_frame in clean_events.groupby(
        ["split", "user_id", "session_id"], sort=False
    ):
        ordered_session = session_frame.sort_values("record_timestamp").reset_index(drop=True)
        label = _resolve_session_label(ordered_session, user_id, target_user_id)
        if label is None or len(ordered_session) < feature_config.min_sequence_length:
            continue

        timing_values = ordered_session.loc[:, feature_config.timing_features].fillna(0.0).to_numpy(
            dtype=np.float32
        )
        movement_values = ordered_session.loc[:, feature_config.movement_features].fillna(0.0).to_numpy(
            dtype=np.float32
        )

        padded_timing, length = _resample_or_pad_sequence(
            timing_values,
            max_length=feature_config.max_sequence_length,
        )
        padded_movement, _ = _resample_or_pad_sequence(
            movement_values,
            max_length=feature_config.max_sequence_length,
        )

        timing_sequences.append(padded_timing)
        movement_sequences.append(padded_movement)
        lengths.append(length)
        labels.append(label)

    if not labels:
        raise ValueError("No session sequences were generated from the supplied mouse events")

    return (
        np.stack(timing_sequences),
        np.stack(movement_sequences),
        np.asarray(lengths, dtype=np.int64),
        np.asarray(labels, dtype=np.float32),
    )


def _validate_required_columns(
    events: pd.DataFrame,
    feature_config: MouseSequenceFeatureConfig,
) -> None:
    required_columns = {
        "split",
        "user_id",
        "session_id",
        "record_timestamp",
        "is_illegal",
        *feature_config.all_features,
    }
    missing_columns = sorted(column for column in required_columns if column not in events.columns)
    if missing_columns:
        raise ValueError(f"Mouse events frame is missing required columns: {missing_columns}")


def _resolve_session_label(
    session_frame: pd.DataFrame,
    user_id: str,
    target_user_id: str,
) -> int | None:
    if user_id != target_user_id:
        return 0

    session_split = str(session_frame["split"].iloc[0]).lower()
    if session_split == "training":
        return 1

    illegal_value = session_frame["is_illegal"].dropna()
    if illegal_value.empty:
        return None
    return int(1 - int(illegal_value.max()))


def _resample_or_pad_sequence(values: np.ndarray, *, max_length: int) -> tuple[np.ndarray, int]:
    sequence_length, feature_count = values.shape
    if sequence_length >= max_length:
        sample_indices = np.linspace(0, sequence_length - 1, num=max_length, dtype=np.int64)
        return values[sample_indices], max_length

    padded = np.zeros((max_length, feature_count), dtype=np.float32)
    padded[:sequence_length] = values
    return padded, sequence_length


def _fit_feature_scaler(values: np.ndarray, lengths: np.ndarray) -> FeatureScaler:
    masks = np.arange(values.shape[1])[None, :] < lengths[:, None]
    valid_rows = values[masks]
    mean = valid_rows.mean(axis=0)
    std = valid_rows.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return FeatureScaler(mean=mean.astype(np.float32), std=std.astype(np.float32))


def _apply_scaler_in_place(values: np.ndarray, lengths: np.ndarray, scaler: FeatureScaler) -> None:
    for index, length in enumerate(lengths):
        values[index, :length] = scaler.transform(values[index, :length]).astype(np.float32)