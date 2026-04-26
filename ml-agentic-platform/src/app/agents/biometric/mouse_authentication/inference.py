"""Checkpoint-backed inference for MouseAuthenticationModel.

This module provides two public functions:

* :func:`load_checkpoint` — deserialise a ``.pt`` file produced by the
  biometric training CLI, reconstruct the model, and cache it in process
  memory so repeated calls pay only the disk-read cost once per file.

* :func:`score_sequence` — normalise raw timing / movement feature arrays
  with stored scaler statistics, run a forward pass, and return a scored
  result dict suitable for consumption by :class:`BiometricAgent`.

Integration note
----------------
These helpers are intentionally decoupled from the data pipeline.  The caller
is responsible for supplying pre-extracted timing and movement arrays; internal
data-loading code is not imported here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.agents.biometric.mouse_authentication.config import (
    MouseAuthenticationModelConfig,
    MouseSequenceFeatureConfig,
)
from app.agents.biometric.mouse_authentication.model import MouseAuthenticationModel

_logger = logging.getLogger(__name__)

# Module-level in-process cache: resolved absolute path str -> _CheckpointArtifact
_CHECKPOINT_CACHE: dict[str, "_CheckpointArtifact"] = {}


class _CheckpointArtifact:
    """Holds a loaded :class:`MouseAuthenticationModel` together with its
    normalization statistics and decision threshold.
    """

    __slots__ = (
        "model",
        "timing_mean",
        "timing_std",
        "movement_mean",
        "movement_std",
        "feature_config",
        "target_user_id",
        "eer_threshold",
    )

    def __init__(
        self,
        *,
        model: MouseAuthenticationModel,
        timing_mean: np.ndarray,
        timing_std: np.ndarray,
        movement_mean: np.ndarray,
        movement_std: np.ndarray,
        feature_config: MouseSequenceFeatureConfig,
        target_user_id: str,
        eer_threshold: float,
    ) -> None:
        self.model = model
        self.timing_mean = timing_mean
        self.timing_std = timing_std
        self.movement_mean = movement_mean
        self.movement_std = movement_std
        self.feature_config = feature_config
        self.target_user_id = target_user_id
        self.eer_threshold = eer_threshold


def load_checkpoint(checkpoint_path: Path) -> _CheckpointArtifact:
    """Load and cache a MouseAuthentication checkpoint from *checkpoint_path*.

    The result is cached in process memory keyed by the resolved absolute path,
    so subsequent calls with the same path are instant.

    Raises:
        FileNotFoundError: if the ``.pt`` file does not exist at *checkpoint_path*.
    """
    cache_key = str(checkpoint_path.resolve())
    if cache_key in _CHECKPOINT_CACHE:
        return _CHECKPOINT_CACHE[cache_key]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Biometric checkpoint not found: {checkpoint_path}")

    _logger.info("Loading biometric checkpoint: %s", checkpoint_path)
    # weights_only=False is required because the checkpoint contains numpy arrays
    # and Python dataclass dicts alongside the model state dict.
    raw: dict[str, Any] = torch.load(  # noqa: S614
        checkpoint_path, map_location="cpu", weights_only=False
    )

    feature_cfg_dict: dict[str, Any] = raw["feature_config"]
    model_cfg_dict: dict[str, Any] = raw["model_config"]

    feature_config = MouseSequenceFeatureConfig(
        timing_features=tuple(feature_cfg_dict["timing_features"]),
        movement_features=tuple(feature_cfg_dict["movement_features"]),
        max_sequence_length=int(feature_cfg_dict["max_sequence_length"]),
        min_sequence_length=int(feature_cfg_dict["min_sequence_length"]),
    )
    model_config = MouseAuthenticationModelConfig(
        lstm_hidden_size=int(model_cfg_dict["lstm_hidden_size"]),
        lstm_layers=int(model_cfg_dict["lstm_layers"]),
        cnn_channels=tuple(int(c) for c in model_cfg_dict["cnn_channels"]),
        projection_dim=int(model_cfg_dict["projection_dim"]),
        dropout=float(model_cfg_dict["dropout"]),
    )

    model = MouseAuthenticationModel(
        timing_input_dim=len(feature_config.timing_features),
        movement_input_dim=len(feature_config.movement_features),
        config=model_config,
    )
    model.load_state_dict(raw["model_state_dict"])
    model.eval()

    val_metrics: dict[str, Any] = raw.get("validation_metrics", {})
    eer_threshold = float(val_metrics.get("eer_threshold", 0.5))

    artifact = _CheckpointArtifact(
        model=model,
        timing_mean=np.asarray(raw["timing_scaler_mean"], dtype=np.float32),
        timing_std=np.asarray(raw["timing_scaler_std"], dtype=np.float32),
        movement_mean=np.asarray(raw["movement_scaler_mean"], dtype=np.float32),
        movement_std=np.asarray(raw["movement_scaler_std"], dtype=np.float32),
        feature_config=feature_config,
        target_user_id=str(raw.get("target_user_id", "")),
        eer_threshold=eer_threshold,
    )
    _CHECKPOINT_CACHE[cache_key] = artifact
    _logger.info(
        "Checkpoint ready  user=%s  eer_threshold=%.3f",
        artifact.target_user_id,
        eer_threshold,
    )
    return artifact


def score_sequence(
    artifact: _CheckpointArtifact,
    timing_sequence: list[list[float]],
    movement_sequence: list[list[float]],
    length: int | None = None,
) -> dict[str, float]:
    """Run inference for a single session against a loaded *artifact*.

    The raw feature arrays are Z-score normalised using the scaler statistics
    stored in the checkpoint before the forward pass.

    Args:
        artifact: Loaded checkpoint produced by :func:`load_checkpoint`.
        timing_sequence: ``[T, n_timing_features]`` list-of-lists (raw, un-normalised).
        movement_sequence: ``[T, n_movement_features]`` list-of-lists (raw, un-normalised).
        length: Effective sequence length; defaults to ``len(timing_sequence)``.

    Returns:
        Dict with keys:

        * ``score`` — probability of genuine user (higher = more likely authentic).
        * ``risk`` — ``1 - score``.
        * ``confidence`` — absolute distance from the EER decision threshold.
        * ``eer_threshold`` — stored per-user EER threshold from training.
    """
    if not timing_sequence or not movement_sequence:
        return {
            "score": 0.5,
            "risk": 0.5,
            "confidence": 0.0,
            "eer_threshold": artifact.eer_threshold,
        }

    max_len = artifact.feature_config.max_sequence_length
    seq_len = length if length is not None else len(timing_sequence)

    timing_arr = np.array(timing_sequence[:max_len], dtype=np.float32)
    movement_arr = np.array(movement_sequence[:max_len], dtype=np.float32)
    effective_len = min(seq_len, max_len)

    # Z-score normalise — guard against zero std to avoid division by zero
    safe_timing_std = np.where(artifact.timing_std > 0, artifact.timing_std, 1.0)
    safe_movement_std = np.where(artifact.movement_std > 0, artifact.movement_std, 1.0)
    timing_arr = (timing_arr - artifact.timing_mean) / safe_timing_std
    movement_arr = (movement_arr - artifact.movement_mean) / safe_movement_std

    # Pad to max_sequence_length so LSTM pack_padded_sequence is well-formed
    pad_len = max_len - timing_arr.shape[0]
    if pad_len > 0:
        timing_arr = np.pad(timing_arr, ((0, pad_len), (0, 0)))
        movement_arr = np.pad(movement_arr, ((0, pad_len), (0, 0)))

    timing_tensor = torch.from_numpy(timing_arr).unsqueeze(0)      # (1, T, F_t)
    movement_tensor = torch.from_numpy(movement_arr).unsqueeze(0)  # (1, T, F_m)
    length_tensor = torch.tensor([effective_len], dtype=torch.int64)

    with torch.inference_mode():
        logit = artifact.model(timing_tensor, movement_tensor, length_tensor)

    probability = float(torch.sigmoid(logit).squeeze().item())
    confidence = abs(probability - artifact.eer_threshold)

    return {
        "score": round(probability, 4),
        "risk": round(1.0 - probability, 4),
        "confidence": round(confidence, 4),
        "eer_threshold": round(artifact.eer_threshold, 4),
    }
