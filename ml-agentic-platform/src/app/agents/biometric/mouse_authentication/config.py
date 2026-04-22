from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MouseSequenceFeatureConfig:
    timing_features: tuple[str, ...] = (
        "dt",
        "button_event",
        "drag_event",
        "pressed_event",
        "released_event",
    )
    movement_features: tuple[str, ...] = (
        "x",
        "y",
        "dx",
        "dy",
        "distance",
        "speed",
    )
    max_sequence_length: int = 256
    min_sequence_length: int = 8

    @property
    def all_features(self) -> tuple[str, ...]:
        return self.timing_features + self.movement_features


@dataclass(frozen=True)
class MouseAuthenticationModelConfig:
    lstm_hidden_size: int = 64
    lstm_layers: int = 2
    cnn_channels: tuple[int, ...] = (32, 64)
    projection_dim: int = 64
    dropout: float = 0.2


@dataclass(frozen=True)
class MouseAuthenticationTrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    validation_fraction: float = 0.2
    random_seed: int = 42
    decision_threshold: float = 0.5
    device: str | None = None