from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TemporalDatasetConfig:
    dataset_name: str
    source_col: str = "source_account"
    target_col: str = "target_account"
    time_col: str = "timestamp"
    label_col: str = "isFraud"
    numeric_edge_cols: list[str] = field(default_factory=list)
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    random_seed: int = 42


@dataclass(frozen=True)
class TemporalModelConfig:
    backbone: str = "gat"
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    time_encoding_dim: int = 16
    gat_heads: int = 4


@dataclass(frozen=True)
class TemporalTrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 80
    patience: int = 15
    gradient_clip_norm: float = 1.0
    device: str | None = None
    log_interval: int = 1  # log every N epochs; 1 = every epoch
