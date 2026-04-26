from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class LocalTrainingConfig:
    epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"


@dataclass(frozen=True)
class ModelConfig:
    model_type: Literal["logistic", "nn"] = "logistic"
    hidden_dim: int = 64


class LogisticFraudModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class ShallowFraudModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def create_model(*, input_dim: int, config: ModelConfig) -> nn.Module:
    if config.model_type == "logistic":
        return LogisticFraudModel(input_dim=input_dim)
    if config.model_type == "nn":
        return ShallowFraudModel(input_dim=input_dim, hidden_dim=config.hidden_dim)
    raise ValueError(f"Unsupported model_type: {config.model_type}")


def model_to_ndarrays(model: nn.Module) -> list[np.ndarray]:
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def ndarrays_to_model(model: nn.Module, parameters: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError("Parameter shape mismatch when loading model parameters")

    state_dict = {
        key: torch.tensor(array, dtype=model.state_dict()[key].dtype)
        for key, array in zip(keys, parameters)
    }
    model.load_state_dict(state_dict, strict=True)


def train_local_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    config: LocalTrainingConfig,
) -> dict[str, float]:
    model.train()
    device = torch.device(config.device)
    model.to(device)

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    positive_count = float(np.sum(y_train > 0))
    negative_count = float(np.sum(y_train <= 0))
    pos_weight_value = negative_count / max(positive_count, 1.0)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    epoch_losses: list[float] = []
    for _ in range(config.epochs):
        running_loss = 0.0
        sample_count = 0
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = int(labels.shape[0])
            running_loss += float(loss.detach().cpu().item()) * batch_size
            sample_count += batch_size

        epoch_losses.append(running_loss / max(sample_count, 1))

    return {"train_loss": float(epoch_losses[-1]) if epoch_losses else 0.0}


def evaluate_model(model: nn.Module, x_eval: np.ndarray, y_eval: np.ndarray) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device

    with torch.inference_mode():
        features = torch.tensor(x_eval, dtype=torch.float32, device=device)
        labels = torch.tensor(y_eval, dtype=torch.float32, device=device)
        logits = model(features)
        loss = nn.BCEWithLogitsLoss()(logits, labels).detach().cpu().item()
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()

    auc_pr = _safe_auc_pr(y_eval.astype(np.float32), probabilities.astype(np.float32))
    return {
        "loss": float(loss),
        "auc_pr": float(auc_pr),
        "accuracy": float(np.mean((probabilities >= 0.5).astype(np.int32) == y_eval.astype(np.int32))),
    }


def predict_probabilities(model: nn.Module, x_values: np.ndarray) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        features = torch.tensor(x_values, dtype=torch.float32, device=device)
        logits = model(features)
        return torch.sigmoid(logits).detach().cpu().numpy()


def _safe_auc_pr(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return 0.0
    return float(average_precision_score(labels, probabilities))
