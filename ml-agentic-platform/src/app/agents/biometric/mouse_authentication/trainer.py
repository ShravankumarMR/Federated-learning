from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from app.agents.biometric.mouse_authentication.config import MouseAuthenticationTrainingConfig
from app.agents.biometric.mouse_authentication.metrics import compute_authentication_metrics
from app.agents.biometric.mouse_authentication.model import MouseAuthenticationModel


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    eer: float
    eer_threshold: float


class MouseAuthenticationTrainer:
    def __init__(
        self,
        model: MouseAuthenticationModel,
        config: MouseAuthenticationTrainingConfig | None = None,
    ) -> None:
        self._config = config or MouseAuthenticationTrainingConfig()
        self._device = torch.device(self._config.device or _detect_device())
        self._model = model.to(self._device)
        self._criterion = nn.BCEWithLogitsLoss()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

    @property
    def model(self) -> MouseAuthenticationModel:
        return self._model

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
    ) -> dict[str, list[EpochMetrics]]:
        history = {"train": [], "validation": []}
        best_state = deepcopy(self._model.state_dict())
        best_validation_loss = float("inf")

        for _ in range(self._config.epochs):
            train_metrics = self._run_epoch(train_loader, training=True)
            validation_metrics = self._run_epoch(validation_loader, training=False)

            history["train"].append(train_metrics)
            history["validation"].append(validation_metrics)

            if validation_metrics.loss < best_validation_loss:
                best_validation_loss = validation_metrics.loss
                best_state = deepcopy(self._model.state_dict())

        self._model.load_state_dict(best_state)
        return history

    def evaluate(self, data_loader: torch.utils.data.DataLoader) -> EpochMetrics:
        return self._run_epoch(data_loader, training=False)

    def predict_probabilities(self, data_loader: torch.utils.data.DataLoader) -> np.ndarray:
        self._model.eval()
        probabilities: list[np.ndarray] = []

        with torch.inference_mode():
            for batch in data_loader:
                logits = self._forward_batch(batch)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())

        if not probabilities:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(probabilities).astype(np.float32)

    def predict_session_probability(
        self,
        timing_sequence: torch.Tensor,
        movement_sequence: torch.Tensor,
        length: int,
    ) -> float:
        batch = {
            "timing_sequence": timing_sequence.unsqueeze(0),
            "movement_sequence": movement_sequence.unsqueeze(0),
            "length": torch.tensor([length], dtype=torch.int64),
        }
        self._model.eval()
        with torch.inference_mode():
            logits = self._forward_batch(batch)
        return float(torch.sigmoid(logits)[0].cpu().item())

    def _run_epoch(
        self,
        data_loader: torch.utils.data.DataLoader,
        *,
        training: bool,
    ) -> EpochMetrics:
        if training:
            self._model.train()
        else:
            self._model.eval()

        total_loss = 0.0
        total_examples = 0
        labels_accumulator: list[np.ndarray] = []
        probability_accumulator: list[np.ndarray] = []

        context_manager = torch.enable_grad if training else torch.inference_mode
        with context_manager():
            for batch in data_loader:
                labels = batch["label"].to(self._device)
                logits = self._forward_batch(batch)
                loss = self._criterion(logits, labels)

                if training:
                    self._optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self._optimizer.step()

                batch_size = int(labels.shape[0])
                total_loss += float(loss.detach().cpu().item()) * batch_size
                total_examples += batch_size
                labels_accumulator.append(labels.detach().cpu().numpy())
                probability_accumulator.append(torch.sigmoid(logits).detach().cpu().numpy())

        if total_examples == 0:
            raise ValueError("Data loader produced no batches")

        labels_array = np.concatenate(labels_accumulator).astype(np.float32)
        probabilities = np.concatenate(probability_accumulator).astype(np.float32)
        metrics = compute_authentication_metrics(
            labels_array,
            probabilities,
            decision_threshold=self._config.decision_threshold,
        )
        return EpochMetrics(
            loss=total_loss / total_examples,
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1,
            eer=metrics.eer,
            eer_threshold=metrics.eer_threshold,
        )

    def _forward_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        timing_sequence = batch["timing_sequence"].to(self._device)
        movement_sequence = batch["movement_sequence"].to(self._device)
        lengths = batch["length"].to(self._device)
        return self._model(timing_sequence, movement_sequence, lengths)


def _detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"