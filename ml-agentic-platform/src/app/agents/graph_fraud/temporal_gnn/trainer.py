from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch import nn

from app.agents.graph_fraud.temporal_gnn.config import TemporalTrainingConfig
from app.agents.graph_fraud.temporal_gnn.model import TemporalGraphModel

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochResult:
    loss: float
    auc_pr: float


def train_and_evaluate(
    model: TemporalGraphModel,
    data: Any,
    *,
    config: TemporalTrainingConfig | None = None,
) -> dict[str, Any]:
    training_config = config or TemporalTrainingConfig()
    device = torch.device(training_config.device or _auto_device())

    data = data.to(device)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    positive_count = float(torch.sum(data.y[data.train_mask] == 1.0).item())
    negative_count = float(torch.sum(data.y[data.train_mask] == 0.0).item())
    pos_weight_value = negative_count / max(positive_count, 1.0)

    _logger.info("Device        : %s", device)
    _logger.info("Parameters    : %s trainable", f"{num_params:,}")
    _logger.info(
        "Train split   : %d nodes  (pos=%d  neg=%d  pos_weight=%.2f)",
        int(positive_count + negative_count),
        int(positive_count),
        int(negative_count),
        pos_weight_value,
    )
    _logger.info(
        "Val / Test    : %d / %d nodes",
        int(data.val_mask.sum().item()),
        int(data.test_mask.sum().item()),
    )
    _logger.info(
        "Schedule      : epochs=%d  patience=%d  lr=%.5f  wd=%.5f  grad_clip=%.1f",
        training_config.epochs,
        training_config.patience,
        training_config.learning_rate,
        training_config.weight_decay,
        training_config.gradient_clip_norm,
    )
    _logger.info("-" * 70)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_validation_auc_pr = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    log_interval = training_config.log_interval

    for epoch in range(1, training_config.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_logits = model(data)
        train_loss = criterion(train_logits[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=training_config.gradient_clip_norm)
        optimizer.step()

        train_result = _evaluate_split(
            model=model,
            data=data,
            mask=data.train_mask,
            criterion=criterion,
        )
        validation_result = _evaluate_split(
            model=model,
            data=data,
            mask=data.val_mask,
            criterion=criterion,
        )
        epoch_secs = time.monotonic() - epoch_start

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_result.loss,
                "train_auc_pr": train_result.auc_pr,
                "val_loss": validation_result.loss,
                "val_auc_pr": validation_result.auc_pr,
            }
        )

        should_log = (
            epoch == 1
            or epoch % log_interval == 0
            or epoch == training_config.epochs
        )
        if should_log:
            _logger.info(
                "Epoch %3d/%d | train loss=%.4f auc_pr=%.4f | val loss=%.4f auc_pr=%.4f | %.1fs",
                epoch,
                training_config.epochs,
                train_result.loss,
                train_result.auc_pr,
                validation_result.loss,
                validation_result.auc_pr,
                epoch_secs,
            )

        if validation_result.auc_pr > best_validation_auc_pr:
            _logger.info(
                "  ↑ val AUC-PR %.4f -> %.4f  (epoch %d, patience reset)",
                best_validation_auc_pr if best_validation_auc_pr >= 0 else 0.0,
                validation_result.auc_pr,
                epoch,
            )
            best_validation_auc_pr = validation_result.auc_pr
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            _logger.debug(
                "  No improvement  patience %d/%d",
                epochs_without_improvement,
                training_config.patience,
            )

        if epochs_without_improvement >= training_config.patience:
            _logger.info(
                "Early stopping at epoch %d — patience %d exhausted  best val AUC-PR=%.4f",
                epoch,
                training_config.patience,
                best_validation_auc_pr,
            )
            break

    model.load_state_dict(best_state)
    _logger.info("-" * 70)
    _logger.info("Best checkpoint restored  (best val AUC-PR=%.4f)", best_validation_auc_pr)

    train_result = _evaluate_split(model=model, data=data, mask=data.train_mask, criterion=criterion)
    validation_result = _evaluate_split(model=model, data=data, mask=data.val_mask, criterion=criterion)
    test_result = _evaluate_split(model=model, data=data, mask=data.test_mask, criterion=criterion)

    _logger.info(
        "Final metrics  train  loss=%.4f  auc_pr=%.4f",
        train_result.loss,
        train_result.auc_pr,
    )
    _logger.info(
        "               val    loss=%.4f  auc_pr=%.4f",
        validation_result.loss,
        validation_result.auc_pr,
    )
    _logger.info(
        "               test   loss=%.4f  auc_pr=%.4f",
        test_result.loss,
        test_result.auc_pr,
    )

    return {
        "history": history,
        "train": {"loss": train_result.loss, "auc_pr": train_result.auc_pr},
        "validation": {"loss": validation_result.loss, "auc_pr": validation_result.auc_pr},
        "test": {"loss": test_result.loss, "auc_pr": test_result.auc_pr},
    }


def _evaluate_split(
    *,
    model: TemporalGraphModel,
    data: Any,
    mask: torch.Tensor,
    criterion: nn.BCEWithLogitsLoss,
) -> EpochResult:
    model.eval()
    with torch.inference_mode():
        logits = model(data)
        split_logits = logits[mask]
        split_labels = data.y[mask]

        if split_logits.numel() == 0:
            return EpochResult(loss=0.0, auc_pr=0.0)

        loss = float(criterion(split_logits, split_labels).cpu().item())
        probabilities = torch.sigmoid(split_logits).detach().cpu().numpy()
        labels = split_labels.detach().cpu().numpy()
        auc_pr = _safe_auc_pr(labels, probabilities)
        return EpochResult(loss=loss, auc_pr=auc_pr)


def _safe_auc_pr(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return 0.0
    return float(average_precision_score(labels, probabilities))


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
