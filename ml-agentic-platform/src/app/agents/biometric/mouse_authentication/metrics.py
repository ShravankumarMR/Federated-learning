from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_curve


@dataclass(frozen=True)
class AuthenticationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    eer: float
    eer_threshold: float


def compute_authentication_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    decision_threshold: float = 0.5,
) -> AuthenticationMetrics:
    accuracy, precision, recall, f1 = compute_classification_metrics(
        labels,
        probabilities,
        decision_threshold=decision_threshold,
    )
    eer, eer_threshold = compute_equal_error_rate(labels, probabilities)
    return AuthenticationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        eer=eer,
        eer_threshold=eer_threshold,
    )


def compute_classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    decision_threshold: float = 0.5,
) -> tuple[float, float, float, float]:
    label_values = labels.astype(np.float32)
    predictions = (probabilities >= decision_threshold).astype(np.float32)

    true_positives = float(np.sum((predictions == 1.0) & (label_values == 1.0)))
    false_positives = float(np.sum((predictions == 1.0) & (label_values == 0.0)))
    false_negatives = float(np.sum((predictions == 0.0) & (label_values == 1.0)))

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives

    precision = true_positives / precision_denominator if precision_denominator > 0.0 else 0.0
    recall = true_positives / recall_denominator if recall_denominator > 0.0 else 0.0
    f1_denominator = precision + recall
    f1 = 2.0 * precision * recall / f1_denominator if f1_denominator > 0.0 else 0.0
    accuracy = float((predictions == label_values).mean())
    return accuracy, precision, recall, f1


def compute_equal_error_rate(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return 0.0, 0.5

    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, probabilities)
    false_negative_rate = 1.0 - true_positive_rate
    index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))
    eer = float((false_positive_rate[index] + false_negative_rate[index]) / 2.0)
    threshold = float(thresholds[index])
    return eer, threshold