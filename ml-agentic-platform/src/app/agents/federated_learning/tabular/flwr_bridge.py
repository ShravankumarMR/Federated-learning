from __future__ import annotations

from collections.abc import Callable
from typing import Any

import flwr as fl
from flwr.common import Metrics, NDArrays, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np

from app.agents.federated_learning.tabular.model import (
    LocalTrainingConfig,
    ModelConfig,
    create_model,
    evaluate_model,
    model_to_ndarrays,
    ndarrays_to_model,
    train_local_model,
)
from app.agents.federated_learning.tabular.partitioning import ClientPartition, apply_vertical_mask


class MaskedNumPyClient(fl.client.NumPyClient):
    def __init__(
        self,
        *,
        partition: ClientPartition,
        feature_names: list[str],
        x_train: Any,
        y_train: Any,
        x_eval: Any,
        y_eval: Any,
        training_config: LocalTrainingConfig,
        model_config: ModelConfig,
    ):
        self.partition = partition
        self.training_config = training_config
        self.model = create_model(input_dim=len(feature_names), config=model_config)

        self.x_train = apply_vertical_mask(x_train, partition.owned_columns).to_numpy(dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.float32)
        self.x_eval = apply_vertical_mask(x_eval, partition.owned_columns).to_numpy(dtype=np.float32)
        self.y_eval = np.asarray(y_eval, dtype=np.float32)

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return model_to_ndarrays(self.model)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]):
        ndarrays_to_model(self.model, parameters)
        metrics = train_local_model(
            self.model,
            self.x_train,
            self.y_train,
            config=self.training_config,
        )
        return model_to_ndarrays(self.model), int(self.y_train.shape[0]), metrics

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]):
        ndarrays_to_model(self.model, parameters)
        metrics = evaluate_model(self.model, self.x_eval, self.y_eval)
        return float(metrics["loss"]), int(self.y_eval.shape[0]), metrics


class TrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.latest_parameters: NDArrays | None = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = parameters_to_ndarrays(aggregated_parameters)
        return aggregated_parameters, aggregated_metrics


def build_client_fn(
    *,
    partitions: list[ClientPartition],
    feature_names: list[str],
    x_train: Any,
    y_train: Any,
    x_eval: Any,
    y_eval: Any,
    training_config: LocalTrainingConfig,
    model_config: ModelConfig,
) -> Callable[[str], fl.client.Client]:
    partition_by_id = {partition.client_id: partition for partition in partitions}

    def client_fn(cid: str) -> fl.client.Client:
        partition = partition_by_id[cid]
        return MaskedNumPyClient(
            partition=partition,
            feature_names=feature_names,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            y_eval=y_eval,
            training_config=training_config,
            model_config=model_config,
        ).to_client()

    return client_fn


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"loss": 0.0, "auc_pr": 0.0, "accuracy": 0.0}

    keys = ("loss", "auc_pr", "accuracy")
    weighted: dict[str, float] = {}
    for key in keys:
        weighted[key] = float(
            sum(num_examples * float(values.get(key, 0.0)) for num_examples, values in metrics)
            / total_examples
        )
    return weighted


def create_tracking_strategy(
    *,
    initial_parameters: NDArrays,
    min_clients: int,
    evaluate_fn: Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None],
) -> TrackingFedAvg:
    return TrackingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=ndarrays_to_parameters(initial_parameters),
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=lambda server_round, parameters, config: evaluate_fn(
            server_round,
            parameters_to_ndarrays(parameters),
            config,
        ),
    )


# ─────────────────────────── Horizontal FL client ────────────────────────────


class HorizontalNumPyClient(fl.client.NumPyClient):
    """Flower NumPy client for horizontal (row-partitioned) federated learning.

    Each client owns a disjoint subset of training rows but has access to all
    features.  All clients share the same full evaluation set so that
    server-side centralised evaluation remains comparable across rounds.
    """

    def __init__(
        self,
        *,
        row_indices: list[int],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
        training_config: LocalTrainingConfig,
        model_config: ModelConfig,
    ) -> None:
        self.training_config = training_config
        self.model = create_model(input_dim=x_train.shape[1], config=model_config)
        self.x_train = x_train[row_indices]
        self.y_train = y_train[row_indices]
        self.x_eval = x_eval
        self.y_eval = y_eval

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return model_to_ndarrays(self.model)

    def fit(self, parameters: NDArrays, config: dict[str, Scalar]):
        ndarrays_to_model(self.model, parameters)
        metrics = train_local_model(
            self.model,
            self.x_train,
            self.y_train,
            config=self.training_config,
        )
        return model_to_ndarrays(self.model), int(self.y_train.shape[0]), metrics

    def evaluate(self, parameters: NDArrays, config: dict[str, Scalar]):
        ndarrays_to_model(self.model, parameters)
        metrics = evaluate_model(self.model, self.x_eval, self.y_eval)
        return float(metrics["loss"]), int(self.y_eval.shape[0]), metrics


def build_horizontal_client_fn(
    *,
    row_partitions: list[list[int]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    training_config: LocalTrainingConfig,
    model_config: ModelConfig,
) -> Callable[[str], fl.client.Client]:
    """Build a Flower client factory for horizontal (row-partitioned) FL.

    Args:
        row_partitions: List of row-index lists — one per client.
        x_train / y_train: Full training arrays (clients receive subsets).
        x_eval / y_eval: Full evaluation arrays (shared across all clients).
        training_config / model_config: Passed to each client unchanged.

    Returns:
        A callable ``client_fn(cid: str) -> fl.client.Client``.
    """

    def client_fn(cid: str) -> fl.client.Client:
        row_indices = row_partitions[int(cid)]
        return HorizontalNumPyClient(
            row_indices=row_indices,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            y_eval=y_eval,
            training_config=training_config,
            model_config=model_config,
        ).to_client()

    return client_fn
