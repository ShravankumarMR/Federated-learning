"""Flower tabular federated simulation — standalone training command.

This module is the **top-level entrypoint** for the IEEE-CIS vertical-FL workflow.
It is intentionally decoupled from the FastAPI / LangGraph request path.

CLI usage (after ``pip install -e .``)
---------------------------------------
::

    # Quickstart — defaults to settings.federated_rounds / federated_min_clients
    fl-tabular-sim --features-dir data/features/ieee_cis_fraud

    # Full example with NN model, MLflow, custom output directory
    fl-tabular-sim \\
        --features-dir data/features/ieee_cis_fraud \\
        --processed-dir data/processed/ieee_cis_fraud \\
        --output-dir artifacts/federated_learning \\
        --num-clients 3 --rounds 5 \\
        --model-type nn --hidden-dim 64 \\
        --local-epochs 2 --batch-size 256 \\
        --mlflow-experiment-name federated_tabular_fraud

    # Same via python -m
    python -m app.agents.federated_learning.tabular.simulation \\
        --features-dir data/features/ieee_cis_fraud --rounds 3 --disable-mlflow

Outputs written to ``--output-dir`` (default ``artifacts/federated_learning/``)
--------------------------------------------------------------------------------
* ``ieee_cis_global_model.pt``        — aggregated global model state-dict
* ``aggregated_global_parameters.npz`` — raw numpy parameter arrays
* ``vertical_partition_map.json``     — per-client feature ownership metadata
* ``shap_summary.json``               — global + per-client SHAP attributions
* ``federated_metrics.json``          — round history + final eval metrics

Integration boundary
--------------------
``run_federated_simulation`` is a **long-running, CPU-bound** function.  Do **not**
call it inside a FastAPI request handler or a LangGraph node that runs in the hot
path.  Use it from:

* The ``fl-tabular-sim`` CLI (CI/CD, scheduled job)
* ``FederatedLearningAgent.simulate_tabular`` called from a background worker
* A Makefile target (``make train-federated``)

For request-path inference against the saved model, add a separate lightweight
wrapper; see the module docstring in ``agent.py`` for details.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import flwr as fl
import mlflow
import numpy as np
import torch

from app.agents.federated_learning.tabular.data import load_ieee_cis_dataset, load_paysim_dataset
from app.agents.federated_learning.tabular.flwr_bridge import (
    build_client_fn,
    build_horizontal_client_fn,
    create_tracking_strategy,
)
from app.agents.federated_learning.tabular.model import (
    LocalTrainingConfig,
    ModelConfig,
    create_model,
    evaluate_model,
    model_to_ndarrays,
    ndarrays_to_model,
)
from app.agents.federated_learning.tabular.partitioning import (
    build_horizontal_partition_metadata,
    build_horizontal_partitions,
    build_partition_metadata,
    build_vertical_partitions,
)
from app.agents.federated_learning.tabular.shap_utils import compute_shap_summary
from app.core.settings import get_settings

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationConfig:
    num_clients: int = 3
    rounds: int = 3
    local_epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    model_type: str = "logistic"
    hidden_dim: int = 64
    random_state: int = 42
    max_explain_samples: int = 128
    include_parameters_in_response: bool = False
    enable_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "federated_tabular_fraud"
    mlflow_run_name: str | None = None
    # Dataset and partitioning mode
    dataset: str = "ieee_cis"       # "ieee_cis" | "paysim"
    partition_mode: str = "vertical"  # "vertical" | "horizontal"


def run_federated_simulation(
    *,
    config: SimulationConfig | None = None,
    features_dir: Path | None = None,
    processed_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    simulation_config = config or SimulationConfig()
    settings = get_settings()

    # ── Dataset loading ──────────────────────────────────────────────────────
    _logger.info("=" * 70)
    _logger.info(
        "Federated simulation  dataset=%s  partition=%s",
        simulation_config.dataset,
        simulation_config.partition_mode,
    )
    _logger.info(
        "Config  clients=%d  rounds=%d  local_epochs=%d  model=%s",
        simulation_config.num_clients,
        simulation_config.rounds,
        simulation_config.local_epochs,
        simulation_config.model_type,
    )
    _logger.info("=" * 70)

    _valid_datasets = {"ieee_cis", "paysim"}
    if simulation_config.dataset not in _valid_datasets:
        raise ValueError(
            f"Unknown dataset '{simulation_config.dataset}'. "
            f"Valid options: {sorted(_valid_datasets)}"
        )
    _valid_modes = {"vertical", "horizontal"}
    if simulation_config.partition_mode not in _valid_modes:
        raise ValueError(
            f"Unknown partition_mode '{simulation_config.partition_mode}'. "
            f"Valid options: {sorted(_valid_modes)}"
        )

    if simulation_config.dataset == "paysim":
        _logger.info("Loading PaySim dataset...")
        dataset = load_paysim_dataset(
            features_dir=features_dir,
            processed_dir=processed_dir,
            random_state=simulation_config.random_state,
        )
    else:
        _logger.info("Loading IEEE-CIS dataset...")
        dataset = load_ieee_cis_dataset(
            features_dir=features_dir,
            processed_dir=processed_dir,
            random_state=simulation_config.random_state,
        )
    _logger.info(
        "Dataset loaded  train=%d  eval=%d  features=%d",
        dataset.x_train.shape[0],
        dataset.x_eval.shape[0],
        len(dataset.feature_names),
    )

    model_config = ModelConfig(
        model_type=simulation_config.model_type,
        hidden_dim=simulation_config.hidden_dim,
    )
    train_config = LocalTrainingConfig(
        epochs=simulation_config.local_epochs,
        batch_size=simulation_config.batch_size,
        learning_rate=simulation_config.learning_rate,
        weight_decay=simulation_config.weight_decay,
    )

    global_model = create_model(input_dim=len(dataset.feature_names), config=model_config)
    initial_parameters = model_to_ndarrays(global_model)

    x_train = dataset.x_train
    y_train = dataset.y_train.to_numpy(dtype=np.float32)
    x_eval = dataset.x_eval
    y_eval = dataset.y_eval.to_numpy(dtype=np.float32)

    # ── Partitioning ─────────────────────────────────────────────────────────
    if simulation_config.partition_mode == "horizontal":
        _logger.info(
            "Horizontal partitioning  %d rows -> %d clients (stratified)",
            int(x_train.shape[0]),
            simulation_config.num_clients,
        )
        row_splits = build_horizontal_partitions(
            num_train_rows=int(x_train.shape[0]),
            num_clients=simulation_config.num_clients,
            random_state=simulation_config.random_state,
            stratify_labels=y_train,
        )
        partition_metadata = build_horizontal_partition_metadata(
            num_clients=simulation_config.num_clients,
            row_index_splits=row_splits,
            feature_names=dataset.feature_names,
            num_eval_rows=int(x_eval.shape[0]),
        )
        client_fn = build_horizontal_client_fn(
            row_partitions=row_splits,
            x_train=x_train.to_numpy(dtype=np.float32),
            y_train=y_train,
            x_eval=x_eval.to_numpy(dtype=np.float32),
            y_eval=y_eval,
            training_config=train_config,
            model_config=model_config,
        )
    else:
        _logger.info(
            "Vertical partitioning  %d features -> %d clients",
            len(dataset.feature_names),
            simulation_config.num_clients,
        )
        partitions = build_vertical_partitions(
            dataset.feature_names, num_clients=simulation_config.num_clients
        )
        partition_metadata = build_partition_metadata(
            partitions=partitions,
            feature_names=dataset.feature_names,
            num_train_rows=int(x_train.shape[0]),
            num_eval_rows=int(x_eval.shape[0]),
        )
        client_fn = build_client_fn(
            partitions=partitions,
            feature_names=dataset.feature_names,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            y_eval=y_eval,
            training_config=train_config,
            model_config=model_config,
        )

    def evaluate_fn(server_round: int, parameters: list[np.ndarray], _: dict[str, Any]):
        del server_round
        ndarrays_to_model(global_model, parameters)
        metrics = evaluate_model(
            global_model,
            x_eval.to_numpy(dtype=np.float32),
            y_eval,
        )
        return metrics["loss"], {
            "auc_pr": float(metrics["auc_pr"]),
            "accuracy": float(metrics["accuracy"]),
            "loss": float(metrics["loss"]),
        }

    strategy = create_tracking_strategy(
        initial_parameters=initial_parameters,
        min_clients=simulation_config.num_clients,
        evaluate_fn=evaluate_fn,
    )

    _logger.info("Starting Flower simulation (%d rounds)...", simulation_config.rounds)
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=simulation_config.num_clients,
        config=fl.server.ServerConfig(num_rounds=simulation_config.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )
    _logger.info("Simulation complete")

    final_parameters = strategy.latest_parameters or initial_parameters
    ndarrays_to_model(global_model, final_parameters)

    final_metrics = evaluate_model(
        global_model,
        x_eval.to_numpy(dtype=np.float32),
        y_eval,
    )

    shap_summary = compute_shap_summary(
        model=global_model,
        x_background=x_train.to_numpy(dtype=np.float32),
        x_explain=x_eval.to_numpy(dtype=np.float32),
        feature_names=dataset.feature_names,
        feature_owner_map=partition_metadata["feature_owner_map"],
        max_explain_samples=simulation_config.max_explain_samples,
    )

    target_output = output_dir or Path("artifacts/federated_learning")
    target_output.mkdir(parents=True, exist_ok=True)

    model_path = target_output / f"{simulation_config.dataset}_global_model.pt"
    torch.save(global_model.state_dict(), model_path)
    _logger.info("Global model saved: %s", model_path)

    parameters_path = target_output / "aggregated_global_parameters.npz"
    np.savez(parameters_path, *final_parameters)

    partition_path = target_output / "vertical_partition_map.json"
    partition_path.write_text(json.dumps(partition_metadata, indent=2), encoding="utf-8")

    shap_path = target_output / "shap_summary.json"
    shap_path.write_text(json.dumps(shap_summary, indent=2), encoding="utf-8")

    metrics_payload = {
        "config": asdict(simulation_config),
        "final": final_metrics,
        "aggregated_global_parameters": _build_parameter_payload(
            final_parameters,
            include_values=False,
            artifact_path=str(parameters_path),
        ),
        "history": {
            "losses_distributed": history.losses_distributed,
            "metrics_distributed_fit": history.metrics_distributed_fit,
            "metrics_distributed": history.metrics_distributed,
            "metrics_centralized": history.metrics_centralized,
        },
    }
    metrics_path = target_output / "federated_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    _logger.info("Metrics saved: %s", metrics_path)

    if simulation_config.enable_mlflow:
        tracking_uri = simulation_config.mlflow_tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(simulation_config.mlflow_experiment_name)
        run_name = _resolve_mlflow_run_name(simulation_config)

        with mlflow.start_run(run_name=run_name):
            _log_mlflow_params(
                simulation_config=simulation_config,
                num_features=len(dataset.feature_names),
                num_train_rows=int(x_train.shape[0]),
                num_eval_rows=int(x_eval.shape[0]),
                features_dir=features_dir,
                processed_dir=processed_dir,
            )
            _log_mlflow_history(history)
            _log_mlflow_final_metrics(final_metrics)
            mlflow.log_artifact(str(model_path), artifact_path="checkpoints")
            mlflow.log_artifact(str(parameters_path), artifact_path="checkpoints")
            mlflow.log_artifact(str(partition_path), artifact_path="metadata")
            mlflow.log_artifact(str(shap_path), artifact_path="metrics")
            mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

    return {
        "final_metrics": final_metrics,
        "aggregated_global_parameters": _build_parameter_payload(
            final_parameters,
            include_values=simulation_config.include_parameters_in_response,
            artifact_path=str(parameters_path),
        ),
        "artifacts": {
            "model": str(model_path),
            "aggregated_parameters": str(parameters_path),
            "partition_map": str(partition_path),
            "shap_summary": str(shap_path),
            "metrics": str(metrics_path),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_federated_simulation(
        config=SimulationConfig(
            num_clients=args.num_clients,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            model_type=args.model_type,
            hidden_dim=args.hidden_dim,
            random_state=args.random_state,
            max_explain_samples=args.max_explain_samples,
            include_parameters_in_response=args.include_parameters_in_response,
            enable_mlflow=not args.disable_mlflow,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name,
            mlflow_run_name=args.mlflow_run_name,
            dataset=args.dataset,
            partition_mode=args.partition_mode,
        ),
        features_dir=Path(args.features_dir).resolve() if args.features_dir else None,
        processed_dir=Path(args.processed_dir).resolve() if args.processed_dir else None,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )
    print(json.dumps(result, indent=2))
    return 0


def _parse_args(argv: Sequence[str] | None) -> Namespace:
    settings = get_settings()
    parser = ArgumentParser(description="Run Flower tabular federated simulation for IEEE-CIS fraud")
    parser.add_argument("--features-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--output-dir", default="artifacts/federated_learning")
    parser.add_argument("--num-clients", type=int, default=settings.federated_min_clients)
    parser.add_argument("--rounds", type=int, default=settings.federated_rounds)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-type", choices=["logistic", "nn"], default="logistic")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-explain-samples", type=int, default=128)
    parser.add_argument("--include-parameters-in-response", action="store_true")
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow logging for this run.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=settings.mlflow_tracking_uri,
        help="Override MLflow tracking URI.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default="federated_tabular_fraud",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional MLflow run name.",
    )
    # Dataset and partitioning
    parser.add_argument(
        "--dataset",
        choices=["ieee_cis", "paysim"],
        default="ieee_cis",
        help="Dataset to use for federated training (default: ieee_cis).",
    )
    parser.add_argument(
        "--partition-mode",
        choices=["vertical", "horizontal"],
        default="vertical",
        help="Partition mode: vertical (column-split) or horizontal (row-split) (default: vertical).",
    )
    return parser.parse_args(argv)


def _build_parameter_payload(
    parameters: list[np.ndarray],
    *,
    include_values: bool,
    artifact_path: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact_path": artifact_path,
        "num_tensors": len(parameters),
        "shapes": [list(array.shape) for array in parameters],
        "dtypes": [str(array.dtype) for array in parameters],
    }
    if include_values:
        payload["values"] = [array.tolist() for array in parameters]
    return payload


def _resolve_mlflow_run_name(simulation_config: SimulationConfig) -> str:
    if simulation_config.mlflow_run_name:
        return simulation_config.mlflow_run_name
    return (
        f"federated_tabular_{simulation_config.model_type}_"
        f"c{simulation_config.num_clients}_r{simulation_config.rounds}"
    )


def _log_mlflow_params(
    *,
    simulation_config: SimulationConfig,
    num_features: int,
    num_train_rows: int,
    num_eval_rows: int,
    features_dir: Path | None,
    processed_dir: Path | None,
) -> None:
    mlflow.log_params(
        {
            "num_clients": simulation_config.num_clients,
            "rounds": simulation_config.rounds,
            "local_epochs": simulation_config.local_epochs,
            "batch_size": simulation_config.batch_size,
            "learning_rate": simulation_config.learning_rate,
            "weight_decay": simulation_config.weight_decay,
            "model_type": simulation_config.model_type,
            "hidden_dim": simulation_config.hidden_dim,
            "random_state": simulation_config.random_state,
            "max_explain_samples": simulation_config.max_explain_samples,
            "dataset": simulation_config.dataset,
            "partition_mode": simulation_config.partition_mode,
            "num_features": num_features,
            "num_train_rows": num_train_rows,
            "num_eval_rows": num_eval_rows,
            "features_dir": str(features_dir) if features_dir else "default",
            "processed_dir": str(processed_dir) if processed_dir else "default",
        }
    )


def _log_mlflow_history(history: fl.server.history.History) -> None:
    for round_idx, loss_value in history.losses_distributed:
        mlflow.log_metric("loss_distributed", float(loss_value), step=int(round_idx))

    for metric_name, series in history.metrics_distributed_fit.items():
        for round_idx, metric_value in series:
            mlflow.log_metric(
                f"fit_{metric_name}",
                float(metric_value),
                step=int(round_idx),
            )

    for metric_name, series in history.metrics_distributed.items():
        for round_idx, metric_value in series:
            mlflow.log_metric(
                f"distributed_{metric_name}",
                float(metric_value),
                step=int(round_idx),
            )

    for metric_name, series in history.metrics_centralized.items():
        for round_idx, metric_value in series:
            mlflow.log_metric(
                f"centralized_{metric_name}",
                float(metric_value),
                step=int(round_idx),
            )


def _log_mlflow_final_metrics(final_metrics: dict[str, float]) -> None:
    mlflow.log_metrics(
        {
            "final_loss": float(final_metrics.get("loss", 0.0)),
            "final_auc_pr": float(final_metrics.get("auc_pr", 0.0)),
            "final_accuracy": float(final_metrics.get("accuracy", 0.0)),
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
