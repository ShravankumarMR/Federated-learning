"""CLI entry point for the Flower tabular federated simulation.

This module owns argument parsing and the ``main()`` function so that
``simulation.py`` stays focused on the simulation logic.

Entry point (registered in ``pyproject.toml``):
    fl-tabular-sim  →  app.agents.federated_learning.tabular.cli:main
"""
from __future__ import annotations

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Sequence

from app.agents.federated_learning.tabular.simulation import SimulationConfig, run_federated_simulation
from app.core.settings import get_settings

_logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    raise SystemExit(main())
