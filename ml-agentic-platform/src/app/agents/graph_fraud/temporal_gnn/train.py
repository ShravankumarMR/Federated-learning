from __future__ import annotations

from argparse import ArgumentParser, Namespace
import json
import logging
from pathlib import Path
from typing import Sequence

import mlflow
import torch

from app.agents.graph_fraud.temporal_gnn.config import TemporalModelConfig, TemporalTrainingConfig
from app.agents.graph_fraud.temporal_gnn.data import build_temporal_node_classification_data
from app.agents.graph_fraud.temporal_gnn.model import TemporalGraphModel
from app.agents.graph_fraud.temporal_gnn.trainer import train_and_evaluate
from app.core.logging import setup_logging
from app.core.settings import get_settings

_logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(args.log_level)

    settings = get_settings()
    dataset_names = _resolve_datasets(args.dataset)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow_enabled = not args.disable_mlflow
    if mlflow_enabled:
        tracking_uri = args.mlflow_tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
        _logger.info(
            "MLflow enabled  tracking_uri=%s  experiment=%s",
            tracking_uri,
            args.mlflow_experiment_name,
        )
    else:
        _logger.info("MLflow disabled for this run")

    for dataset_name in dataset_names:
        csv_path = _resolve_dataset_path(args, dataset_name)

        _logger.info("")
        _logger.info("=" * 70)
        _logger.info("Dataset       : %s", dataset_name)
        _logger.info("Input CSV     : %s", csv_path)
        _logger.info("Output dir    : %s", output_dir)
        _logger.info(
            "Backbone      : %s  hidden=%d  layers=%d  dropout=%.2f  "
            "time_enc=%d  gat_heads=%d",
            args.backbone,
            args.hidden_dim,
            args.num_layers,
            args.dropout,
            args.time_encoding_dim,
            args.gat_heads,
        )
        _logger.info(
            "Schedule      : epochs=%d  patience=%d  lr=%.5f  wd=%.5f  clip=%.1f",
            args.epochs,
            args.patience,
            args.learning_rate,
            args.weight_decay,
            args.gradient_clip_norm,
        )
        _logger.info(
            "Splits        : train=%.0f%%  val=%.0f%%  test=%.0f%%  seed=%d",
            args.train_ratio * 100,
            args.validation_ratio * 100,
            (1 - args.train_ratio - args.validation_ratio) * 100,
            args.random_seed,
        )
        _logger.info("=" * 70)

        data, _ = build_temporal_node_classification_data(
            dataset_name=dataset_name,
            csv_path=csv_path,
            random_seed=args.random_seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
        )

        model_config = TemporalModelConfig(
            backbone=args.backbone,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            time_encoding_dim=args.time_encoding_dim,
            gat_heads=args.gat_heads,
        )
        model = TemporalGraphModel(
            node_feature_dim=data.x.shape[1],
            edge_attr_dim=data.edge_attr.shape[1],
            config=model_config,
        )

        training_config = TemporalTrainingConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            gradient_clip_norm=args.gradient_clip_norm,
            device=args.device,
            log_interval=args.log_interval,
        )

        model_path = output_dir / f"{dataset_name}_temporal_{args.backbone}.pt"
        metrics_path = output_dir / f"{dataset_name}_metrics.json"

        if mlflow_enabled:
            run_name = _resolve_mlflow_run_name(args.mlflow_run_name, dataset_name)
            with mlflow.start_run(run_name=run_name):
                _log_mlflow_params(args=args, dataset_name=dataset_name, csv_path=csv_path)
                results = train_and_evaluate(model=model, data=data, config=training_config)
                _log_mlflow_history(results)
                _log_mlflow_final_metrics(results)
                _print_summary(results)

                torch.save(model.state_dict(), model_path)
                metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
                mlflow.log_artifact(str(model_path), artifact_path="checkpoints")
                mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        else:
            results = train_and_evaluate(model=model, data=data, config=training_config)
            _print_summary(results)
            torch.save(model.state_dict(), model_path)
            metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        _logger.info("Saved model   : %s", model_path)
        _logger.info("Saved metrics : %s", metrics_path)

    return 0


def _parse_args(argv: Sequence[str] | None) -> Namespace:
    parser = ArgumentParser(description="Train temporal GNN models for fraud node classification.")
    parser.add_argument(
        "--dataset",
        choices=["ieee_cis", "paysim", "both"],
        default="both",
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--ieee-csv-path",
        default=None,
        help="Path to IEEE-CIS transactions CSV file.",
    )
    parser.add_argument(
        "--paysim-csv-path",
        default=None,
        help="Path to PaySim transactions CSV file.",
    )
    parser.add_argument("--output-dir", default="artifacts/temporal_gnn")

    parser.add_argument("--backbone", choices=["gcn", "gat"], default="gat")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--time-encoding-dim", type=int, default=16)
    parser.add_argument("--gat-heads", type=int, default=4)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu or cuda")
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow tracking for this run.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help="Override MLflow tracking URI. Defaults to app settings.",
    )
    parser.add_argument(
        "--mlflow-experiment-name",
        default="temporal_graph_fraud",
        help="MLflow experiment name used when tracking is enabled.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional base run name. Dataset name is appended when training multiple datasets.",
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Terminal logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="Print epoch progress every N epochs (default: 1 = every epoch).",
    )

    return parser.parse_args(argv)


def _resolve_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "both":
        return ["ieee_cis", "paysim"]
    return [dataset_arg]


def _resolve_dataset_path(args: Namespace, dataset_name: str) -> Path:
    if dataset_name == "ieee_cis":
        if not args.ieee_csv_path:
            raise ValueError("--ieee-csv-path is required for IEEE-CIS training")
        return Path(args.ieee_csv_path).resolve()

    if not args.paysim_csv_path:
        raise ValueError("--paysim-csv-path is required for PaySim training")
    return Path(args.paysim_csv_path).resolve()


def _print_summary(results: dict[str, object]) -> None:
    train_metrics = results["train"]
    validation_metrics = results["validation"]
    test_metrics = results["test"]
    epochs_run = len(results.get("history", []))  # type: ignore[arg-type]

    _logger.info("-" * 70)
    _logger.info("Summary  (epochs run: %d)", epochs_run)
    _logger.info(
        "  AUC-PR  train=%.4f  val=%.4f  test=%.4f",
        train_metrics["auc_pr"],
        validation_metrics["auc_pr"],
        test_metrics["auc_pr"],
    )
    _logger.info(
        "  Loss    train=%.4f  val=%.4f  test=%.4f",
        train_metrics["loss"],
        validation_metrics["loss"],
        test_metrics["loss"],
    )


def _resolve_mlflow_run_name(base_run_name: str | None, dataset_name: str) -> str:
    if not base_run_name:
        return f"temporal_gnn_{dataset_name}"
    return f"{base_run_name}_{dataset_name}"


def _log_mlflow_params(*, args: Namespace, dataset_name: str, csv_path: Path) -> None:
    mlflow.log_params(
        {
            "dataset": dataset_name,
            "csv_path": str(csv_path),
            "backbone": args.backbone,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "time_encoding_dim": args.time_encoding_dim,
            "gat_heads": args.gat_heads,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "gradient_clip_norm": args.gradient_clip_norm,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "random_seed": args.random_seed,
            "device": args.device or "auto",
        }
    )


def _log_mlflow_history(results: dict[str, object]) -> None:
    history = results.get("history", [])
    for epoch_metrics in history:
        step = int(epoch_metrics["epoch"])
        mlflow.log_metrics(
            {
                "train_loss": float(epoch_metrics["train_loss"]),
                "train_auc_pr": float(epoch_metrics["train_auc_pr"]),
                "val_loss": float(epoch_metrics["val_loss"]),
                "val_auc_pr": float(epoch_metrics["val_auc_pr"]),
            },
            step=step,
        )


def _log_mlflow_final_metrics(results: dict[str, object]) -> None:
    train_metrics = results["train"]
    validation_metrics = results["validation"]
    test_metrics = results["test"]
    mlflow.log_metrics(
        {
            "final_train_loss": float(train_metrics["loss"]),
            "final_train_auc_pr": float(train_metrics["auc_pr"]),
            "final_validation_loss": float(validation_metrics["loss"]),
            "final_validation_auc_pr": float(validation_metrics["auc_pr"]),
            "final_test_loss": float(test_metrics["loss"]),
            "final_test_auc_pr": float(test_metrics["auc_pr"]),
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
