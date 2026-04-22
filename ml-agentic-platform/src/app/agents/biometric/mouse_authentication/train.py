from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Sequence

import mlflow
import torch

from app.agents.biometric.mouse_authentication.config import (
    MouseAuthenticationModelConfig,
    MouseAuthenticationTrainingConfig,
    MouseSequenceFeatureConfig,
)
from app.agents.biometric.mouse_authentication.data import (
    create_mouse_authentication_dataloaders,
    load_mouse_events_frame,
)
from app.agents.biometric.mouse_authentication.model import MouseAuthenticationModel
from app.agents.biometric.mouse_authentication.trainer import MouseAuthenticationTrainer
from app.core.settings import Settings, get_settings


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdout_encoding()
    args = _parse_args(argv)
    project_root = _find_project_root()
    events_path = (
        (project_root / args.events_path).resolve()
        if not Path(args.events_path).is_absolute()
        else Path(args.events_path)
    )
    settings = get_settings()
    resolved_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    allowed_splits = None if args.include_test_split else ("training",)

    if args.all_users and args.target_user_id:
        raise ValueError("Do not pass --target-user-id when --all-users is enabled")
    if not args.all_users and not args.target_user_id:
        raise ValueError("--target-user-id is required unless --all-users is enabled")

    feature_config = MouseSequenceFeatureConfig(
        max_sequence_length=args.max_sequence_length,
        min_sequence_length=args.min_sequence_length,
    )
    model_config = MouseAuthenticationModelConfig(
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        cnn_channels=tuple(args.cnn_channels),
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    )
    training_config = MouseAuthenticationTrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        validation_fraction=args.validation_fraction,
        random_seed=args.random_seed,
        decision_threshold=args.decision_threshold,
        device=resolved_device,
    )

    print(f"Loading events from: {events_path}")

    if args.all_users:
        target_users = _list_available_users(events_path, allowed_splits=allowed_splits)
        print(f"Training across all users: count={len(target_users)}")
    else:
        _validate_target_user(events_path, args.target_user_id, allowed_splits=allowed_splits)
        target_users = [args.target_user_id]

    for target_user_id in target_users:
        output_path = _build_output_path(project_root, args, target_user_id=target_user_id)
        _train_single_user(
            args=args,
            settings=settings,
            events_path=events_path,
            output_path=output_path,
            target_user_id=target_user_id,
            feature_config=feature_config,
            model_config=model_config,
            training_config=training_config,
            resolved_device=resolved_device,
            allowed_splits=allowed_splits,
        )

    return 0


def _configure_stdout_encoding() -> None:
    stdout = getattr(sys, "stdout", None)
    if stdout is not None and hasattr(stdout, "reconfigure"):
        stdout.reconfigure(encoding="utf-8", errors="replace")


def _parse_args(argv: Sequence[str] | None) -> Namespace:
    parser = ArgumentParser(description="Train a mouse dynamics authentication model for one user.")
    parser.add_argument(
        "--target-user-id",
        required=False,
        help="User identifier to treat as the legitimate user.",
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="Train one authentication model per available user.",
    )
    parser.add_argument(
        "--events-path",
        default="data/processed/mouse_dynamics/mouse_events.csv",
        help="Path to the processed mouse events dataset.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Path to save the trained checkpoint. Defaults to artifacts/mouse_authentication/<user>.pt.",
    )
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--min-sequence-length", type=int, default=8)
    parser.add_argument("--lstm-hidden-size", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--cnn-channels", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, for example cpu or cuda.",
    )
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
        default="mouse_authentication",
        help="MLflow experiment name used when tracking is enabled.",
    )
    parser.add_argument(
        "--mlflow-run-name",
        default=None,
        help="Optional MLflow run name.",
    )
    parser.add_argument(
        "--include-test-split",
        action="store_true",
        help="Include split=test sessions in training input. Disabled by default to avoid leakage.",
    )
    return parser.parse_args(argv)


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Unable to locate project root containing pyproject.toml")


def _build_output_path(project_root: Path, args: Namespace, *, target_user_id: str) -> Path:
    if args.output_path:
        if args.all_users:
            if "{user_id}" in args.output_path:
                output_path = Path(args.output_path.replace("{user_id}", target_user_id))
            else:
                base_path = Path(args.output_path)
                if base_path.suffix:
                    raise ValueError(
                        "When --all-users is enabled, --output-path must be a directory or contain {user_id}"
                    )
                output_path = base_path / f"{target_user_id}.pt"
        else:
            output_path = Path(args.output_path)
        return output_path.resolve() if output_path.is_absolute() else (project_root / output_path).resolve()
    return (project_root / "artifacts" / "mouse_authentication" / f"{target_user_id}.pt").resolve()


def _validate_target_user(
    events_path: Path,
    target_user_id: str,
    *,
    allowed_splits: tuple[str, ...] | None,
) -> None:
    events = load_mouse_events_frame(events_path)
    if allowed_splits is not None:
        allowed = {split.lower() for split in allowed_splits}
        events = events[events["split"].astype(str).str.lower().isin(allowed)].copy()
    available_users = set(events["user_id"].astype(str).unique().tolist())
    if target_user_id not in available_users:
        sample_users = ", ".join(sorted(available_users)[:10])
        raise ValueError(
            f"target_user_id '{target_user_id}' not found in dataset. Available users include: {sample_users}"
        )


def _list_available_users(
    events_path: Path,
    *,
    allowed_splits: tuple[str, ...] | None,
) -> list[str]:
    events = load_mouse_events_frame(events_path)
    if allowed_splits is not None:
        allowed = {split.lower() for split in allowed_splits}
        events = events[events["split"].astype(str).str.lower().isin(allowed)].copy()
    users = sorted(events["user_id"].astype(str).unique().tolist())
    if not users:
        raise ValueError("No users found in the dataset for the selected split filter")
    return users


def _train_single_user(
    *,
    args: Namespace,
    settings: Settings,
    events_path: Path,
    output_path: Path,
    target_user_id: str,
    feature_config: MouseSequenceFeatureConfig,
    model_config: MouseAuthenticationModelConfig,
    training_config: MouseAuthenticationTrainingConfig,
    resolved_device: str,
    allowed_splits: tuple[str, ...] | None,
) -> None:
    data_bundle = create_mouse_authentication_dataloaders(
        events_source=events_path,
        target_user_id=target_user_id,
        feature_config=feature_config,
        batch_size=training_config.batch_size,
        validation_fraction=training_config.validation_fraction,
        random_seed=training_config.random_seed,
        allowed_splits=allowed_splits,
    )

    model = MouseAuthenticationModel(
        timing_input_dim=len(feature_config.timing_features),
        movement_input_dim=len(feature_config.movement_features),
        config=model_config,
    )
    trainer = MouseAuthenticationTrainer(model=model, config=training_config)

    print(
        "Training mouse authentication model "
        f"for user={target_user_id} train_sessions={data_bundle.train_size} "
        f"validation_sessions={data_bundle.validation_size} device={resolved_device}"
    )

    mlflow_tracking_enabled = not args.disable_mlflow
    if mlflow_tracking_enabled:
        tracking_uri = args.mlflow_tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
        print(
            "MLflow enabled "
            f"tracking_uri={tracking_uri} experiment={args.mlflow_experiment_name}"
        )

        with mlflow.start_run(run_name=args.mlflow_run_name):
            _log_mlflow_params(
                target_user_id=target_user_id,
                events_path=events_path,
                output_path=output_path,
                feature_config=feature_config,
                model_config=model_config,
                training_config=training_config,
                train_size=data_bundle.train_size,
                validation_size=data_bundle.validation_size,
            )

            history, validation_metrics = _train_and_evaluate(trainer, data_bundle)
            _save_checkpoint(
                output_path=output_path,
                trainer=trainer,
                feature_config=feature_config,
                model_config=model_config,
                training_config=training_config,
                target_user_id=target_user_id,
                events_path=events_path,
                data_bundle=data_bundle,
                validation_metrics=validation_metrics,
            )
            _log_mlflow_history(history)
            _log_mlflow_validation_metrics(validation_metrics, step=len(history["train"]))
            mlflow.log_artifact(str(output_path), artifact_path="checkpoints")
    else:
        print("MLflow disabled for this run")
        history, validation_metrics = _train_and_evaluate(trainer, data_bundle)
        _save_checkpoint(
            output_path=output_path,
            trainer=trainer,
            feature_config=feature_config,
            model_config=model_config,
            training_config=training_config,
            target_user_id=target_user_id,
            events_path=events_path,
            data_bundle=data_bundle,
            validation_metrics=validation_metrics,
        )


def _print_history(history: dict[str, list[object]]) -> None:
    train_history = history["train"]
    validation_history = history["validation"]
    for epoch_index, (train_metrics, validation_metrics) in enumerate(
        zip(train_history, validation_history),
        start=1,
    ):
        print(
            f"Epoch {epoch_index:02d} | "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} "
            f"train_precision={train_metrics.precision:.4f} train_recall={train_metrics.recall:.4f} "
            f"train_f1={train_metrics.f1:.4f} train_eer={train_metrics.eer:.4f} | "
            f"val_loss={validation_metrics.loss:.4f} val_acc={validation_metrics.accuracy:.4f} "
            f"val_precision={validation_metrics.precision:.4f} val_recall={validation_metrics.recall:.4f} "
            f"val_f1={validation_metrics.f1:.4f} val_eer={validation_metrics.eer:.4f}"
        )


def _train_and_evaluate(
    trainer: MouseAuthenticationTrainer,
    data_bundle: object,
) -> tuple[dict[str, list[object]], object]:
    history = trainer.fit(
        train_loader=data_bundle.train_loader,
        validation_loader=data_bundle.validation_loader,
    )
    _print_history(history)

    validation_metrics = trainer.evaluate(data_bundle.validation_loader)
    print(
        "Best validation metrics: "
        f"loss={validation_metrics.loss:.4f} "
        f"accuracy={validation_metrics.accuracy:.4f} "
        f"precision={validation_metrics.precision:.4f} "
        f"recall={validation_metrics.recall:.4f} "
        f"f1={validation_metrics.f1:.4f} "
        f"eer={validation_metrics.eer:.4f} "
        f"eer_threshold={validation_metrics.eer_threshold:.4f}"
    )
    return history, validation_metrics


def _save_checkpoint(
    *,
    output_path: Path,
    trainer: MouseAuthenticationTrainer,
    feature_config: MouseSequenceFeatureConfig,
    model_config: MouseAuthenticationModelConfig,
    training_config: MouseAuthenticationTrainingConfig,
    target_user_id: str,
    events_path: Path,
    data_bundle: object,
    validation_metrics: object,
) -> None:
    checkpoint = {
        "target_user_id": target_user_id,
        "events_path": str(events_path),
        "feature_config": asdict(feature_config),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "model_state_dict": trainer.model.state_dict(),
        "timing_scaler_mean": data_bundle.timing_scaler.mean,
        "timing_scaler_std": data_bundle.timing_scaler.std,
        "movement_scaler_mean": data_bundle.movement_scaler.mean,
        "movement_scaler_std": data_bundle.movement_scaler.std,
        "validation_metrics": asdict(validation_metrics),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    print(f"Saved checkpoint to: {output_path}")


def _log_mlflow_params(
    *,
    target_user_id: str,
    events_path: Path,
    output_path: Path,
    feature_config: MouseSequenceFeatureConfig,
    model_config: MouseAuthenticationModelConfig,
    training_config: MouseAuthenticationTrainingConfig,
    train_size: int,
    validation_size: int,
) -> None:
    params: dict[str, object] = {
        "target_user_id": target_user_id,
        "events_path": str(events_path),
        "output_path": str(output_path),
        "train_size": train_size,
        "validation_size": validation_size,
        "max_sequence_length": feature_config.max_sequence_length,
        "min_sequence_length": feature_config.min_sequence_length,
        "timing_feature_count": len(feature_config.timing_features),
        "movement_feature_count": len(feature_config.movement_features),
        "lstm_hidden_size": model_config.lstm_hidden_size,
        "lstm_layers": model_config.lstm_layers,
        "cnn_channels": ",".join(str(channel) for channel in model_config.cnn_channels),
        "projection_dim": model_config.projection_dim,
        "dropout": model_config.dropout,
        "batch_size": training_config.batch_size,
        "learning_rate": training_config.learning_rate,
        "weight_decay": training_config.weight_decay,
        "epochs": training_config.epochs,
        "validation_fraction": training_config.validation_fraction,
        "random_seed": training_config.random_seed,
        "decision_threshold": training_config.decision_threshold,
        "device": training_config.device,
    }
    mlflow.log_params(params)


def _log_mlflow_history(history: dict[str, list[object]]) -> None:
    for epoch_index, (train_metrics, validation_metrics) in enumerate(
        zip(history["train"], history["validation"]),
        start=1,
    ):
        mlflow.log_metrics(
            {
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "train_precision": train_metrics.precision,
                "train_recall": train_metrics.recall,
                "train_f1": train_metrics.f1,
                "train_eer": train_metrics.eer,
                "train_eer_threshold": train_metrics.eer_threshold,
                "val_loss": validation_metrics.loss,
                "val_accuracy": validation_metrics.accuracy,
                "val_precision": validation_metrics.precision,
                "val_recall": validation_metrics.recall,
                "val_f1": validation_metrics.f1,
                "val_eer": validation_metrics.eer,
                "val_eer_threshold": validation_metrics.eer_threshold,
            },
            step=epoch_index,
        )


def _log_mlflow_validation_metrics(validation_metrics: object, *, step: int) -> None:
    mlflow.log_metrics(
        {
            "best_val_loss": validation_metrics.loss,
            "best_val_accuracy": validation_metrics.accuracy,
            "best_val_precision": validation_metrics.precision,
            "best_val_recall": validation_metrics.recall,
            "best_val_f1": validation_metrics.f1,
            "best_val_eer": validation_metrics.eer,
            "best_val_eer_threshold": validation_metrics.eer_threshold,
        },
        step=step,
    )


if __name__ == "__main__":
    raise SystemExit(main())