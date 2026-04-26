"""Federated learning agent — integration boundary.

This module exposes two public methods:

* ``score`` — lightweight quorum/quality scorer for request-path inference.
  It reads pre-computed ``global_model_quality`` from a payload; it does **not**
  invoke Flower and is safe to call inside a live FastAPI/LangGraph request.

* ``simulate_tabular`` — one-shot training trigger that runs the full Flower
  simulation, persists artifacts, and returns a metrics/artifact summary.
  It is **not** suitable for use inside a request path (long-running, CPU-bound).
  Call it from a background job, CI pipeline, or the ``fl-tabular-sim`` CLI:

      fl-tabular-sim --features-dir data/features/ieee_cis_fraud --rounds 3

  or equivalently:

      python -m app.agents.federated_learning.tabular.simulation \\
          --features-dir data/features/ieee_cis_fraud \\
          --rounds 3 --model-type nn --disable-mlflow

LangGraph / API inference integration is explicitly **out of scope** unless later
requested. If runtime scoring via the saved global model is needed, add a
separate inference-only wrapper around ``artifacts/federated_learning/ieee_cis_global_model.pt``
rather than embedding Flower simulation inside the orchestration graph.
"""
from typing import Any
from pathlib import Path

from app.agents.federated_learning.tabular.simulation import (
    SimulationConfig,
    run_federated_simulation,
)
from app.core.settings import get_settings


class FederatedLearningAgent:
    def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = get_settings()
        participating_clients = int(payload.get("participating_clients", 0))
        min_clients = settings.federated_min_clients

        quorum_ok = participating_clients >= min_clients
        quality = float(payload.get("global_model_quality", 0.5))
        confidence = quality if quorum_ok else quality * 0.7

        return {
            "score": round(confidence, 4),
            "risk": round(1.0 - confidence, 4),
            "signal": "federated_consensus",
            "quorum_ok": quorum_ok,
            "rounds": settings.federated_rounds,
        }

    def simulate_tabular(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = get_settings()
        config = SimulationConfig(
            num_clients=int(payload.get("num_clients", settings.federated_min_clients)),
            rounds=int(payload.get("rounds", settings.federated_rounds)),
            local_epochs=int(payload.get("local_epochs", 2)),
            batch_size=int(payload.get("batch_size", 256)),
            learning_rate=float(payload.get("learning_rate", 1e-3)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            model_type=str(payload.get("model_type", "logistic")),
            hidden_dim=int(payload.get("hidden_dim", 64)),
            random_state=int(payload.get("random_state", 42)),
            max_explain_samples=int(payload.get("max_explain_samples", 128)),
            include_parameters_in_response=bool(payload.get("include_parameters_in_response", False)),
            enable_mlflow=bool(payload.get("enable_mlflow", False)),
            mlflow_tracking_uri=(
                str(payload["mlflow_tracking_uri"])
                if payload.get("mlflow_tracking_uri")
                else None
            ),
            mlflow_experiment_name=str(
                payload.get("mlflow_experiment_name", "federated_tabular_fraud")
            ),
            mlflow_run_name=(str(payload["mlflow_run_name"]) if payload.get("mlflow_run_name") else None),
        )

        features_dir = payload.get("features_dir")
        processed_dir = payload.get("processed_dir")
        output_dir = payload.get("output_dir")

        return run_federated_simulation(
            config=config,
            features_dir=Path(features_dir).resolve() if features_dir else None,
            processed_dir=Path(processed_dir).resolve() if processed_dir else None,
            output_dir=Path(output_dir).resolve() if output_dir else None,
        )
