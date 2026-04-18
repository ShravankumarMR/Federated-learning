from typing import Any

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
