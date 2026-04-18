from typing import Any


class GraphFraudAgent:
    def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        degree = int(payload.get("node_degree", 0))
        shared_devices = int(payload.get("shared_devices", 0))
        risk = min(1.0, (degree * 0.1) + (shared_devices * 0.15))
        return {
            "score": round(1.0 - risk, 4),
            "risk": round(risk, 4),
            "signal": "graph_connectivity",
        }
