from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from app.agents.graph_fraud.temporal_gnn.config import TemporalModelConfig
from app.agents.graph_fraud.temporal_gnn.model import TemporalGraphModel

try:
    from torch_geometric.data import Data
except ImportError:
    Data = Any


class GraphFraudAgent:
    def __init__(self) -> None:
        self._model_cache: dict[tuple[str, str, int, int, int, int, int, int, float], TemporalGraphModel] = {}

    def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._can_use_temporal_model(payload):
            try:
                return self._score_with_temporal_gnn(payload)
            except Exception as exc:
                heuristic_result = self._score_with_connectivity_heuristic(payload)
                heuristic_result["fallback_reason"] = f"temporal_gnn_failed: {exc}"
                return heuristic_result

        return self._score_with_connectivity_heuristic(payload)

    def _can_use_temporal_model(self, payload: dict[str, Any]) -> bool:
        required_fields = ["model_path", "x", "edge_index", "edge_attr", "edge_time", "target_node_index"]
        return all(field in payload for field in required_fields)

    def _score_with_temporal_gnn(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = self._load_model(
            model_path=Path(str(payload["model_path"])),
            backbone=str(payload.get("backbone", "gat")),
            node_feature_dim=int(payload.get("node_feature_dim", len(payload["x"][0]))),
            edge_attr_dim=int(payload.get("edge_attr_dim", len(payload["edge_attr"][0]))),
            hidden_dim=int(payload.get("hidden_dim", 64)),
            num_layers=int(payload.get("num_layers", 2)),
            time_encoding_dim=int(payload.get("time_encoding_dim", 16)),
            gat_heads=int(payload.get("gat_heads", 4)),
            dropout=float(payload.get("dropout", 0.2)),
        )

        graph_data = Data(
            x=torch.tensor(payload["x"], dtype=torch.float32),
            edge_index=torch.tensor(payload["edge_index"], dtype=torch.long),
            edge_attr=torch.tensor(payload["edge_attr"], dtype=torch.float32),
            edge_time=torch.tensor(payload["edge_time"], dtype=torch.float32),
        )
        target_node_index = int(payload["target_node_index"])
        threshold = float(payload.get("decision_threshold", 0.5))

        model.eval()
        with torch.inference_mode():
            logits = model(graph_data)
            fraud_probability = float(torch.sigmoid(logits[target_node_index]).cpu().item())

        risk = max(0.0, min(1.0, fraud_probability))
        decision = "fraud" if risk >= threshold else "legit"
        return {
            "score": round(1.0 - risk, 4),
            "risk": round(risk, 4),
            "decision": decision,
            "signal": "temporal_gnn",
        }

    def _load_model(
        self,
        *,
        model_path: Path,
        backbone: str,
        node_feature_dim: int,
        edge_attr_dim: int,
        hidden_dim: int,
        num_layers: int,
        time_encoding_dim: int,
        gat_heads: int,
        dropout: float,
    ) -> TemporalGraphModel:
        resolved_path = model_path.resolve()
        cache_key = (
            str(resolved_path),
            backbone,
            node_feature_dim,
            edge_attr_dim,
            hidden_dim,
            num_layers,
            time_encoding_dim,
            gat_heads,
            dropout,
        )
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        config = TemporalModelConfig(
            backbone=backbone,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            time_encoding_dim=time_encoding_dim,
            gat_heads=gat_heads,
        )
        model = TemporalGraphModel(
            node_feature_dim=node_feature_dim,
            edge_attr_dim=edge_attr_dim,
            config=config,
        )
        state_dict = torch.load(resolved_path, map_location="cpu")
        model.load_state_dict(state_dict)
        self._model_cache[cache_key] = model
        return model

    def _score_with_connectivity_heuristic(self, payload: dict[str, Any]) -> dict[str, Any]:
        degree = int(payload.get("node_degree", 0))
        shared_devices = int(payload.get("shared_devices", 0))
        risk = min(1.0, (degree * 0.1) + (shared_devices * 0.15))
        return {
            "score": round(1.0 - risk, 4),
            "risk": round(risk, 4),
            "signal": "graph_connectivity",
        }
