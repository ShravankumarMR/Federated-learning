from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class ParamChangeEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    model_id: str = Field(default="global-federated-model")
    change_type: Literal[
        "graph_drift",
        "biometric_drift",
        "federated_drift",
        "rag_context_change",
        "mixed",
    ] = "mixed"
    changed_parameters: list[str] = Field(default_factory=list)
    delta_magnitude: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str = Field(default="streamlit-poc")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    affected_agents: list[Literal["biometric", "graph_fraud", "federated", "rag"]] = Field(
        default_factory=list
    )


class OrchestrationRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    session_id: str = Field(..., description="Session identifier")
    correlation_id: str | None = Field(default=None, description="Trace id across all agent nodes")
    dataset: Literal["ieee_cis", "paysim"] = Field(
        default="ieee_cis",
        description="Source dataset that determines which graph fraud agent handles scoring",
    )
    param_change_event: ParamChangeEvent | None = None
    biometric_payload: dict[str, Any] = Field(default_factory=dict)
    graph_payload: dict[str, Any] = Field(default_factory=dict)
    federated_payload: dict[str, Any] = Field(default_factory=dict)
    rag_query: str = ""
