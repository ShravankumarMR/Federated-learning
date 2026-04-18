from typing import Any

from pydantic import BaseModel, Field


class OrchestrationRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    session_id: str = Field(..., description="Session identifier")
    biometric_payload: dict[str, Any] = Field(default_factory=dict)
    graph_payload: dict[str, Any] = Field(default_factory=dict)
    federated_payload: dict[str, Any] = Field(default_factory=dict)
    rag_query: str = ""
