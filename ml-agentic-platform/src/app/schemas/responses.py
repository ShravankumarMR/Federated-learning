from typing import Any

from pydantic import BaseModel


class AgentResponse(BaseModel):
    agent: str
    result: dict[str, Any]


class OrchestrationResponse(BaseModel):
    user_id: str
    session_id: str
    decision: str
    confidence: float
    details: dict[str, Any]
