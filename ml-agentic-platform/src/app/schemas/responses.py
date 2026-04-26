from typing import Any, Literal

from pydantic import BaseModel


class AgentResponse(BaseModel):
    agent: str
    result: dict[str, Any]


class ExplanationPayload(BaseModel):
    summary: str
    trigger: str
    evidence: dict[str, Any]
    policy_threshold: float
    policy_version: str


class OrchestrationResponse(BaseModel):
    user_id: str
    session_id: str
    correlation_id: str
    decision: str
    confidence: float
    risk_score: float
    adaptive_mfa: Literal["allow", "step_up_mfa", "deny_review"]
    invoked_agents: list[str]
    policy_version: str
    explanation: ExplanationPayload
    details: dict[str, Any]
