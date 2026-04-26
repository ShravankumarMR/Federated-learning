from typing import Any, TypedDict


class FraudState(TypedDict, total=False):
    user_id: str
    session_id: str
    correlation_id: str
    dataset: str
    param_change_event: dict[str, Any]
    biometric_payload: dict[str, Any]
    graph_payload: dict[str, Any]
    federated_payload: dict[str, Any]
    rag_query: str
    selected_agents: list[str]
    agent_errors: dict[str, str]
    biometric_result: dict[str, Any]
    graph_result: dict[str, Any]
    federated_result: dict[str, Any]
    rag_result: dict[str, Any]
    final_decision: str
    final_confidence: float
    risk_score: float
    adaptive_mfa: str
    policy_version: str
    explanation: dict[str, Any]
    details: dict[str, Any]
