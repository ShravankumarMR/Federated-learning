from typing import Any, TypedDict


class FraudState(TypedDict, total=False):
    user_id: str
    session_id: str
    biometric_payload: dict[str, Any]
    graph_payload: dict[str, Any]
    federated_payload: dict[str, Any]
    rag_query: str
    biometric_result: dict[str, Any]
    graph_result: dict[str, Any]
    federated_result: dict[str, Any]
    rag_result: dict[str, Any]
    final_decision: str
    final_confidence: float
    details: dict[str, Any]
