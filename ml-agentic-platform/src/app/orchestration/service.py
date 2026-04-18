from app.orchestration.langgraph.graph import build_orchestration_graph
from app.schemas.requests import OrchestrationRequest
from app.schemas.responses import OrchestrationResponse


class OrchestrationService:
    def __init__(self) -> None:
        self.workflow = build_orchestration_graph()

    def run(self, request: OrchestrationRequest) -> OrchestrationResponse:
        state = {
            "user_id": request.user_id,
            "session_id": request.session_id,
            "biometric_payload": request.biometric_payload,
            "graph_payload": request.graph_payload,
            "federated_payload": request.federated_payload,
            "rag_query": request.rag_query,
        }

        output = self.workflow.invoke(state)
        return OrchestrationResponse(
            user_id=request.user_id,
            session_id=request.session_id,
            decision=output.get("final_decision", "unknown"),
            confidence=output.get("final_confidence", 0.0),
            details=output.get("details", {}),
        )
