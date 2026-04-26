from uuid import uuid4

from app.mlops.monitoring.metrics import DECISION_DISTRIBUTION, EVENT_INGEST_COUNT, MFA_TRIGGER_COUNT
from app.orchestration.langgraph.graph import build_orchestration_graph
from app.schemas.requests import OrchestrationRequest
from app.schemas.responses import ExplanationPayload, OrchestrationResponse


class OrchestrationService:
    def __init__(self) -> None:
        self.workflow = build_orchestration_graph()

    def run(self, request: OrchestrationRequest) -> OrchestrationResponse:
        correlation_id = request.correlation_id or str(uuid4())
        event = request.param_change_event

        if event is not None:
            EVENT_INGEST_COUNT.labels(source=event.source, change_type=event.change_type).inc()

        state = {
            "user_id": request.user_id,
            "session_id": request.session_id,
            "correlation_id": correlation_id,
            "dataset": request.dataset,
            "param_change_event": (
                event.model_dump(mode="json")
                if event is not None
                else {
                    "event_id": f"fallback-{correlation_id}",
                    "model_id": "global-federated-model",
                    "change_type": "mixed",
                    "changed_parameters": [],
                    "delta_magnitude": 0.5,
                    "source": "legacy-request",
                    "affected_agents": [],
                }
            ),
            "biometric_payload": request.biometric_payload,
            "graph_payload": request.graph_payload,
            "federated_payload": request.federated_payload,
            "rag_query": request.rag_query,
        }

        output = self.workflow.invoke(state)
        response = OrchestrationResponse(
            user_id=request.user_id,
            session_id=request.session_id,
            correlation_id=correlation_id,
            decision=output.get("final_decision", "unknown"),
            confidence=output.get("final_confidence", 0.0),
            risk_score=output.get("risk_score", 0.5),
            adaptive_mfa=output.get("adaptive_mfa", "allow"),
            invoked_agents=output.get("selected_agents", []),
            policy_version=output.get("policy_version", "adaptive-mfa-v1"),
            explanation=ExplanationPayload.model_validate(
                output.get(
                    "explanation",
                    {
                        "summary": "No explanation generated.",
                        "trigger": "unknown",
                        "evidence": {},
                        "policy_threshold": 0.55,
                        "policy_version": "adaptive-mfa-v1",
                    },
                )
            ),
            details=output.get("details", {}),
        )
        DECISION_DISTRIBUTION.labels(decision=response.decision).inc()
        MFA_TRIGGER_COUNT.labels(action=response.adaptive_mfa).inc()
        return response
