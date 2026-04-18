from app.orchestration.service import OrchestrationService
from app.schemas.requests import OrchestrationRequest


def test_orchestration_service_smoke() -> None:
    service = OrchestrationService()
    response = service.run(
        OrchestrationRequest(
            user_id="user-1",
            session_id="session-1",
            biometric_payload={"mean_velocity": 0.6},
            graph_payload={"node_degree": 2, "shared_devices": 1},
            federated_payload={"participating_clients": 4, "global_model_quality": 0.8},
            rag_query="similar fraud cases",
        )
    )

    assert response.decision in {"fraud", "legit"}
    assert 0.0 <= response.confidence <= 1.0
