from app.orchestration.service import OrchestrationService
from app.schemas.requests import OrchestrationRequest, ParamChangeEvent


def test_orchestration_service_smoke() -> None:
    service = OrchestrationService()
    response = service.run(
        OrchestrationRequest(
            user_id="user-1",
            session_id="session-1",
            dataset="ieee_cis",
            param_change_event=ParamChangeEvent(
                change_type="mixed",
                delta_magnitude=0.6,
                changed_parameters=["global_head"],
            ),
            biometric_payload={"mean_velocity": 0.6},
            graph_payload={"node_degree": 2, "shared_devices": 1},
            federated_payload={"participating_clients": 4, "global_model_quality": 0.8},
            rag_query="similar fraud cases",
        )
    )

    assert response.decision in {"fraud", "legit"}
    assert 0.0 <= response.confidence <= 1.0
    assert 0.0 <= response.risk_score <= 1.0
    assert response.adaptive_mfa in {"allow", "step_up_mfa", "deny_review"}
    assert response.correlation_id
    assert response.policy_version == "adaptive-mfa-v1"
    assert "summary" in response.explanation.model_dump()


def test_orchestration_service_paysim() -> None:
    service = OrchestrationService()
    response = service.run(
        OrchestrationRequest(
            user_id="user-2",
            session_id="session-2",
            dataset="paysim",
            param_change_event=ParamChangeEvent(
                change_type="graph_drift",
                delta_magnitude=0.85,
                changed_parameters=["temporal_edges"],
            ),
            biometric_payload={"mean_velocity": 0.8},
            graph_payload={"node_degree": 5, "shared_devices": 3},
            federated_payload={"participating_clients": 3, "global_model_quality": 0.7},
            rag_query="paysim fraud patterns",
        )
    )

    assert response.decision in {"fraud", "legit"}
    assert 0.0 <= response.confidence <= 1.0
    assert "graph_fraud" in response.invoked_agents
    assert response.adaptive_mfa in {"allow", "step_up_mfa", "deny_review"}
