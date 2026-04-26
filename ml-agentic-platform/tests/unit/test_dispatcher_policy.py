from app.orchestration.langgraph.graph import decision_node, dispatcher_node


def test_dispatcher_uses_event_mapping() -> None:
    state = {
        "param_change_event": {
            "change_type": "biometric_drift",
            "delta_magnitude": 0.4,
            "source": "unit-test",
            "affected_agents": [],
        }
    }

    output = dispatcher_node(state)
    assert output["selected_agents"] == ["biometric", "federated", "rag"]


def test_dispatcher_adds_federated_for_high_delta() -> None:
    state = {
        "param_change_event": {
            "change_type": "rag_context_change",
            "delta_magnitude": 0.9,
            "source": "unit-test",
            "affected_agents": ["rag"],
        }
    }

    output = dispatcher_node(state)
    assert "rag" in output["selected_agents"]
    assert "federated" in output["selected_agents"]


def test_decision_node_sets_step_up_mfa() -> None:
    state = {
        "biometric_result": {"risk": 0.8},
        "graph_result": {"risk": 0.7},
        "federated_result": {"risk": 0.6},
        "rag_result": {"confidence": 0.4},
    }

    output = decision_node(state)
    assert output["adaptive_mfa"] == "step_up_mfa"
    assert output["final_decision"] == "fraud"
    assert 0.55 <= output["risk_score"] < 0.8
