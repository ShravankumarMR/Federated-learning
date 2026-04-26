from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import streamlit as st

from app.orchestration.service import OrchestrationService
from app.schemas.requests import OrchestrationRequest, ParamChangeEvent


@st.cache_resource
def _get_service() -> OrchestrationService:
    return OrchestrationService()

_SAMPLE_EVENTS: dict[str, dict] = {
    "Graph drift (PaySim)": {
        "dataset": "paysim",
        "change_type": "graph_drift",
        "delta_magnitude": 0.82,
        "changed_parameters": ["temporal_edges", "edge_attention"],
        "affected_agents": ["graph_fraud", "federated", "rag"],
        "graph_payload": {"node_degree": 6, "shared_devices": 4},
    },
    "Biometric drift": {
        "dataset": "ieee_cis",
        "change_type": "biometric_drift",
        "delta_magnitude": 0.64,
        "changed_parameters": ["mouse_velocity_embeddings"],
        "affected_agents": ["biometric", "federated", "rag"],
        "biometric_payload": {"mean_velocity": 0.2},
    },
    "Mixed drift": {
        "dataset": "ieee_cis",
        "change_type": "mixed",
        "delta_magnitude": 0.9,
        "changed_parameters": ["fed_head", "graph_encoder", "retrieval_conf"],
        "affected_agents": ["biometric", "graph_fraud", "federated", "rag"],
    },
}


def _build_request(payload: dict) -> OrchestrationRequest:
    event = ParamChangeEvent(
        model_id=payload.get("model_id", "global-federated-model"),
        change_type=payload.get("change_type", "mixed"),
        changed_parameters=payload.get("changed_parameters", []),
        delta_magnitude=float(payload.get("delta_magnitude", 0.5)),
        source="streamlit-poc",
        timestamp=datetime.now(timezone.utc),
        affected_agents=payload.get("affected_agents", []),
    )
    return OrchestrationRequest(
        user_id=payload.get("user_id", "demo-user"),
        session_id=payload.get("session_id", f"session-{uuid4().hex[:8]}"),
        correlation_id=payload.get("correlation_id", str(uuid4())),
        dataset=payload.get("dataset", "ieee_cis"),
        param_change_event=event,
        biometric_payload=payload.get("biometric_payload", {"mean_velocity": 0.4}),
        graph_payload=payload.get("graph_payload", {"node_degree": 2, "shared_devices": 1}),
        federated_payload=payload.get(
            "federated_payload",
            {"participating_clients": 4, "global_model_quality": 0.78},
        ),
        rag_query=payload.get("rag_query", "relevant fraud cases for explanation"),
    )


def _submit_local(req: OrchestrationRequest) -> dict:
    return _get_service().run(req).model_dump(mode="json")


def _submit_via_api(req: OrchestrationRequest) -> dict:
    api_url = os.getenv("POC_ORCHESTRATION_URL", "http://127.0.0.1:8000/orchestrate")
    with httpx.Client(timeout=20.0) as client:
        response = client.post(api_url, json=req.model_dump(mode="json"))
        response.raise_for_status()
        return response.json()


def main() -> None:
    st.set_page_config(page_title="Federated Drift Risk Console", layout="wide")
    st.title("Federated Parameter Change Orchestration (POC)")
    st.caption("Streamlit intake UI/API for dispatcher-driven multi-agent fraud risk decisions")

    mode = st.radio(
        "Execution mode",
        options=["Local service", "REST API"],
        horizontal=True,
        help="Local service runs orchestration in-process; REST API calls FastAPI /orchestrate.",
    )

    sample_name = st.selectbox("Sample payload", options=list(_SAMPLE_EVENTS.keys()))
    payload = dict(_SAMPLE_EVENTS[sample_name])

    with st.expander("Edit payload"):
        raw = st.text_area("Payload JSON", value=json.dumps(payload, indent=2), height=300)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON payload: {exc}")
            return

    if st.button("Dispatch change event", type="primary"):
        try:
            request = _build_request(payload)
            result = _submit_local(request) if mode == "Local service" else _submit_via_api(request)
        except Exception as exc:  # pragma: no cover - UI error reporting path
            st.exception(exc)
            return

        st.subheader("Decision")
        c1, c2, c3 = st.columns(3)
        c1.metric("Decision", result.get("decision", "unknown"))
        c2.metric("Risk score", result.get("risk_score", 0.0))
        c3.metric("Adaptive MFA", result.get("adaptive_mfa", "allow"))

        st.subheader("Explanation")
        st.json(result.get("explanation", {}))

        st.subheader("Full response")
        st.json(result)


if __name__ == "__main__":
    main()
