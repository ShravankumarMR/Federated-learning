# Architecture Overview

## Layers
- API layer: FastAPI endpoints and request/response contracts.
- Orchestration layer: LangGraph state machine coordinating all agents.
- Agent layer: biometric, graph fraud, federated, and RAG agents.
- Data engineering layer: ETL and feature building.
- MLOps layer: model tracking, registry, and monitoring.

## Runtime Flow
1. API receives orchestration request.
2. LangGraph dispatcher maps the parameter-change event to required agents.
3. Selected agents execute in parallel (biometric, graph fraud, federated, RAG).
4. Decision node fuses available signals into risk score and adaptive MFA action.
5. Explanation node produces audit-ready rationale and execution metadata.
6. API responds with decision, confidence, risk score, MFA action, and details.

## Streamlit POC Intake
- POC ingress runs via Streamlit: `src/app/ui/streamlit_app.py`.
- Streamlit can invoke orchestration locally in-process or via REST (`/orchestrate`).
- Payloads use `ParamChangeEvent` to represent federated model parameter drift.

## Dispatcher and Policy
- Dispatcher selects agents from change type, optional affected-agent overrides, and drift magnitude.
- Adaptive MFA policy (`adaptive-mfa-v1`):
	- `risk < 0.55`: allow
	- `0.55 <= risk < 0.80`: step-up MFA
	- `risk >= 0.80`: deny/review

## Observability
- Structured JSON logs include correlation-ready fields.
- Prometheus metrics cover ingest volume, dispatch latency, per-agent latency/errors, decision distribution, and MFA trigger distribution.

## Deployment
- Local API: `uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload`.
- Local UI (optional): `streamlit run src/app/ui/streamlit_app.py --server.port 8501`.
- Local MLflow (optional): run MLflow server as a standalone local process if experiment tracking is needed.
- CI: lint, type-check, and test workflow.
