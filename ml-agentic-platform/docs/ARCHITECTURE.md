# Architecture Overview

## Layers
- API layer: FastAPI endpoints and request/response contracts.
- Orchestration layer: LangGraph state machine coordinating all agents.
- Agent layer: biometric, graph fraud, federated, and RAG agents.
- Data engineering layer: ETL and feature building.
- MLOps layer: model tracking, registry, and monitoring.

## Runtime Flow
1. API receives orchestration request.
2. LangGraph runs agent nodes in sequence.
3. Decision node fuses risk signals into final outcome.
4. API responds with decision, confidence, and per-agent details.

## Deployment
- Local: Docker Compose with API and MLflow.
- Cluster: Kubernetes deployment and service manifests.
- CI: lint, type-check, and test workflow.
