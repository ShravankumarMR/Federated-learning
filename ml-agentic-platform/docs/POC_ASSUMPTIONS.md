# POC Assumptions and Missing Criteria

## Implemented Assumptions
- Input transport is Streamlit for the POC instead of a production message bus.
- Adaptive MFA is emitted as a policy recommendation only (`allow`, `step_up_mfa`, `deny_review`).
- Agent execution is best-effort: failed agents degrade to neutral scoring and are captured in audit evidence.
- Existing agent contracts are reused without model retraining or inference-service decomposition.

## Missing Criteria Identified
- Threshold ownership and calibration process for decision and MFA cutoffs.
- External IdP / MFA provider integration contract.
- Tenant-specific policy override model (weights, thresholds, escalation destinations).
- SLO/SLA targets for dispatch latency and end-to-end response times.
- Production dead-letter queue backend and retention policy.

## Migration Path
- Replace Streamlit ingest with Kafka/PubSub consumer that creates the same `ParamChangeEvent` payload.
- Keep orchestration service and LangGraph contracts unchanged while swapping only ingress adapters.
