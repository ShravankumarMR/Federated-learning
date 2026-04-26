from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "api_request_count",
    "Total API requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint", "method"],
)

EVENT_INGEST_COUNT = Counter(
    "event_ingest_count",
    "Total parameter-change events ingested",
    ["source", "change_type"],
)

DISPATCH_LATENCY = Histogram(
    "dispatch_latency_seconds",
    "Task dispatcher latency in seconds",
)

AGENT_LATENCY = Histogram(
    "agent_latency_seconds",
    "Agent execution latency in seconds",
    ["agent"],
)

AGENT_ERROR_COUNT = Counter(
    "agent_error_count",
    "Total agent execution errors",
    ["agent"],
)

DECISION_DISTRIBUTION = Counter(
    "decision_distribution_count",
    "Final decision distribution",
    ["decision"],
)

MFA_TRIGGER_COUNT = Counter(
    "adaptive_mfa_trigger_count",
    "Adaptive MFA action distribution",
    ["action"],
)
