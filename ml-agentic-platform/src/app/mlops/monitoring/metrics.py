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
