import time

from fastapi import APIRouter

from app.mlops.monitoring.metrics import REQUEST_COUNT, REQUEST_LATENCY
from app.orchestration.service import OrchestrationService
from app.schemas.requests import OrchestrationRequest
from app.schemas.responses import OrchestrationResponse

router = APIRouter(prefix="/orchestrate", tags=["orchestration"])
service = OrchestrationService()


@router.post("", response_model=OrchestrationResponse, summary="Run multi-agent fraud workflow")
def orchestrate(request: OrchestrationRequest) -> OrchestrationResponse:
    started = time.perf_counter()
    endpoint = "/orchestrate"
    method = "POST"
    try:
        response = service.run(request)
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="200").inc()
        return response
    except Exception:
        REQUEST_COUNT.labels(endpoint=endpoint, method=method, status="500").inc()
        raise
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(
            max(time.perf_counter() - started, 0.0)
        )
