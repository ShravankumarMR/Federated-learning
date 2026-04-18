from fastapi import APIRouter

from app.orchestration.service import OrchestrationService
from app.schemas.requests import OrchestrationRequest
from app.schemas.responses import OrchestrationResponse

router = APIRouter(prefix="/orchestrate", tags=["orchestration"])
service = OrchestrationService()


@router.post("", response_model=OrchestrationResponse, summary="Run multi-agent fraud workflow")
def orchestrate(request: OrchestrationRequest) -> OrchestrationResponse:
    return service.run(request)
