from fastapi import APIRouter

from app.services.health import health_payload

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", summary="Health check")
def health() -> dict[str, str]:
    return health_payload()
