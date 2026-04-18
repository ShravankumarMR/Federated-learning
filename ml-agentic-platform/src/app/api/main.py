from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.orchestrate import router as orchestration_router
from app.core.logging import setup_logging
from app.core.settings import get_settings

settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(
    title="ML Agentic Platform API",
    version="0.1.0",
    description="API for modular fraud intelligence agents and orchestration",
)

app.include_router(health_router)
app.include_router(orchestration_router)
