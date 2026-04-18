import mlflow

from app.core.settings import get_settings


class TrackingClient:
    def __init__(self) -> None:
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric(key, value)
