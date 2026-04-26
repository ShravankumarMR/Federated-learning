import mlflow

from app.core.settings import get_settings


class TrackingClient:
    def __init__(self, tracking_uri: str | None = None) -> None:
        settings = get_settings()
        uri = tracking_uri or settings.mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)

    def set_experiment(self, name: str) -> None:
        mlflow.set_experiment(name)

    def log_metric(self, run_id: str, key: str, value: float) -> None:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric(key, value)
