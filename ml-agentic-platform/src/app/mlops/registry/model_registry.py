import mlflow

from app.core.settings import get_settings


class ModelRegistry:
    def __init__(self) -> None:
        settings = get_settings()
        mlflow.set_registry_uri(settings.model_registry_uri)

    def register(self, model_uri: str, name: str) -> str:
        result = mlflow.register_model(model_uri=model_uri, name=name)
        return result.version
