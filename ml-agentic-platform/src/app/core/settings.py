from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "dev"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    mlflow_tracking_uri: str = "http://localhost:5000"
    model_registry_uri: str = "http://localhost:5000"

    rag_vector_store: str = "local"
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    federated_rounds: int = 5
    federated_min_clients: int = 3


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
