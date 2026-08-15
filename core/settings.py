from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FastAPI Scaffold"
    environment: str = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e4b"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
