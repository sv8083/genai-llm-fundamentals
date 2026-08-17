from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "FastAPI Scaffold"
    environment: str = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    env: str = "local"  # Environment for Phoenix project naming
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    phoenix_otlp_endpoint: str = "http://localhost:6006/v1/traces"
    ollama_temperature: float = 0.7
    ollama_top_p: float = 0.9

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @property
    def phoenix_project_name(self) -> str:
        """Generate Phoenix project name from environment."""
        return f"llm-fundamental-{self.env}"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
