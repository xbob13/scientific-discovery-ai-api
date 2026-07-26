from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./research_lab.db"
    backend_service_token: SecretStr = SecretStr("development-only-token")
    research_kill_switch: bool = False
    max_run_cost_usd: float = 5.0
    max_run_seconds: int = 900
    crossref_mailto: str = "research@example.com"
    openalex_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
