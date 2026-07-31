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
    portal_token_signing_secret: SecretStr = SecretStr("development-portal-signing-secret")
    patentsview_api_key: SecretStr | None = None
    patentsview_search_url: str = "https://search.patentsview.org/api/v1/patent/"
    epo_ops_consumer_key: SecretStr | None = None
    epo_ops_consumer_secret: SecretStr | None = None
    epo_ops_auth_url: str = "https://ops.epo.org/3.2/auth/accesstoken"
    epo_ops_search_url: str = "https://ops.epo.org/3.2/rest-services/published-data/search"
    outreach_send_enabled: bool = False
    outreach_delivery_webhook_url: str | None = None
    outreach_delivery_token: SecretStr | None = None
    prospect_discovery_enabled: bool = False
    prospect_discovery_feed_url: str | None = None
    prospect_discovery_token: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
