"""
Centralized configuration for URCM.

Uses Pydantic Settings for validated, type-safe configuration from environment variables.
All env vars are prefixed with URCM_ for namespace isolation.
"""

from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """URCM application settings loaded from environment variables."""

    # --- Server ---
    metrics_bind: str = Field(default="127.0.0.1", alias="URCM_METRICS_BIND")
    metrics_port: int = Field(default=8008, alias="URCM_METRICS_PORT")
    env: str = Field(default="", alias="URCM_ENV")
    cors_origins: str = Field(default="*", alias="URCM_CORS_ORIGINS")
    rate_limit: int = Field(default=100, alias="URCM_RATE_LIMIT")

    # --- Authentication ---
    metrics_token: str = Field(default="", alias="URCM_METRICS_TOKEN")
    admin_key: str = Field(default="", alias="URCM_ADMIN_KEY")

    # --- SLO Thresholds ---
    slo_min_final_mu: float = Field(default=0.4, alias="URCM_SLO_MIN_FINAL_MU")
    slo_max_steps_rate: float = Field(default=0.5, alias="URCM_SLO_MAX_STEPS_RATE")

    # --- Logging ---
    log_max_bytes: int = Field(default=5_000_000, alias="URCM_LOG_MAX_BYTES")
    log_backups: int = Field(default=5, alias="URCM_LOG_BACKUPS")
    run_id: str = Field(default="", alias="URCM_RUN_ID")

    @field_validator("cors_origins")
    @classmethod
    def parse_cors(cls, v: str) -> str:
        return v

    @property
    def cors_origins_list(self) -> List[str]:
        return self.cors_origins.split(",")

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @field_validator("admin_key")
    @classmethod
    def validate_admin_key_for_production(cls, v: str, info) -> str:
        """Validate admin_key is set for production environments."""
        env = info.data.get("env", "")
        if env == "production" and not v:
            raise ValueError("URCM_ADMIN_KEY must be set in production environments")
        return v

    model_config = {"env_prefix": "", "extra": "ignore"}


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from environment (useful for testing)."""
    global _settings
    _settings = Settings()
    return _settings
