"""
Application Configuration

Central configuration management for the backend.

All environment variables are loaded from .env using
Pydantic Settings.

Do not hardcode secrets anywhere else.
"""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Backend application settings.
    """

    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------

    APP_NAME: str = "AI-Powered Financial Analyst"

    APP_VERSION: str = "1.0.0"

    APP_DESCRIPTION: str = (
        "AI-Powered Financial Analyst and Investment "
        "Decision Support System"
    )

    API_V1_PREFIX: str = "/api/v1"

    DEBUG: bool = True

    ENVIRONMENT: str = Field(
        default="development",
        description="development | testing | production",
    )

    # ---------------------------------------------------------
    # Server
    # ---------------------------------------------------------

    HOST: str = "0.0.0.0"

    PORT: int = 8000

    # ---------------------------------------------------------
    # Database
    # ---------------------------------------------------------

    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/financial_analyst"
    )

    # ---------------------------------------------------------
    # JWT
    # (Used in Phase 2)
    # ---------------------------------------------------------

    SECRET_KEY: str = Field(
        default="CHANGE_ME_IN_PRODUCTION"
    )

    ALGORITHM: str = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ---------------------------------------------------------
    # CORS
    # ---------------------------------------------------------

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
    ]

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------

    LOG_LEVEL: str = "INFO"

    LOG_FILE: str = "storage/logs/backend.log"

    # ---------------------------------------------------------
    # AI Module Paths
    # ---------------------------------------------------------

    AI_MODULE_PATH: str = "../ai"

    MODEL_DIRECTORY: str = "../storage/models"

    REPORT_DIRECTORY: str = "../storage/reports"

    # ---------------------------------------------------------
    # Settings Configuration
    # ---------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    Ensures configuration is loaded only once
    during application lifetime.
    """

    return Settings()


settings = get_settings()