from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    DATABASE_URL: str = "postgresql://forge:forge@localhost:5432/forge"

    # Inference backends
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLAMACPP_SERVER_URL: str = "http://localhost:8080"

    # API Keys (optional)
    ANTHROPIC_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    HF_TOKEN: str = ""

    # FORGE directories
    FORGE_SKILLS_DIR: str = "~/.forge/skills"
    FORGE_MODELS_DIR: str = "~/.forge/models"
    FORGE_DATASETS_DIR: str = "~/.forge/datasets"
    FORGE_RUNS_DIR: str = "~/.forge/runs"
    FORGE_EXPORTS_DIR: str = "~/.forge/exports"

    # Hardware
    HARDWARE_CHECK_INTERVAL_SECONDS: int = 30

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]


settings = Settings()
