from pathlib import Path
from typing import List
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # General
    ENV: str = Field(default="dev", env="ENV")
    DEBUG: bool = Field(default=False, env="DEBUG")
    APP_NAME: str = Field(default="Aura Agent")

    # Security and secrets
    SECRET_KEY: str = Field(default="changeme-in-prod", env="SECRET_KEY")
    ALLOWED_ORIGINS: List[str] = Field(default_factory=lambda: ["*"], env="ALLOWED_ORIGINS")

    # Paths
    BASE_DIR: Path = BASE_DIR
    MODULES_DIR: Path = BASE_DIR / "agent_service" / "modules"
    DOCS_DIR: Path = BASE_DIR / "docs"

    # Database
    POSTGRES_URL: str = Field(
        default="postgresql://user:pass@localhost:5432/aura_db", env="POSTGRES_URL"
    )

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    REDIS_MAX_CONNECTIONS: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")

    # Memory Configuration
    MEMORY_TTL: int = Field(default=3600, env="MEMORY_TTL")
    VECTOR_STORE_SIZE: int = Field(default=1000, env="VECTOR_STORE_SIZE")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")

    # Workflow
    MAX_WORKFLOW_STEPS: int = Field(default=50, env="MAX_WORKFLOW_STEPS")
    WORKFLOW_TIMEOUT: int = Field(default=1800, env="WORKFLOW_TIMEOUT")

    # Human-in-the-loop
    HUMAN_APPROVAL_TIMEOUT: int = Field(default=3600, env="HUMAN_APPROVAL_TIMEOUT")

    # LLM API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    GROQ_API_KEY: Optional[str] = Field(default=None, env="GROQ_API_KEY")

    # Model Defaults
    DEFAULT_MODEL_PROVIDER: str = Field(default="openai", env="DEFAULT_MODEL_PROVIDER")
    DEFAULT_MODEL_NAME: str = Field(default="gpt-4", env="DEFAULT_MODEL_NAME")

    # Banking Module Configuration
    ECL_CALCULATION_TIMEOUT: int = Field(default=300, env="ECL_CALCULATION_TIMEOUT")  # 5 minutes
    RWA_CALCULATION_TIMEOUT: int = Field(default=600, env="RWA_CALCULATION_TIMEOUT")  # 10 minutes
    MODEL_VALIDATION_TIMEOUT: int = Field(default=180, env="MODEL_VALIDATION_TIMEOUT")  # 3 minutes

    class Config:
        env_file = ".env"
        case_sensitive = True


# Instantiate global settings object
settings = Settings()
