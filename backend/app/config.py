from urllib.parse import urlparse

from pydantic import computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str
    DATABASE_URL: str
    OPENAI_API_KEY: str
    OPENAI_CHAT_MODEL: str
    OPENAI_FAST_MODEL: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIMENSIONS: int = 1536
    ALLOWED_ORIGINS: str = "http://localhost:5173"
    # LangSmith reads these from os.environ itself; they are mirrored here and the
    # tracer is built explicitly from them so .env values are honoured.
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "document-copilot"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    @field_validator(
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "DATABASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_CHAT_MODEL",
        "OPENAI_FAST_MODEL",
        "OPENAI_EMBEDDING_MODEL",
    )
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @field_validator("OPENAI_CHAT_MODEL", "OPENAI_FAST_MODEL")
    @classmethod
    def require_openai_chat_model(cls, value: str) -> str:
        if not value.startswith("openai:") or not value.removeprefix("openai:"):
            raise ValueError("must start with 'openai:' and include a model name")
        return value

    @field_validator("SUPABASE_URL")
    @classmethod
    def require_http_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("must be an HTTP URL")
        return value.rstrip("/")

    @field_validator("DATABASE_URL")
    @classmethod
    def require_postgres_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"postgresql", "postgresql+psycopg"} or not parsed.netloc:
            raise ValueError("must be a PostgreSQL URL")
        return value

    @field_validator("OPENAI_EMBEDDING_DIMENSIONS")
    @classmethod
    def require_positive_dimensions(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value

    @model_validator(mode="after")
    def require_langsmith_key_when_tracing(self) -> "Settings":
        if self.LANGSMITH_TRACING and not (self.LANGSMITH_API_KEY or "").strip():
            raise ValueError("LANGSMITH_API_KEY is required when LANGSMITH_TRACING is true")
        return self

    @computed_field
    @property
    def allowed_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    @computed_field
    @property
    def sqlalchemy_database_url(self) -> str:
        if self.DATABASE_URL.startswith("postgresql://"):
            return self.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
        return self.DATABASE_URL


settings = Settings()
