import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _normalize_database_url(database_url: str) -> str:
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+psycopg2://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return database_url


def _default_database_url() -> str:
    local_data_root = Path(os.getenv("LOCALAPPDATA") or Path.home() / ".local" / "share")
    database_path = (
        local_data_root
        / "macro-event-intelligence"
        / "macro_event_intelligence.db"
    )
    return f"sqlite:///{database_path.as_posix()}"


class Settings:
    database_url: str = _normalize_database_url(os.getenv("DATABASE_URL") or _default_database_url())
    frontend_origins: list[str] = [
        origin.strip()
        for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
        if origin.strip()
    ]
    news_api_key: str | None = os.getenv("NEWS_API_KEY")
    news_api_url: str = os.getenv(
        "NEWS_API_URL",
        "https://newsapi.org/v2/everything",
    )
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    fred_api_key: str | None = os.getenv("FRED_API_KEY")
    news_cache_ttl_seconds: int = int(os.getenv("NEWS_CACHE_TTL_SECONDS", "900"))
    price_cache_ttl_seconds: int = int(os.getenv("PRICE_CACHE_TTL_SECONDS", "60"))
    classification_cache_ttl_seconds: int = int(
        os.getenv("CLASSIFICATION_CACHE_TTL_SECONDS", "86400")
    )
    outcome_noise_threshold: float = float(os.getenv("OUTCOME_NOISE_THRESHOLD", "0.002"))


settings = Settings()
