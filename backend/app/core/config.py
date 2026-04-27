import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _default_database_url() -> str:
    local_data_root = Path(os.getenv("LOCALAPPDATA") or Path.home() / ".local" / "share")
    database_path = (
        local_data_root
        / "macro-event-intelligence"
        / "macro_event_intelligence.db"
    )
    return f"sqlite:///{database_path.as_posix()}"


class Settings:
    database_url: str = os.getenv("DATABASE_URL") or _default_database_url()
    news_api_key: str | None = os.getenv("NEWS_API_KEY")
    news_api_url: str = os.getenv(
        "NEWS_API_URL",
        "https://newsapi.org/v2/everything",
    )
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    news_cache_ttl_seconds: int = int(os.getenv("NEWS_CACHE_TTL_SECONDS", "900"))
    price_cache_ttl_seconds: int = int(os.getenv("PRICE_CACHE_TTL_SECONDS", "60"))
    classification_cache_ttl_seconds: int = int(
        os.getenv("CLASSIFICATION_CACHE_TTL_SECONDS", "86400")
    )


settings = Settings()
