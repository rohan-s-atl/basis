from collections.abc import Generator
import os
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import settings


def ensure_sqlite_parent_directory(database_url: str) -> None:
    if not database_url.startswith("sqlite:///") or database_url in {"sqlite://", "sqlite:///:memory:"}:
        return

    parsed = urlparse(database_url)
    database_path_value = parsed.path
    if os.name == "nt" and len(database_path_value) >= 4 and database_path_value[0] == "/" and database_path_value[2] == ":":
        database_path_value = database_path_value[1:]

    database_path = Path(database_path_value)
    if database_path.parent != Path("."):
        database_path.parent.mkdir(parents=True, exist_ok=True)


ensure_sqlite_parent_directory(settings.database_url)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
