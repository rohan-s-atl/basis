from sqlalchemy import text

from app.db import models
from app.core.config import settings
from app.db.session import Base, engine, ensure_sqlite_parent_directory

_NEW_COLUMNS = [
    ("event_records", "assets_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("event_records", "price_at_event", "FLOAT"),
    ("event_records", "price_at_evaluation", "FLOAT"),
    ("event_records", "signal_correct", "BOOLEAN"),
    ("event_records", "evaluated_at", "DATETIME"),
]


def _migrate_columns() -> None:
    with engine.connect() as conn:
        for table, column, col_type in _NEW_COLUMNS:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
                conn.commit()
            except Exception:
                pass  # column already exists


def init_db() -> None:
    ensure_sqlite_parent_directory(settings.database_url)
    Base.metadata.create_all(bind=engine)
    _migrate_columns()
