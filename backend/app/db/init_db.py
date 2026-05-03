import hashlib

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
    ("outcomes", "raw_return", "FLOAT NOT NULL DEFAULT 0.0"),
    ("outcomes", "return_magnitude", "FLOAT NOT NULL DEFAULT 0.0"),
    ("outcomes", "filtered_label", "INTEGER"),
    ("outcomes", "threshold_used", "FLOAT NOT NULL DEFAULT 0.002"),
    ("events", "text_embedding", "TEXT"),
    ("events", "content_hash", "VARCHAR(64)"),
]


def _migrate_columns() -> None:
    with engine.connect() as conn:
        for table, column, col_type in _NEW_COLUMNS:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
                conn.commit()
            except Exception:
                pass  # column already exists


def _migrate_event_content_hash() -> None:
    """Deduplicate events and establish a unique content_hash index."""
    with engine.connect() as conn:
        # Remove duplicate rows, keeping the earliest insert per (timestamp, raw_text, source).
        # Orphaned predictions for deleted events are acceptable — they were duplicate data.
        conn.execute(text("""
            DELETE FROM events
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM events
                GROUP BY timestamp, raw_text, source
            )
        """))
        conn.commit()

        # Populate content_hash for any rows that don't have one yet.
        rows = conn.execute(
            text("SELECT id, timestamp, raw_text, source FROM events WHERE content_hash IS NULL")
        ).fetchall()
        for row in rows:
            content = f"{row[1]}|{row[2]}|{row[3]}"
            hash_val = hashlib.sha256(content.encode()).hexdigest()
            conn.execute(
                text("UPDATE events SET content_hash = :h WHERE id = :id"),
                {"h": hash_val, "id": str(row[0])},
            )
        conn.commit()

        # Create unique index (idempotent).
        conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_events_content_hash ON events (content_hash)"
        ))
        conn.commit()


def _fix_outcomes_nullable_label() -> None:
    """Rebuild outcomes table if label column was created NOT NULL (old schema)."""
    with engine.connect() as conn:
        rows = conn.execute(text("PRAGMA table_info(outcomes)")).fetchall()
        label_col = next((r for r in rows if r[1] == "label"), None)
        if label_col is None or label_col[3] == 0:
            return  # column missing (handled by create_all) or already nullable

        conn.execute(text("""
            CREATE TABLE outcomes_new (
                id VARCHAR NOT NULL,
                prediction_id VARCHAR NOT NULL,
                actual_return FLOAT NOT NULL,
                raw_return FLOAT NOT NULL DEFAULT 0.0,
                return_magnitude FLOAT NOT NULL DEFAULT 0.0,
                label INTEGER,
                filtered_label INTEGER,
                threshold_used FLOAT NOT NULL DEFAULT 0.002,
                computed_at DATETIME,
                PRIMARY KEY (id),
                UNIQUE (prediction_id),
                FOREIGN KEY(prediction_id) REFERENCES predictions (id)
            )
        """))
        conn.execute(text("""
            INSERT INTO outcomes_new
            SELECT id, prediction_id, actual_return, raw_return, return_magnitude,
                   label, filtered_label, threshold_used, computed_at
            FROM outcomes
        """))
        conn.execute(text("DROP TABLE outcomes"))
        conn.execute(text("ALTER TABLE outcomes_new RENAME TO outcomes"))
        conn.commit()


def init_db() -> None:
    ensure_sqlite_parent_directory(settings.database_url)
    Base.metadata.create_all(bind=engine)
    _migrate_columns()
    _fix_outcomes_nullable_label()
    _migrate_event_content_hash()
