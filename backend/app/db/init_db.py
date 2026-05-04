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
    ("outcomes", "return_bucket", "TEXT NOT NULL DEFAULT 'flat'"),
    ("outcomes", "benchmark_return", "FLOAT"),
    ("outcomes", "excess_return", "FLOAT"),
    ("outcomes", "benchmark_label", "INTEGER"),
    ("outcomes", "filtered_label", "INTEGER"),
    ("outcomes", "threshold_used", "FLOAT NOT NULL DEFAULT 0.002"),
    ("events", "text_embedding", "TEXT"),
    ("predictions", "model_version", "TEXT NOT NULL DEFAULT 'unknown'"),
    ("multi_horizon_outcomes", "return_bucket", "TEXT NOT NULL DEFAULT 'flat'"),
    ("multi_horizon_outcomes", "benchmark_return", "FLOAT"),
    ("multi_horizon_outcomes", "excess_return", "FLOAT"),
    ("multi_horizon_outcomes", "benchmark_label", "INTEGER"),
    ("events", "content_hash", "VARCHAR(64)"),
]


def _migrate_columns() -> None:
    with engine.connect() as conn:
        for table, column, col_type in _NEW_COLUMNS:
            if not _column_exists(conn, table, column):
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
                conn.commit()


def _column_exists(conn, table: str, column: str) -> bool:
    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return any(row[1] == column for row in rows)


def _migrate_event_content_hash() -> None:
    """Deduplicate events and establish a unique content_hash index."""
    with engine.connect() as conn:
        # Remove duplicate rows, keeping the earliest insert per (timestamp, raw_text, source).
        # Orphaned predictions for deleted events are acceptable — they were duplicate data.
        duplicate_groups = conn.execute(text("""
            SELECT COUNT(*)
            FROM (
                SELECT 1
                FROM events
                GROUP BY timestamp, raw_text, source
                HAVING COUNT(*) > 1
            )
        """)).scalar()
        if duplicate_groups:
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
        if rows:
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
                return_bucket TEXT NOT NULL DEFAULT 'flat',
                benchmark_return FLOAT,
                excess_return FLOAT,
                benchmark_label INTEGER,
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
                   'flat', NULL, NULL, NULL, label, filtered_label, threshold_used, computed_at
            FROM outcomes
        """))
        conn.execute(text("DROP TABLE outcomes"))
        conn.execute(text("ALTER TABLE outcomes_new RENAME TO outcomes"))
        conn.commit()


def _backfill_derived_columns() -> None:
    with engine.connect() as conn:
        did_backfill = False
        for table in ("outcomes", "multi_horizon_outcomes"):
            needs_bucket_backfill = conn.execute(text(f"""
                SELECT COUNT(*) FROM {table}
                WHERE return_bucket IS NULL
                   OR (return_bucket = 'flat' AND ABS(raw_return) >= 0.002)
            """)).scalar()
            if needs_bucket_backfill:
                did_backfill = True
                conn.execute(text(f"""
                    UPDATE {table}
                    SET return_bucket = CASE
                        WHEN ABS(raw_return) < 0.002 THEN 'flat'
                        WHEN ABS(raw_return) < 0.01 THEN 'small'
                        WHEN ABS(raw_return) < 0.03 THEN 'medium'
                        ELSE 'large'
                    END
                    WHERE return_bucket IS NULL
                       OR (return_bucket = 'flat' AND ABS(raw_return) >= 0.002)
                """))

        needs_model_backfill = conn.execute(text("""
            SELECT COUNT(*) FROM predictions
            WHERE model_version IS NULL OR model_version = 'unknown'
        """)).scalar()
        if needs_model_backfill:
            did_backfill = True
            conn.execute(text("""
                UPDATE predictions
                SET model_version = CASE
                    WHEN horizon = 'multi' THEN 'historical_seed'
                    ELSE 'baseline-v1'
                END
                WHERE model_version IS NULL OR model_version = 'unknown'
            """))
        if did_backfill:
            conn.commit()


def init_db() -> None:
    ensure_sqlite_parent_directory(settings.database_url)
    Base.metadata.create_all(bind=engine)
    _migrate_columns()
    _fix_outcomes_nullable_label()
    _backfill_derived_columns()
    _migrate_event_content_hash()
