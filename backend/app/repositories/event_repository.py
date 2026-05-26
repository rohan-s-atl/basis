import json
from datetime import datetime

from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.db.models import EventRecord, MarketSnapshot


def upsert_event_record(
    db: Session,
    *,
    article_hash: str,
    article: dict,
    classification: dict,
    mapped_assets: list[str],
    assets_json: str = "[]",
    price_at_event: float | None = None,
) -> EventRecord:
    values = {
        "article_hash": article_hash,
        "title": str(article.get("title") or ""),
        "description": str(article.get("description") or ""),
        "published_at": str(article.get("publishedAt") or ""),
        "event_type": str(classification["event_type"]),
        "affected_sectors": json.dumps(classification["affected_sectors"]),
        "impact_direction": str(classification["impact_direction"]),
        "confidence": float(classification["confidence"]),
        "severity": str(classification["severity"]),
        "reasoning": str(classification["reasoning"]),
        "mapped_assets": json.dumps(mapped_assets),
        "assets_json": assets_json,
        "price_at_event": price_at_event,
    }

    update_set = {k: v for k, v in values.items() if k != "article_hash"}
    if price_at_event is None:
        # Never overwrite an existing price with None
        del update_set["price_at_event"]

    if db.bind is not None and db.bind.dialect.name == "postgresql":
        stmt = postgres_insert(EventRecord).values(**values)
    else:
        stmt = sqlite_insert(EventRecord).values(**values)

    db.execute(stmt.on_conflict_do_update(index_elements=["article_hash"], set_=update_set))
    db.commit()

    record = get_event_by_article_hash(db, article_hash)
    return record


def get_event_by_article_hash(db: Session, article_hash: str) -> EventRecord | None:
    return (
        db.query(EventRecord)
        .filter(EventRecord.article_hash == article_hash)
        .one_or_none()
    )


def list_recent_events(db: Session, limit: int = 25) -> list[EventRecord]:
    return (
        db.query(EventRecord)
        .order_by(EventRecord.created_at.desc())
        .limit(limit)
        .all()
    )


def list_combined_from_db(db: Session, limit: int = 25) -> list[dict]:
    """Reconstruct CombinedArticle dicts from stored EventRecords."""
    records = list_recent_events(db, limit=limit)
    result = []
    for r in records:
        try:
            assets = json.loads(r.assets_json) if r.assets_json else []
        except Exception:
            assets = []
        try:
            sectors = json.loads(r.affected_sectors)
        except Exception:
            sectors = []
        result.append({
            "title": r.title,
            "description": r.description,
            "event_type": r.event_type,
            "affected_sectors": sectors,
            "impact_direction": r.impact_direction,
            "confidence": r.confidence,
            "severity": r.severity,
            "reasoning": r.reasoning,
            "assets": assets,
        })
    return result


def list_unevaluated_events(db: Session, before: datetime) -> list[EventRecord]:
    """Events older than `before` with a non-neutral direction and a recorded entry price."""
    return (
        db.query(EventRecord)
        .filter(
            EventRecord.evaluated_at.is_(None),
            EventRecord.signal_correct.is_(None),
            EventRecord.impact_direction != "neutral",
            EventRecord.price_at_event.isnot(None),
            EventRecord.created_at < before,
        )
        .all()
    )


def update_signal_outcome(
    db: Session,
    *,
    record_id: int,
    price_at_evaluation: float,
    signal_correct: bool,
) -> None:
    record = db.query(EventRecord).filter(EventRecord.id == record_id).one_or_none()
    if record is None:
        return
    record.price_at_evaluation = price_at_evaluation
    record.signal_correct = signal_correct
    record.evaluated_at = datetime.utcnow()
    db.commit()


def get_signal_stats(db: Session) -> dict:
    evaluated = (
        db.query(EventRecord)
        .filter(EventRecord.signal_correct.isnot(None))
        .order_by(EventRecord.evaluated_at.desc())
        .all()
    )

    if not evaluated:
        return {
            "total_evaluated": 0,
            "correct": 0,
            "accuracy_pct": 0.0,
            "by_event_type": {},
            "recent": [],
        }

    correct_count = sum(1 for r in evaluated if r.signal_correct)
    accuracy_pct = round(correct_count / len(evaluated) * 100, 1)

    by_event_type: dict[str, dict] = {}
    for r in evaluated:
        et = r.event_type
        if et not in by_event_type:
            by_event_type[et] = {"total": 0, "correct": 0, "accuracy_pct": 0.0}
        by_event_type[et]["total"] += 1
        if r.signal_correct:
            by_event_type[et]["correct"] += 1
    for et in by_event_type:
        t = by_event_type[et]
        t["accuracy_pct"] = round(t["correct"] / t["total"] * 100, 1)

    recent = evaluated[:5]
    return {
        "total_evaluated": len(evaluated),
        "correct": correct_count,
        "accuracy_pct": accuracy_pct,
        "by_event_type": by_event_type,
        "recent": [
            {
                "title": r.title,
                "event_type": r.event_type,
                "impact_direction": r.impact_direction,
                "signal_correct": r.signal_correct,
                "price_at_event": r.price_at_event,
                "price_at_evaluation": r.price_at_evaluation,
                "evaluated_at": r.evaluated_at.isoformat() if r.evaluated_at else None,
            }
            for r in recent
        ],
    }


def create_market_snapshot(
    db: Session,
    *,
    article_hash: str,
    symbol: str,
    price: float,
) -> MarketSnapshot:
    snapshot = MarketSnapshot(
        article_hash=article_hash,
        symbol=symbol,
        price=price,
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot
