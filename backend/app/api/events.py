import json

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.event_repository import list_recent_events
from app.schemas import StoredEvent

router = APIRouter(tags=["events"])


@router.get("/events", response_model=list[StoredEvent])
def get_events(limit: int = 25, db: Session = Depends(get_db)) -> list[dict[str, object]]:
    records = list_recent_events(db, limit=min(limit, 100))

    return [
        {
            "id": record.id,
            "article_hash": record.article_hash,
            "title": record.title,
            "description": record.description,
            "publishedAt": record.published_at,
            "event_type": record.event_type,
            "affected_sectors": json.loads(record.affected_sectors),
            "impact_direction": record.impact_direction,
            "confidence": record.confidence,
            "severity": record.severity,
            "reasoning": record.reasoning,
            "mapped_assets": json.loads(record.mapped_assets),
            "created_at": record.created_at,
        }
        for record in records
    ]
