import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.event_repository import list_combined_from_db
from app.schemas import CombinedArticle
from app.services.ingestion_service import ingest_events

logger = logging.getLogger(__name__)

router = APIRouter(tags=["combined"])


@router.get("/combined", response_model=list[CombinedArticle])
def get_combined(db: Session = Depends(get_db)) -> list[dict]:
    try:
        results = ingest_events(db)
        if results:
            return results
    except Exception as exc:
        logger.warning("Ingestion failed, serving from DB: %s", exc)
        db.rollback()

    stored = list_combined_from_db(db)
    if stored:
        return stored

    raise HTTPException(status_code=503, detail="No event data available")
