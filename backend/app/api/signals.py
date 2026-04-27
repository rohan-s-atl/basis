from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.event_repository import get_signal_stats

router = APIRouter(tags=["signals"])


@router.get("/signals/accuracy")
def get_signal_accuracy(db: Session = Depends(get_db)) -> dict:
    return get_signal_stats(db)
