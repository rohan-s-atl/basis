from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas import PredictionSummary
from app.services.prediction_service import generate_predictions

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=PredictionSummary)
def read_predictions(db: Session = Depends(get_db), limit: int = 50) -> dict:
    return generate_predictions(db, limit=limit)
