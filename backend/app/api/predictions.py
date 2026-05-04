from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas import PredictionSummary
from app.services.prediction_service import generate_predictions

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("", response_model=PredictionSummary)
def read_predictions(
    db: Session = Depends(get_db),
    limit: int = 50,
    min_quality: float = 0.62,
    include_weak: bool = False,
) -> dict:
    return generate_predictions(
        db,
        limit=limit,
        min_quality=min_quality,
        include_weak=include_weak,
    )
