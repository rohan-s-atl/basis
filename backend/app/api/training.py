from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas import TrainingExample
from app.services.training_data_service import export_training_dataset

router = APIRouter(tags=["training-data"])


@router.get("/export-training-data", response_model=list[TrainingExample])
def export_training_data(
    limit: int = 10_000,
    db: Session = Depends(get_db),
) -> list[dict]:
    return export_training_dataset(db, limit=min(limit, 50_000))
