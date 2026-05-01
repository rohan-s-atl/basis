from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas import TrainingDataset
from app.services.training_data_service import (
    export_training_dataset,
    get_confidence_analysis,
    get_dataset_stats,
    get_train_test_split,
    validate_dataset,
)

router = APIRouter(tags=["training-data"])


@router.get("/export-training-data", response_model=TrainingDataset)
def export_training_data(
    limit: int = 10_000,
    db: Session = Depends(get_db),
) -> dict:
    return export_training_dataset(db, limit=min(limit, 50_000))


@router.get("/training-data/split")
def read_train_test_split(
    limit: int = 10_000,
    train_fraction: float = 0.8,
    db: Session = Depends(get_db),
) -> dict:
    return get_train_test_split(
        db,
        train_fraction=train_fraction,
        limit=min(limit, 50_000),
    )


@router.get("/training-data/confidence-buckets")
def read_confidence_buckets(db: Session = Depends(get_db)) -> dict:
    return get_confidence_analysis(db)


@router.get("/training-data/stats")
def read_dataset_stats(
    limit: int = 10_000,
    db: Session = Depends(get_db),
) -> dict:
    return get_dataset_stats(db, limit=min(limit, 50_000))


@router.get("/training-data/validation")
def read_dataset_validation(
    limit: int = 10_000,
    db: Session = Depends(get_db),
) -> dict:
    return validate_dataset(db, limit=min(limit, 50_000))
