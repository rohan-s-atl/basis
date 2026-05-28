from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas import TrainingDataset
from app.services.multi_horizon_service import compute_multi_horizon_outcomes
from app.services.outcome_service import compute_outcomes
from app.services.training_data_service import (
    export_training_dataset,
    get_confidence_analysis,
    get_dataset_stats,
    get_train_test_split,
    validate_dataset,
)
from app.services.model_evaluation_service import get_model_evaluation
from app.services.training_run_service import (
    get_model_health,
    get_training_history,
    save_training_run,
)
from app.services.prediction_pipeline import run_prediction_pipeline
from app.db.models import Event

router = APIRouter(tags=["training-data"])


@router.post("/compute-multi-horizon-outcomes")
def trigger_multi_horizon_outcomes(
    limit: int = 200,
    noise_threshold: float | None = None,
    db: Session = Depends(get_db),
) -> dict:
    return compute_multi_horizon_outcomes(db=db, limit=limit, noise_threshold=noise_threshold)


@router.post("/compute-outcomes")
def trigger_compute_outcomes(
    limit: int = 500,
    noise_threshold: float | None = None,
    force: bool = False,
    db: Session = Depends(get_db),
) -> dict:
    return compute_outcomes(db=db, limit=limit, noise_threshold=noise_threshold, force=force)


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


@router.get("/training-history")
def read_training_history(
    limit: int = 50,
    db: Session = Depends(get_db),
) -> list[dict[str, Any]]:
    return get_training_history(db, limit=min(limit, 200))


@router.get("/model-health")
def read_model_health(db: Session = Depends(get_db)) -> dict[str, Any]:
    return get_model_health(db)


@router.get("/model-evaluation")
def read_model_evaluation(
    limit: int = 50_000,
    model_version: str | None = None,
    current_only: bool = False,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    return get_model_evaluation(
        db,
        limit=min(limit, 50_000),
        model_version=model_version,
        current_only=current_only,
    )


@router.post("/train-model")
def train_model(
    limit: int = 10_000,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    from ml.model_store import invalidate_cache
    from ml.train_model import result_to_dict, train_xgboost_from_payload

    bounded_limit = min(limit, 50_000)
    result = train_xgboost_from_payload(export_training_dataset(db, limit=bounded_limit))
    invalidate_cache()
    payload = result_to_dict(result)

    save_training_run(db, payload, triggered_by="manual")

    return {
        "dataset_size": payload["dataset_size"],
        "label_balance": payload["label_balance"],
        "train_size": payload["train_size"],
        "validation_size": payload["validation_size"],
        "test_size": payload["test_size"],
        "accuracy": payload["accuracy"],
        "roc_auc": payload["roc_auc"],
        "walk_forward_cv": payload["walk_forward_cv"],
        "model_comparison": payload["model_comparison"],
        "calibration": payload["calibration"],
        "decision_threshold": payload["decision_threshold"],
        "segment_models": payload["segment_models"],
        "top_features": payload["top_features"],
        "shap_summary": payload["shap_summary"],
        "confidence_analysis": payload["confidence_analysis"],
    }


@router.post("/refresh-predictions")
def refresh_predictions(
    limit: int = 100,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    events = (
        db.query(Event)
        .order_by(Event.timestamp.desc())
        .limit(min(limit, 500))
        .all()
    )
    created = 0
    skipped = 0
    failed = 0
    for event in events:
        try:
            event_created = len(run_prediction_pipeline(event, db=db))
        except Exception:
            db.rollback()
            failed += 1
            continue
        created += event_created
        if event_created == 0:
            skipped += 1
    return {
        "events_considered": len(events),
        "predictions_created": created,
        "events_skipped_or_already_current": skipped,
        "events_failed": failed,
    }
