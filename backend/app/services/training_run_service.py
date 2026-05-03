from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import Outcome, Prediction, TrainingRun

logger = logging.getLogger(__name__)

AUTO_RETRAIN_MIN_GROWTH = 25
_DRIFT_WINDOW = 30
_DRIFT_THRESHOLD = 0.10  # 10 percentage-point drop from peak triggers drift alert


def save_training_run(
    db: Session,
    payload: dict[str, Any],
    *,
    triggered_by: str = "manual",
) -> TrainingRun:
    calibration = payload.get("calibration") or {}
    wfcv = payload.get("walk_forward_cv") or {}
    comparison = payload.get("model_comparison") or {}

    run = TrainingRun(
        triggered_by=triggered_by,
        dataset_size=int(payload["dataset_size"]),
        train_size=int(payload["train_size"]),
        test_size=int(payload["test_size"]),
        accuracy=float(payload["accuracy"]),
        roc_auc=float(payload["roc_auc"]) if payload.get("roc_auc") is not None else None,
        brier_score_calibrated=float(calibration.get("brier_score_calibrated", 0.0)),
        brier_improvement=float(calibration.get("improvement", 0.0)),
        walk_forward_mean=float(wfcv.get("mean_accuracy", 0.0)),
        walk_forward_std=float(wfcv.get("std_accuracy", 0.0)),
        walk_forward_folds=int(wfcv.get("n_folds", 0)),
        xgboost_roc_auc=comparison.get("xgboost_roc_auc"),
        logistic_roc_auc=comparison.get("logistic_roc_auc"),
        winner_model=str(comparison.get("winner", "unknown")),
        top_features=json.dumps(payload.get("top_features", [])),
        label_balance=json.dumps(payload.get("label_balance", {})),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    logger.info(
        "Training run saved: size=%d acc=%.4f roc_auc=%s triggered_by=%s",
        run.dataset_size,
        run.accuracy,
        run.roc_auc,
        triggered_by,
    )
    return run


def get_training_history(db: Session, *, limit: int = 50) -> list[dict[str, Any]]:
    runs = (
        db.query(TrainingRun)
        .order_by(TrainingRun.trained_at.desc())
        .limit(limit)
        .all()
    )
    return [_run_to_dict(r) for r in runs]


def get_latest_run(db: Session) -> TrainingRun | None:
    return (
        db.query(TrainingRun)
        .order_by(TrainingRun.trained_at.desc())
        .first()
    )


def should_auto_retrain(db: Session, current_dataset_size: int) -> bool:
    latest = get_latest_run(db)
    last_size = latest.dataset_size if latest is not None else 0
    growth = current_dataset_size - last_size
    if growth >= AUTO_RETRAIN_MIN_GROWTH:
        logger.info(
            "Auto-retrain triggered: dataset grew %d → %d (+%d)",
            last_size,
            current_dataset_size,
            growth,
        )
        return True
    return False


def get_model_health(db: Session) -> dict[str, Any]:
    latest = get_latest_run(db)
    rolling = _rolling_accuracy(db, window=_DRIFT_WINDOW)

    if latest is None:
        return {
            "status": "no_model",
            "rolling_accuracy": rolling,
            "peak_accuracy": None,
            "drift_detected": False,
            "last_trained_at": None,
            "dataset_size_at_training": None,
            "retraining_threshold": AUTO_RETRAIN_MIN_GROWTH,
        }

    peak = float(latest.accuracy)
    drift = (
        rolling is not None
        and rolling["samples"] >= 10
        and (peak - rolling["accuracy"]) > _DRIFT_THRESHOLD
    )

    return {
        "status": "drift_detected" if drift else "healthy",
        "rolling_accuracy": rolling,
        "peak_accuracy": peak,
        "drift_detected": drift,
        "last_trained_at": latest.trained_at.isoformat(),
        "dataset_size_at_training": latest.dataset_size,
        "retraining_threshold": AUTO_RETRAIN_MIN_GROWTH,
    }


def _rolling_accuracy(db: Session, *, window: int) -> dict[str, Any] | None:
    labels = [
        int(label)
        for (label,) in (
            db.query(Outcome.filtered_label)
            .join(Prediction, Prediction.id == Outcome.prediction_id)
            .filter(Outcome.filtered_label.isnot(None))
            .order_by(Prediction.timestamp.desc())
            .limit(window)
            .all()
        )
    ]
    if not labels:
        return None
    return {
        "accuracy": round(sum(labels) / len(labels), 4),
        "samples": len(labels),
        "window": window,
    }


def _run_to_dict(run: TrainingRun) -> dict[str, Any]:
    return {
        "id": str(run.id),
        "trained_at": run.trained_at.isoformat(),
        "triggered_by": run.triggered_by,
        "dataset_size": run.dataset_size,
        "train_size": run.train_size,
        "test_size": run.test_size,
        "accuracy": run.accuracy,
        "roc_auc": run.roc_auc,
        "brier_score_calibrated": run.brier_score_calibrated,
        "brier_improvement": run.brier_improvement,
        "walk_forward_mean": run.walk_forward_mean,
        "walk_forward_std": run.walk_forward_std,
        "walk_forward_folds": run.walk_forward_folds,
        "xgboost_roc_auc": run.xgboost_roc_auc,
        "logistic_roc_auc": run.logistic_roc_auc,
        "winner_model": run.winner_model,
        "top_features": json.loads(run.top_features),
        "label_balance": json.loads(run.label_balance),
    }
