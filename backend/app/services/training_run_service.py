from __future__ import annotations

import json
import logging
import math
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import MultiHorizonOutcome, Outcome, Prediction, TrainingRun

logger = logging.getLogger(__name__)

AUTO_RETRAIN_MIN_GROWTH = 25
_DRIFT_WINDOW = 30
_DRIFT_THRESHOLD = 0.10  # 10 percentage-point drop from peak triggers drift alert
_CONFIDENCE_DRIFT_WINDOW = 50
_CONFIDENCE_DRIFT_THRESHOLD = 0.20
_CONFIDENCE_BUCKETS = (0.6, 0.7, 0.8)


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
        accuracy=_deployment_accuracy(payload),
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
    confidence_drift = _confidence_distribution_drift(db, latest)

    if latest is None:
        return {
            "status": "no_model",
            "rolling_accuracy": rolling,
            "peak_accuracy": None,
            "trained_accuracy": None,
            "deployment_accuracy": None,
            "accuracy_metric": "threshold_optimized_calibrated_accuracy",
            "trained_roc_auc": None,
            "drift_detected": False,
            "confidence_drift": confidence_drift,
            "last_trained_at": None,
            "dataset_size_at_training": None,
            "retraining_threshold": AUTO_RETRAIN_MIN_GROWTH,
        }

    peak = float(latest.accuracy)
    accuracy_drift = (
        rolling is not None
        and rolling["samples"] >= 10
        and (peak - rolling["accuracy"]) > _DRIFT_THRESHOLD
    )
    drift = accuracy_drift or bool(confidence_drift.get("drift_detected"))

    return {
        "status": "drift_detected" if drift else "healthy",
        "rolling_accuracy": rolling,
        "peak_accuracy": peak,
        "trained_accuracy": peak,
        "deployment_accuracy": peak,
        "accuracy_metric": "threshold_optimized_calibrated_accuracy",
        "trained_roc_auc": latest.roc_auc,
        "drift_detected": drift,
        "confidence_drift": confidence_drift,
        "last_trained_at": latest.trained_at.isoformat(),
        "dataset_size_at_training": latest.dataset_size,
        "retraining_threshold": AUTO_RETRAIN_MIN_GROWTH,
    }


def _rolling_accuracy(db: Session, *, window: int) -> dict[str, Any] | None:
    multi_rows = [
        (timestamp, _direction_matches_label(direction, _target_label(benchmark_label, label)))
        for timestamp, direction, benchmark_label, label in (
            db.query(
                Prediction.timestamp,
                Prediction.predicted_direction,
                MultiHorizonOutcome.benchmark_label,
                MultiHorizonOutcome.label,
            )
            .join(MultiHorizonOutcome, MultiHorizonOutcome.prediction_id == Prediction.id)
            .filter(MultiHorizonOutcome.benchmark_label.isnot(None) | MultiHorizonOutcome.label.isnot(None))
            .all()
        )
    ]
    multi_prediction_ids = [
        prediction_id
        for (prediction_id,) in (
            db.query(MultiHorizonOutcome.prediction_id)
            .filter(MultiHorizonOutcome.benchmark_label.isnot(None) | MultiHorizonOutcome.label.isnot(None))
            .distinct()
            .all()
        )
    ]
    single_rows = [
        (timestamp, _direction_matches_label(direction, _target_label(benchmark_label, label)))
        for timestamp, direction, benchmark_label, label in (
            db.query(Prediction.timestamp, Prediction.predicted_direction, Outcome.benchmark_label, Outcome.label)
            .join(Outcome, Outcome.prediction_id == Prediction.id)
            .filter(
                Outcome.benchmark_label.isnot(None) | Outcome.label.isnot(None),
                ~Prediction.id.in_(multi_prediction_ids),
            )
            .all()
        )
    ]
    labels = [
        label
        for _, label in sorted(
            [*multi_rows, *single_rows],
            key=lambda row: row[0],
            reverse=True,
        )[:window]
    ]
    if not labels:
        return None
    return {
        "accuracy": round(sum(labels) / len(labels), 4),
        "samples": len(labels),
        "window": window,
    }


def _confidence_distribution_drift(db: Session, latest: TrainingRun | None) -> dict[str, Any]:
    if latest is None:
        return {
            "drift_detected": False,
            "psi": 0.0,
            "threshold": _CONFIDENCE_DRIFT_THRESHOLD,
            "training": [],
            "recent": [],
            "samples": 0,
        }

    training_confidences = [
        float(confidence)
        for (confidence,) in (
            db.query(Prediction.confidence)
            .join(Outcome, Outcome.prediction_id == Prediction.id)
            .filter(
                Outcome.filtered_label.isnot(None),
                Prediction.timestamp <= latest.trained_at,
            )
            .all()
        )
    ]
    recent_confidences = [
        float(confidence)
        for (confidence,) in (
            db.query(Prediction.confidence)
            .order_by(Prediction.timestamp.desc())
            .limit(_CONFIDENCE_DRIFT_WINDOW)
            .all()
        )
    ]

    training_dist = _confidence_distribution(training_confidences)
    recent_dist = _confidence_distribution(recent_confidences)
    psi = _population_stability_index(training_dist, recent_dist)
    enough_samples = len(training_confidences) >= 20 and len(recent_confidences) >= 20

    return {
        "drift_detected": bool(enough_samples and psi >= _CONFIDENCE_DRIFT_THRESHOLD),
        "psi": round(psi, 4),
        "threshold": _CONFIDENCE_DRIFT_THRESHOLD,
        "training": training_dist,
        "recent": recent_dist,
        "samples": len(recent_confidences),
    }


def _confidence_distribution(confidences: list[float]) -> list[float]:
    if not confidences:
        return [0.0, 0.0, 0.0, 0.0]

    buckets = [0, 0, 0, 0]
    for confidence in confidences:
        if confidence < _CONFIDENCE_BUCKETS[0]:
            buckets[0] += 1
        elif confidence < _CONFIDENCE_BUCKETS[1]:
            buckets[1] += 1
        elif confidence < _CONFIDENCE_BUCKETS[2]:
            buckets[2] += 1
        else:
            buckets[3] += 1

    total = len(confidences)
    return [round(count / total, 4) for count in buckets]


def _population_stability_index(expected: list[float], actual: list[float]) -> float:
    epsilon = 0.0001
    psi = 0.0
    for expected_share, actual_share in zip(expected, actual):
        e = max(expected_share, epsilon)
        a = max(actual_share, epsilon)
        psi += (a - e) * math.log(a / e)
    return psi


def _run_to_dict(run: TrainingRun) -> dict[str, Any]:
    return {
        "id": str(run.id),
        "trained_at": run.trained_at.isoformat(),
        "triggered_by": run.triggered_by,
        "dataset_size": run.dataset_size,
        "train_size": run.train_size,
        "test_size": run.test_size,
        "accuracy": run.accuracy,
        "calibrated_accuracy": run.accuracy,
        "deployment_accuracy": run.accuracy,
        "accuracy_metric": "threshold_optimized_calibrated_accuracy",
        "roc_auc": run.roc_auc,
        "brier_score_calibrated": run.brier_score_calibrated,
        "brier_improvement": run.brier_improvement,
        "walk_forward_mean": run.walk_forward_mean,
        "walk_forward_std": run.walk_forward_std,
        "walk_forward_folds": run.walk_forward_folds,
        "xgboost_roc_auc": run.xgboost_roc_auc,
        "logistic_roc_auc": run.logistic_roc_auc,
        "winner_model": run.winner_model,
        "comparison_winner": run.winner_model,
        "top_features": json.loads(run.top_features),
        "label_balance": json.loads(run.label_balance),
    }


def _deployment_accuracy(payload: dict[str, Any]) -> float:
    value = payload.get("calibrated_accuracy")
    if value is None:
        value = payload.get("deployment_accuracy")
    if value is None:
        value = payload["accuracy"]
    return float(value)


def _target_label(benchmark_label: int | None, raw_label: int | None) -> int:
    label = benchmark_label if benchmark_label is not None else raw_label
    if label is None:
        raise ValueError("health row is missing both benchmark and raw labels")
    return int(label)


def _direction_matches_label(direction: str, label: int) -> int:
    return int((direction == "up" and label == 1) or (direction == "down" and label == 0))
