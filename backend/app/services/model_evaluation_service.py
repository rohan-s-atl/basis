from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import Event, FeatureSnapshot, MultiHorizonOutcome, Outcome, Prediction
from app.services.training_data_service import validate_dataset


@dataclass(frozen=True)
class EvaluationRow:
    prediction_id: str
    timestamp: datetime
    asset: str
    event_type: str
    horizon_days: int
    model_version: str
    confidence: float
    predicted_direction: str
    event_sentiment: float
    spy_trend: float
    sector_trend: float
    raw_return: float
    excess_return: float | None
    return_bucket: str
    target_label: int
    model_correct: int
    benchmark_label: int | None


def get_model_evaluation(db: Session, *, limit: int = 50_000) -> dict[str, Any]:
    rows = _evaluation_rows(db, limit=limit)
    dataset_validation = validate_dataset(db, limit=limit)
    return {
        "sample_count": len(rows),
        "overall": _performance(rows),
        "high_confidence": _performance([row for row in rows if row.confidence >= 0.7]),
        "benchmark_relative": _benchmark_relative_performance(rows),
        "baselines": _baseline_comparison(rows),
        "raw_return_direction": _raw_return_direction(rows),
        "by_asset": _group_performance(rows, "asset", limit=10),
        "by_event_type": _group_performance(rows, "event_type"),
        "by_horizon": _group_performance(rows, "horizon_days"),
        "by_return_bucket": _group_performance(rows, "return_bucket"),
        "by_model_version": _group_performance(rows, "model_version"),
        "calibration": _calibration(rows),
        "calibration_by_horizon": _calibration_by_group(rows, "horizon_days"),
        "calibration_by_event_type": _calibration_by_group(rows, "event_type"),
        "data_quality": _data_quality(db, dataset_validation),
        "recommendations": _recommendations(rows, dataset_validation),
    }


def _evaluation_rows(db: Session, *, limit: int) -> list[EvaluationRow]:
    multi_prediction_ids = _multi_labeled_prediction_ids(db)
    multi_rows = [
        _row_from_multi(prediction, event, snapshot, outcome)
        for prediction, event, snapshot, outcome in (
            db.query(Prediction, Event, FeatureSnapshot, MultiHorizonOutcome)
            .join(Event, Event.id == Prediction.event_id)
            .join(FeatureSnapshot, FeatureSnapshot.prediction_id == Prediction.id)
            .join(MultiHorizonOutcome, MultiHorizonOutcome.prediction_id == Prediction.id)
            .filter(MultiHorizonOutcome.benchmark_label.isnot(None) | MultiHorizonOutcome.label.isnot(None))
            .order_by(Prediction.timestamp.asc())
            .limit(limit)
            .all()
        )
    ]
    single_rows = [
        _row_from_single(prediction, event, snapshot, outcome)
        for prediction, event, snapshot, outcome in (
            db.query(Prediction, Event, FeatureSnapshot, Outcome)
            .join(Event, Event.id == Prediction.event_id)
            .join(FeatureSnapshot, FeatureSnapshot.prediction_id == Prediction.id)
            .join(Outcome, Outcome.prediction_id == Prediction.id)
            .filter(
                Outcome.benchmark_label.isnot(None) | Outcome.label.isnot(None),
                ~Prediction.id.in_(multi_prediction_ids),
            )
            .order_by(Prediction.timestamp.asc())
            .limit(limit)
            .all()
        )
    ]
    rows = [*multi_rows, *single_rows]
    rows.sort(key=lambda row: row.timestamp)
    return rows[:limit]


def _row_from_multi(
    prediction: Prediction,
    event: Event,
    snapshot: FeatureSnapshot,
    outcome: MultiHorizonOutcome,
) -> EvaluationRow:
    return EvaluationRow(
        prediction_id=str(prediction.id),
        timestamp=prediction.timestamp,
        asset=prediction.asset,
        event_type=event.event_type,
        horizon_days=int(outcome.horizon_days),
        model_version=prediction.model_version,
        confidence=float(prediction.confidence),
        predicted_direction=prediction.predicted_direction,
        event_sentiment=float(event.sentiment),
        spy_trend=float((snapshot.derived_features or {}).get("spy_trend", 0.0)),
        sector_trend=float((snapshot.market_features or {}).get("sector_return_5d", 0.0)),
        raw_return=float(outcome.raw_return),
        excess_return=outcome.excess_return,
        return_bucket=outcome.return_bucket,
        target_label=_target_label(outcome.benchmark_label, outcome.label),
        model_correct=_direction_matches_label(prediction.predicted_direction, _target_label(outcome.benchmark_label, outcome.label)),
        benchmark_label=outcome.benchmark_label,
    )


def _row_from_single(
    prediction: Prediction,
    event: Event,
    snapshot: FeatureSnapshot,
    outcome: Outcome,
) -> EvaluationRow:
    return EvaluationRow(
        prediction_id=str(prediction.id),
        timestamp=prediction.timestamp,
        asset=prediction.asset,
        event_type=event.event_type,
        horizon_days=_horizon_days(prediction.horizon),
        model_version=prediction.model_version,
        confidence=float(prediction.confidence),
        predicted_direction=prediction.predicted_direction,
        event_sentiment=float(event.sentiment),
        spy_trend=float((snapshot.derived_features or {}).get("spy_trend", 0.0)),
        sector_trend=float((snapshot.market_features or {}).get("sector_return_5d", 0.0)),
        raw_return=float(outcome.raw_return),
        excess_return=outcome.excess_return,
        return_bucket=outcome.return_bucket,
        target_label=_target_label(outcome.benchmark_label, outcome.label),
        model_correct=_direction_matches_label(prediction.predicted_direction, _target_label(outcome.benchmark_label, outcome.label)),
        benchmark_label=outcome.benchmark_label,
    )


def _performance(rows: list[EvaluationRow]) -> dict[str, float | int]:
    if not rows:
        return {"samples": 0, "accuracy": 0.0, "avg_return": 0.0, "avg_excess_return": 0.0}
    excess_rows = [row.excess_return for row in rows if row.excess_return is not None]
    return {
        "samples": len(rows),
        "accuracy": round(sum(row.model_correct for row in rows) / len(rows), 4),
        "avg_return": round(sum(row.raw_return for row in rows) / len(rows), 6),
        "avg_excess_return": round(sum(float(value) for value in excess_rows) / len(excess_rows), 6) if excess_rows else 0.0,
    }


def _benchmark_relative_performance(rows: list[EvaluationRow]) -> dict[str, float | int]:
    benchmark_rows = [row for row in rows if row.benchmark_label is not None]
    if not benchmark_rows:
        return {"samples": 0, "accuracy": 0.0, "avg_excess_return": 0.0}
    return {
        "samples": len(benchmark_rows),
        "accuracy": round(sum(row.model_correct for row in benchmark_rows) / len(benchmark_rows), 4),
        "avg_excess_return": round(
            sum(float(row.excess_return or 0.0) for row in benchmark_rows) / len(benchmark_rows),
            6,
        ),
    }


def _baseline_comparison(rows: list[EvaluationRow]) -> dict[str, dict[str, float | int]]:
    return {
        "model": _performance(rows),
        "always_up": _baseline_performance(rows, lambda row: "up"),
        "always_down": _baseline_performance(rows, lambda row: "down"),
        "event_sentiment": _baseline_performance(
            rows,
            lambda row: "up" if row.event_sentiment >= 0 else "down",
        ),
        "spy_trend": _baseline_performance(
            rows,
            lambda row: "up" if row.spy_trend >= 0 else "down",
        ),
        "sector_trend": _baseline_performance(
            rows,
            lambda row: "up" if row.sector_trend >= 0 else "down",
        ),
    }


def _raw_return_direction(rows: list[EvaluationRow]) -> dict[str, float | int]:
    if not rows:
        return {
            "samples": 0,
            "up": 0.0,
            "down": 0.0,
            "flat": 0.0,
            "up_count": 0,
            "down_count": 0,
            "flat_count": 0,
        }

    up = sum(1 for row in rows if row.raw_return > 0)
    down = sum(1 for row in rows if row.raw_return < 0)
    flat = len(rows) - up - down
    total = len(rows)
    return {
        "samples": total,
        "up": round(up / total, 4),
        "down": round(down / total, 4),
        "flat": round(flat / total, 4),
        "up_count": up,
        "down_count": down,
        "flat_count": flat,
    }


def _baseline_performance(rows: list[EvaluationRow], direction_for_row) -> dict[str, float | int]:
    matches = [
        _direction_matches_label(direction_for_row(row), row.target_label)
        for row in rows
    ]
    if not matches:
        return {"samples": 0, "accuracy": 0.0}
    return {
        "samples": len(matches),
        "accuracy": round(sum(matches) / len(matches), 4),
    }


def _group_performance(
    rows: list[EvaluationRow],
    attr: str,
    *,
    limit: int | None = None,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[EvaluationRow]] = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, attr))].append(row)

    ordered = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
    if limit is not None:
        ordered = ordered[:limit]
    return {key: _performance(items) for key, items in ordered}


def _calibration(rows: list[EvaluationRow]) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[EvaluationRow]] = {
        "0.5-0.6": [],
        "0.6-0.7": [],
        "0.7-0.8": [],
        "0.8+": [],
    }
    for row in rows:
        buckets[_confidence_bucket(row.confidence)].append(row)
    return {name: _calibration_bucket(items) for name, items in buckets.items()}


def _calibration_by_group(rows: list[EvaluationRow], attr: str) -> dict[str, dict[str, dict[str, float | int]]]:
    groups: dict[str, list[EvaluationRow]] = defaultdict(list)
    for row in rows:
        groups[str(getattr(row, attr))].append(row)
    return {
        key: _calibration(items)
        for key, items in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)[:10]
    }


def _calibration_bucket(rows: list[EvaluationRow]) -> dict[str, float | int]:
    if not rows:
        return {"samples": 0, "observed_accuracy": 0.0, "avg_confidence": 0.0, "avg_excess_return": 0.0}
    excess_rows = [row.excess_return for row in rows if row.excess_return is not None]
    return {
        "samples": len(rows),
        "observed_accuracy": round(sum(row.model_correct for row in rows) / len(rows), 4),
        "avg_confidence": round(sum(row.confidence for row in rows) / len(rows), 4),
        "avg_excess_return": round(sum(float(value) for value in excess_rows) / len(excess_rows), 6) if excess_rows else 0.0,
    }


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.6:
        return "0.5-0.6"
    if confidence < 0.7:
        return "0.6-0.7"
    if confidence < 0.8:
        return "0.7-0.8"
    return "0.8+"


def _data_quality(db: Session, validation: dict[str, Any]) -> dict[str, Any]:
    duplicate_events = (
        db.query(Event.raw_text, Event.source)
        .group_by(Event.raw_text, Event.source)
        .having(func.count(Event.id) > 1)
        .count()
    )
    predictions_missing_snapshots = (
        db.query(Prediction)
        .outerjoin(FeatureSnapshot, FeatureSnapshot.prediction_id == Prediction.id)
        .filter(FeatureSnapshot.id.is_(None))
        .count()
    )
    unlabeled_predictions = (
        db.query(Prediction)
        .outerjoin(Outcome, Outcome.prediction_id == Prediction.id)
        .outerjoin(MultiHorizonOutcome, MultiHorizonOutcome.prediction_id == Prediction.id)
        .filter(Outcome.id.is_(None), MultiHorizonOutcome.id.is_(None))
        .count()
    )
    return {
        "dataset_samples": validation["num_samples"],
        "feature_count": validation["num_features"],
        "issues": validation["issues"],
        "duplicate_event_groups": duplicate_events,
        "predictions_missing_snapshots": predictions_missing_snapshots,
        "unlabeled_predictions": unlabeled_predictions,
    }


def _recommendations(rows: list[EvaluationRow], validation: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    overall = _performance(rows)
    high_confidence = _performance([row for row in rows if row.confidence >= 0.7])
    baselines = _baseline_comparison(rows)

    model_accuracy = float(overall["accuracy"])
    best_baseline = max(
        float(payload["accuracy"])
        for name, payload in baselines.items()
        if name != "model"
    ) if rows else 0.0

    if rows and model_accuracy <= best_baseline:
        recommendations.append("Model is not beating simple baselines yet; keep signals gated.")
    if high_confidence["samples"] and high_confidence["accuracy"] <= overall["accuracy"]:
        recommendations.append("High-confidence predictions are not outperforming the full set.")
    if validation["issues"]:
        recommendations.append("Resolve dataset validation issues before trusting retrains.")
    if len(rows) < 1000:
        recommendations.append("Keep accumulating labeled live outcomes; sample count is still thin.")
    return recommendations


def _multi_labeled_prediction_ids(db: Session) -> list[Any]:
    return [
        prediction_id
        for (prediction_id,) in (
            db.query(MultiHorizonOutcome.prediction_id)
            .filter(MultiHorizonOutcome.benchmark_label.isnot(None) | MultiHorizonOutcome.label.isnot(None))
            .distinct()
            .all()
        )
    ]


def _target_label(benchmark_label: int | None, raw_label: int | None) -> int:
    label = benchmark_label if benchmark_label is not None else raw_label
    if label is None:
        raise ValueError("evaluation row is missing both benchmark and raw labels")
    return int(label)


def _direction_matches_label(direction: str, label: int) -> int:
    return int((direction == "up" and label == 1) or (direction == "down" and label == 0))


def _horizon_days(value: str | None) -> int:
    if not value:
        return 1
    try:
        return int(value.lower().replace("d", "").split("-")[0])
    except ValueError:
        return 1
