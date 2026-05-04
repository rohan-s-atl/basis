import math
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import FeatureSnapshot, MultiHorizonOutcome, Outcome, Prediction
from app.services.feature_service import flatten_feature_snapshot

logger = logging.getLogger(__name__)
MIN_RECOMMENDED_DATASET_SIZE = 300


def export_training_dataset(db: Session, *, limit: int = 10_000) -> dict[str, Any]:
    rows = _labeled_training_rows(db, limit=limit, chronological=True)
    return _dataset_payload(rows)


def get_train_test_split(
    db: Session,
    *,
    train_fraction: float = 0.8,
    limit: int = 10_000,
) -> dict[str, dict[str, Any]]:
    rows = _labeled_training_rows(db, limit=limit, chronological=True)
    split_index = int(len(rows) * max(0.0, min(train_fraction, 1.0)))
    train = rows[:split_index]
    test = rows[split_index:]
    return {
        "train": _dataset_payload(train),
        "test": _dataset_payload(test),
    }


def get_confidence_analysis(db: Session) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[int]] = {
        "0.5-0.6": [],
        "0.6-0.7": [],
        "0.7-0.8": [],
        "0.8+": [],
    }
    rows = _confidence_label_rows(db)

    for confidence, label in rows:
        buckets[_confidence_bucket(float(confidence))].append(int(label))

    return {
        bucket: {
            "accuracy": round(sum(labels) / len(labels), 4) if labels else 0.0,
            "samples": len(labels),
        }
        for bucket, labels in buckets.items()
    }


def get_confidence_bucket_metrics(db: Session) -> dict[str, dict[str, float | int]]:
    return get_confidence_analysis(db)


def get_dataset_stats(db: Session, *, limit: int = 10_000) -> dict[str, Any]:
    dataset = export_training_dataset(db, limit=limit)
    num_samples = int(dataset["metadata"]["num_samples"])
    if num_samples < MIN_RECOMMENDED_DATASET_SIZE:
        logger.warning(
            "Training dataset has only %d samples; recommended minimum is %d",
            num_samples,
            MIN_RECOMMENDED_DATASET_SIZE,
        )
    return {
        "num_samples": num_samples,
        "class_distribution": dataset["metadata"]["class_balance"],
        "feature_count": dataset["metadata"]["feature_count"],
    }


def validate_dataset(db: Session, *, limit: int = 10_000) -> dict[str, Any]:
    dataset = export_training_dataset(db, limit=limit)
    issues: list[str] = []
    labels = [int(label) for label in dataset["labels"]]
    feature_names = list(dataset["feature_names"])
    feature_rows = list(dataset["features"])

    if not labels:
        issues.append("dataset is empty")

    missing_features = _missing_feature_issues(feature_rows, feature_names)
    issues.extend(missing_features)

    class_balance = _class_balance(labels)
    positive_ratio = float(class_balance.get("positive", 0.0))
    if labels and (positive_ratio < 0.3 or positive_ratio > 0.7):
        issues.append("label balance outside 30-70% range")

    constant_columns = _constant_feature_columns(feature_rows, feature_names)
    if constant_columns:
        issues.append(f"constant feature columns: {', '.join(constant_columns[:10])}")

    return {
        "num_samples": len(labels),
        "class_balance": class_balance,
        "num_features": len(feature_names),
        "issues": issues,
    }


def _labeled_training_rows(
    db: Session,
    *,
    limit: int,
    chronological: bool,
) -> list[dict[str, Any]]:
    rows = [
        *_multi_horizon_rows(db, limit=limit, chronological=chronological),
        *_single_horizon_rows(db, limit=limit, chronological=chronological),
    ]
    if chronological:
        rows.sort(key=lambda row: row["timestamp"])
    else:
        rows.sort(key=lambda row: row["computed_at"], reverse=True)
    return rows[:limit]


def _single_horizon_rows(
    db: Session,
    *,
    limit: int,
    chronological: bool,
) -> list[dict[str, Any]]:
    multi_prediction_ids = _multi_labeled_prediction_ids(db)
    order_column = Prediction.timestamp.asc() if chronological else Outcome.computed_at.desc()
    rows = (
        db.query(FeatureSnapshot, Outcome)
        .join(Prediction, Prediction.id == FeatureSnapshot.prediction_id)
        .join(Outcome, Outcome.prediction_id == Prediction.id)
        .filter(
            Outcome.filtered_label.isnot(None),
            ~Prediction.id.in_(multi_prediction_ids),
        )
        .order_by(order_column)
        .limit(limit)
        .all()
    )
    dataset: list[dict[str, Any]] = []
    for snapshot, outcome in rows:
        derived = {
            **snapshot.derived_features,
            "horizon_days": _horizon_days_from_label(snapshot.prediction.horizon),
        }
        flattened = flatten_feature_snapshot(
            snapshot.event_features,
            snapshot.market_features,
            derived,
        )
        dataset.append({
            "features": flattened,
            "label": int(outcome.filtered_label),
            "timestamp": snapshot.prediction.timestamp,
            "computed_at": outcome.computed_at,
        })
    return dataset


def _multi_horizon_rows(
    db: Session,
    *,
    limit: int,
    chronological: bool,
) -> list[dict[str, Any]]:
    order_column = (
        Prediction.timestamp.asc() if chronological else MultiHorizonOutcome.computed_at.desc()
    )
    rows = (
        db.query(FeatureSnapshot, MultiHorizonOutcome)
        .join(Prediction, Prediction.id == FeatureSnapshot.prediction_id)
        .join(MultiHorizonOutcome, MultiHorizonOutcome.prediction_id == Prediction.id)
        .filter(MultiHorizonOutcome.label.isnot(None))
        .order_by(order_column)
        .limit(limit)
        .all()
    )
    dataset: list[dict[str, Any]] = []
    for snapshot, outcome in rows:
        derived = {**snapshot.derived_features, "horizon_days": outcome.horizon_days}
        flattened = flatten_feature_snapshot(
            snapshot.event_features,
            snapshot.market_features,
            derived,
        )
        dataset.append({
            "features": flattened,
            "label": int(outcome.label),
            "timestamp": snapshot.prediction.timestamp,
            "computed_at": outcome.computed_at,
        })
    return dataset


def _confidence_label_rows(db: Session) -> list[tuple[float, int]]:
    multi_rows = [
        (float(confidence), int(label))
        for confidence, label in (
            db.query(Prediction.confidence, MultiHorizonOutcome.label)
            .join(MultiHorizonOutcome, MultiHorizonOutcome.prediction_id == Prediction.id)
            .filter(MultiHorizonOutcome.label.isnot(None))
            .all()
        )
    ]
    multi_prediction_ids = _multi_labeled_prediction_ids(db)
    single_rows = [
        (float(confidence), int(label))
        for confidence, label in (
            db.query(Prediction.confidence, Outcome.filtered_label)
            .join(Outcome, Outcome.prediction_id == Prediction.id)
            .filter(
                Outcome.filtered_label.isnot(None),
                ~Prediction.id.in_(multi_prediction_ids),
            )
            .all()
        )
    ]
    return [*multi_rows, *single_rows]


def _multi_labeled_prediction_ids(db: Session) -> list[Any]:
    return [
        prediction_id
        for (prediction_id,) in (
            db.query(MultiHorizonOutcome.prediction_id)
            .filter(MultiHorizonOutcome.label.isnot(None))
            .distinct()
            .all()
        )
    ]


def _horizon_days_from_label(value: str | None) -> int:
    if not value:
        return 1
    normalized = value.strip().lower()
    if normalized.endswith("d"):
        normalized = normalized[:-1]
    try:
        return max(1, int(normalized))
    except ValueError:
        return 1


def _dataset_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    feature_names = _feature_names(rows)
    labels = [int(row["label"]) for row in rows]
    feature_matrix = [
        [_safe_numeric(row["features"].get(feature_name, 0.0)) for feature_name in feature_names]
        for row in rows
    ]
    return {
        "features": feature_matrix,
        "labels": labels,
        "feature_names": feature_names,
        "metadata": {
            "num_samples": len(rows),
            "class_balance": _class_balance(labels),
            "feature_count": len(feature_names),
        },
    }


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({key for row in rows for key in row["features"]})


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.6:
        return "0.5-0.6"
    if confidence < 0.7:
        return "0.6-0.7"
    if confidence < 0.8:
        return "0.7-0.8"
    return "0.8+"


def _class_balance(labels: list[int]) -> dict[str, float | int]:
    total = len(labels)
    positives = sum(1 for label in labels if label == 1)
    negatives = total - positives
    return {
        "positive": round(positives / total, 4) if total else 0.0,
        "negative": round(negatives / total, 4) if total else 0.0,
        "positive_count": positives,
        "negative_count": negatives,
    }


def _missing_feature_issues(feature_rows: list[list[float]], feature_names: list[str]) -> list[str]:
    issues: list[str] = []
    for row_index, row in enumerate(feature_rows):
        if len(row) != len(feature_names):
            return [f"row {row_index} feature count does not match feature_names"]
        for feature_index, feature_name in enumerate(feature_names):
            value = row[feature_index]
            if value is None or _is_invalid_number(value):
                issues.append(f"missing feature value at row {row_index}: {feature_name}")
                return issues
    return issues


def _constant_feature_columns(feature_rows: list[list[float]], feature_names: list[str]) -> list[str]:
    constant_columns: list[str] = []
    for feature_index, feature_name in enumerate(feature_names):
        values = {row[feature_index] for row in feature_rows}
        if len(values) <= 1 and feature_rows:
            constant_columns.append(feature_name)
    return constant_columns


def _is_invalid_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return True
    return math.isnan(float(value)) or math.isinf(float(value))


def _safe_numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)) and not _is_invalid_number(value):
        return float(value)
    return 0.0
