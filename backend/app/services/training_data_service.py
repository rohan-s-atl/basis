import math
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import FeatureSnapshot, Outcome, Prediction
from app.services.feature_service import flatten_feature_snapshot


def export_training_dataset(db: Session, *, limit: int = 10_000) -> list[dict[str, Any]]:
    return [
        {"features": row["features"], "label": row["label"]}
        for row in _labeled_training_rows(db, limit=limit, chronological=False)
    ]


def get_train_test_split(
    db: Session,
    *,
    train_fraction: float = 0.8,
    limit: int = 10_000,
) -> dict[str, list[dict[str, Any]]]:
    rows = _labeled_training_rows(db, limit=limit, chronological=True)
    split_index = int(len(rows) * max(0.0, min(train_fraction, 1.0)))
    train = rows[:split_index]
    test = rows[split_index:]
    return {
        "train": [_strip_timestamp(row) for row in train],
        "test": [_strip_timestamp(row) for row in test],
    }


def get_confidence_bucket_metrics(db: Session) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[int]] = {
        "0.5-0.6": [],
        "0.6-0.7": [],
        "0.7-0.8": [],
        "0.8+": [],
    }
    rows = (
        db.query(Prediction.confidence, Outcome.filtered_label)
        .join(Outcome, Outcome.prediction_id == Prediction.id)
        .filter(Outcome.filtered_label.isnot(None))
        .all()
    )

    for confidence, label in rows:
        buckets[_confidence_bucket(float(confidence))].append(int(label))

    return {
        bucket: {
            "accuracy": round(sum(labels) / len(labels), 4) if labels else 0.0,
            "samples": len(labels),
        }
        for bucket, labels in buckets.items()
    }


def validate_dataset(db: Session, *, limit: int = 10_000) -> dict[str, Any]:
    dataset = export_training_dataset(db, limit=limit)
    issues: list[str] = []
    labels = [int(row["label"]) for row in dataset]
    feature_names = sorted({key for row in dataset for key in row["features"]})

    if not dataset:
        issues.append("dataset is empty")

    missing_features = _missing_feature_issues(dataset, feature_names)
    issues.extend(missing_features)

    class_balance = _class_balance(labels)
    if labels and (class_balance.get("positive", 0.0) < 0.1 or class_balance.get("positive", 0.0) > 0.9):
        issues.append("label balance is extremely skewed")

    constant_columns = _constant_feature_columns(dataset, feature_names)
    if constant_columns:
        issues.append(f"constant feature columns: {', '.join(constant_columns[:10])}")

    return {
        "num_samples": len(dataset),
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
    order_column = Prediction.timestamp.asc() if chronological else Outcome.computed_at.desc()
    rows = (
        db.query(FeatureSnapshot, Outcome)
        .join(Prediction, Prediction.id == FeatureSnapshot.prediction_id)
        .join(Outcome, Outcome.prediction_id == Prediction.id)
        .filter(Outcome.filtered_label.isnot(None))
        .order_by(order_column)
        .limit(limit)
        .all()
    )

    dataset: list[dict[str, Any]] = []
    for snapshot, outcome in rows:
        dataset.append(
            {
                "features": flatten_feature_snapshot(
                    snapshot.event_features,
                    snapshot.market_features,
                    snapshot.derived_features,
                ),
                "label": int(outcome.filtered_label),
                "timestamp": snapshot.prediction.timestamp,
            }
        )
    return dataset


def _strip_timestamp(row: dict[str, Any]) -> dict[str, Any]:
    return {"features": row["features"], "label": row["label"]}


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


def _missing_feature_issues(dataset: list[dict[str, Any]], feature_names: list[str]) -> list[str]:
    issues: list[str] = []
    for row_index, row in enumerate(dataset):
        features = row["features"]
        for feature_name in feature_names:
            value = features.get(feature_name)
            if value is None or _is_invalid_number(value):
                issues.append(f"missing feature value at row {row_index}: {feature_name}")
                return issues
    return issues


def _constant_feature_columns(dataset: list[dict[str, Any]], feature_names: list[str]) -> list[str]:
    constant_columns: list[str] = []
    for feature_name in feature_names:
        values = {row["features"].get(feature_name) for row in dataset}
        if len(values) <= 1 and dataset:
            constant_columns.append(feature_name)
    return constant_columns


def _is_invalid_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return True
    return math.isnan(float(value)) or math.isinf(float(value))
