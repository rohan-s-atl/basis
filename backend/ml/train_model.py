from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from xgboost import XGBClassifier

DEFAULT_EXPORT_URL = "http://127.0.0.1:8001/export-training-data"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "xgboost_model.json"
TRAIN_FRACTION = 0.8
CONFIDENCE_BUCKETS = (
    ("0.5-0.6", 0.5, 0.6),
    ("0.6-0.7", 0.6, 0.7),
    ("0.7-0.8", 0.7, 0.8),
    ("0.8+", 0.8, math.inf),
)


@dataclass(frozen=True)
class TrainingData:
    X: pd.DataFrame
    y: pd.Series
    feature_names: list[str]


@dataclass(frozen=True)
class TrainingResult:
    dataset_size: int
    label_balance: dict[str, float | int]
    train_size: int
    test_size: int
    accuracy: float
    roc_auc: float | None
    classification_report: dict[str, Any]
    top_features: list[dict[str, float | str]]
    confidence_analysis: dict[str, dict[str, float | int]]
    model_path: str


def load_training_data(api_url: str = DEFAULT_EXPORT_URL, *, timeout: float = 30.0) -> TrainingData:
    response = requests.get(api_url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    feature_names = list(payload["feature_names"])
    X = pd.DataFrame(payload["features"], columns=feature_names)
    y = pd.Series(payload["labels"], name="label")

    return TrainingData(X=X, y=y, feature_names=feature_names)


def time_aware_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_fraction: float = TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_index = int(len(X) * train_fraction)
    return (
        X.iloc[:split_index],
        X.iloc[split_index:],
        y.iloc[:split_index],
        y.iloc[split_index:],
    )


def build_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )


def feature_importance_frame(
    model: XGBClassifier,
    feature_names: list[str],
) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def confidence_analysis(
    y_true: pd.Series,
    y_pred: np.ndarray,
    positive_probabilities: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    confidence = np.maximum(positive_probabilities, 1.0 - positive_probabilities)
    results: dict[str, dict[str, float | int]] = {}

    for name, lower_bound, upper_bound in CONFIDENCE_BUCKETS:
        if math.isinf(upper_bound):
            mask = confidence >= lower_bound
        else:
            mask = (confidence >= lower_bound) & (confidence < upper_bound)

        count = int(mask.sum())
        if count:
            bucket_accuracy = float(accuracy_score(y_true.iloc[mask], y_pred[mask]))
        else:
            bucket_accuracy = 0.0

        results[name] = {
            "accuracy": round(bucket_accuracy, 4),
            "samples": count,
        }

    return results


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float | None, dict[str, Any], dict[str, dict[str, float | int]]]:
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    roc_auc = _safe_roc_auc(y_test, probabilities)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    confidence = confidence_analysis(y_test, y_pred, probabilities)

    return accuracy, roc_auc, report, confidence


def train_xgboost_model(
    *,
    api_url: str = DEFAULT_EXPORT_URL,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    timeout: float = 30.0,
) -> TrainingResult:
    data = load_training_data(api_url, timeout=timeout)
    _validate_training_data(data)

    X_train, X_test, y_train, y_test = time_aware_split(data.X, data.y)
    _validate_split(y_train, y_test)

    model = build_model()
    model.fit(X_train, y_train)

    accuracy, roc_auc, report, confidence = evaluate_model(model, X_test, y_test)
    importance = feature_importance_frame(model, data.feature_names)

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)

    result = TrainingResult(
        dataset_size=len(data.y),
        label_balance=_label_balance(data.y),
        train_size=len(y_train),
        test_size=len(y_test),
        accuracy=round(accuracy, 4),
        roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
        classification_report=report,
        top_features=_top_feature_records(importance, limit=15),
        confidence_analysis=confidence,
        model_path=str(output_path),
    )
    print_training_summary(result)
    return result


def print_training_summary(result: TrainingResult) -> None:
    print(f"Dataset size: {result.dataset_size}")
    print(f"Label balance: {result.label_balance}")
    print(f"Train size: {result.train_size}")
    print(f"Test size: {result.test_size}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"ROC AUC: {result.roc_auc:.4f}" if result.roc_auc is not None else "ROC AUC: N/A")
    print("Classification report:")
    print(result.classification_report)
    print("Top 15 features:")
    for item in result.top_features:
        print(f"  {item['feature']}: {float(item['importance']):.6f}")
    print("Confidence analysis:")
    for bucket, metrics in result.confidence_analysis.items():
        print(
            f"  {bucket}: accuracy={float(metrics['accuracy']):.4f}, "
            f"samples={int(metrics['samples'])}"
        )
    print(f"Saved model: {result.model_path}")


def result_to_dict(result: TrainingResult) -> dict[str, Any]:
    return asdict(result)


def _validate_training_data(data: TrainingData) -> None:
    if data.X.empty or data.y.empty:
        raise ValueError("Training dataset is empty.")
    if len(data.X) != len(data.y):
        raise ValueError("Feature row count does not match label count.")
    if not data.feature_names:
        raise ValueError("Training dataset has no feature names.")
    if data.X.isnull().any().any():
        raise ValueError("Training dataset contains null feature values.")
    if data.y.nunique() < 2:
        raise ValueError("Training labels must contain at least two classes.")


def _validate_split(y_train: pd.Series, y_test: pd.Series) -> None:
    if y_train.empty or y_test.empty:
        raise ValueError("Train/test split produced an empty partition.")
    if y_train.nunique() < 2:
        raise ValueError("Training split must contain at least two label classes.")


def _safe_roc_auc(y_true: pd.Series, probabilities: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _label_balance(y: pd.Series) -> dict[str, float | int]:
    total = len(y)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    return {
        "positive": round(positives / total, 4) if total else 0.0,
        "negative": round(negatives / total, 4) if total else 0.0,
        "positive_count": positives,
        "negative_count": negatives,
    }


def _top_feature_records(
    importance: pd.DataFrame,
    *,
    limit: int,
) -> list[dict[str, float | str]]:
    return [
        {
            "feature": str(row.feature),
            "importance": float(row.importance),
        }
        for row in importance.head(limit).itertuples(index=False)
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first XGBoost model.")
    parser.add_argument("--api-url", default=DEFAULT_EXPORT_URL)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_xgboost_model(
        api_url=args.api_url,
        model_path=args.model_path,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
