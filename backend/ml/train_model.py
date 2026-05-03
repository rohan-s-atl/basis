from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DEFAULT_EXPORT_URL = "http://127.0.0.1:8000/export-training-data"
_MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = _MODELS_DIR / "xgboost_model.json"
DEFAULT_CALIBRATED_PATH = _MODELS_DIR / "calibrated_model.pkl"
DEFAULT_FEATURE_NAMES_PATH = _MODELS_DIR / "feature_names.json"

TRAIN_FRACTION = 0.8
MIN_FOLD_SAMPLES = 5
N_CV_FOLDS = 5
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
class ModelComparison:
    xgboost_accuracy: float
    xgboost_roc_auc: float | None
    logistic_accuracy: float
    logistic_roc_auc: float | None
    winner: str


@dataclass(frozen=True)
class CalibrationMetrics:
    brier_score_uncalibrated: float
    brier_score_calibrated: float
    improvement: float


@dataclass(frozen=True)
class WalkForwardCV:
    fold_accuracies: list[float]
    mean_accuracy: float
    std_accuracy: float
    n_folds: int


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
    shap_summary: list[dict[str, float | str]]
    confidence_analysis: dict[str, dict[str, float | int]]
    walk_forward_cv: WalkForwardCV
    model_comparison: ModelComparison
    calibration: CalibrationMetrics
    model_path: str
    calibrated_model_path: str


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(api_url: str = DEFAULT_EXPORT_URL, *, timeout: float = 30.0) -> TrainingData:
    response = requests.get(api_url, timeout=timeout)
    response.raise_for_status()
    return training_data_from_payload(response.json())


def training_data_from_payload(payload: dict[str, Any]) -> TrainingData:
    feature_names = list(payload["feature_names"])
    X = pd.DataFrame(payload["features"], columns=feature_names)
    y = pd.Series(payload["labels"], name="label")
    return TrainingData(X=X, y=y, feature_names=feature_names)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_xgboost() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )


def build_logistic() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    ])


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def walk_forward_cv(X: pd.DataFrame, y: pd.Series) -> WalkForwardCV:
    n = len(X)
    min_train = int(n * (1.0 - 1.0 / N_CV_FOLDS))
    min_train = max(min_train, int(n * 0.5))
    remaining = n - min_train
    fold_size = max(remaining // N_CV_FOLDS, MIN_FOLD_SAMPLES)

    fold_accuracies: list[float] = []
    train_end = min_train

    while train_end + fold_size <= n:
        test_end = train_end + fold_size
        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_te, y_te = X.iloc[train_end:test_end], y.iloc[train_end:test_end]

        if y_tr.nunique() >= 2 and y_te.nunique() >= 2 and len(y_te) >= MIN_FOLD_SAMPLES:
            model = build_xgboost()
            model.fit(X_tr, y_tr)
            fold_accuracies.append(float(accuracy_score(y_te, model.predict(X_te))))

        train_end += fold_size

    if not fold_accuracies:
        return WalkForwardCV(fold_accuracies=[], mean_accuracy=0.0, std_accuracy=0.0, n_folds=0)

    return WalkForwardCV(
        fold_accuracies=[round(a, 4) for a in fold_accuracies],
        mean_accuracy=round(float(np.mean(fold_accuracies)), 4),
        std_accuracy=round(float(np.std(fold_accuracies)), 4),
        n_folds=len(fold_accuracies),
    )


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    xgb: XGBClassifier,
) -> ModelComparison:
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_acc = round(float(accuracy_score(y_test, xgb_pred)), 4)
    xgb_auc = _safe_roc_auc(y_test, xgb_proba)

    lr = build_logistic()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_acc = round(float(accuracy_score(y_test, lr_pred)), 4)
    lr_auc = _safe_roc_auc(y_test, lr_proba)

    winner = "xgboost" if xgb_acc >= lr_acc else "logistic_regression"

    return ModelComparison(
        xgboost_accuracy=xgb_acc,
        xgboost_roc_auc=round(xgb_auc, 4) if xgb_auc is not None else None,
        logistic_accuracy=lr_acc,
        logistic_roc_auc=round(lr_auc, 4) if lr_auc is not None else None,
        winner=winner,
    )


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap_summary(model: XGBClassifier, X: pd.DataFrame) -> list[dict[str, float | str]]:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)
        return sorted(
            [
                {"feature": str(f), "mean_abs_shap": round(float(v), 6)}
                for f, v in zip(X.columns, mean_abs)
            ],
            key=lambda x: float(x["mean_abs_shap"]),
            reverse=True,
        )[:15]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Platt calibration
# ---------------------------------------------------------------------------

def calibrate(X_train: pd.DataFrame, y_train: pd.Series) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(build_xgboost(), method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def calibration_metrics(
    raw_model: XGBClassifier,
    calibrated_model: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> CalibrationMetrics:
    raw_proba = raw_model.predict_proba(X_test)[:, 1]
    cal_proba = calibrated_model.predict_proba(X_test)[:, 1]
    brier_raw = round(float(brier_score_loss(y_test, raw_proba)), 6)
    brier_cal = round(float(brier_score_loss(y_test, cal_proba)), 6)
    return CalibrationMetrics(
        brier_score_uncalibrated=brier_raw,
        brier_score_calibrated=brier_cal,
        improvement=round(brier_raw - brier_cal, 6),
    )


# ---------------------------------------------------------------------------
# Confidence bucket analysis
# ---------------------------------------------------------------------------

def confidence_analysis(
    y_true: pd.Series,
    y_pred: np.ndarray,
    positive_probabilities: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    confidence = np.maximum(positive_probabilities, 1.0 - positive_probabilities)
    results: dict[str, dict[str, float | int]] = {}
    for name, lo, hi in CONFIDENCE_BUCKETS:
        mask = confidence >= lo if math.isinf(hi) else (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        bucket_acc = float(accuracy_score(y_true.iloc[mask], y_pred[mask])) if count else 0.0
        results[name] = {"accuracy": round(bucket_acc, 4), "samples": count}
    return results


# ---------------------------------------------------------------------------
# Feature importance (XGBoost gain)
# ---------------------------------------------------------------------------

def feature_importance_frame(model: XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_xgboost_model(
    *,
    api_url: str = DEFAULT_EXPORT_URL,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    calibrated_path: str | Path = DEFAULT_CALIBRATED_PATH,
    feature_names_path: str | Path = DEFAULT_FEATURE_NAMES_PATH,
    timeout: float = 30.0,
) -> TrainingResult:
    data = load_training_data(api_url, timeout=timeout)
    return train_xgboost_from_data(
        data,
        model_path=model_path,
        calibrated_path=calibrated_path,
        feature_names_path=feature_names_path,
    )


def train_xgboost_from_payload(
    payload: dict[str, Any],
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    calibrated_path: str | Path = DEFAULT_CALIBRATED_PATH,
    feature_names_path: str | Path = DEFAULT_FEATURE_NAMES_PATH,
) -> TrainingResult:
    return train_xgboost_from_data(
        training_data_from_payload(payload),
        model_path=model_path,
        calibrated_path=calibrated_path,
        feature_names_path=feature_names_path,
    )


def train_xgboost_from_data(
    data: TrainingData,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    calibrated_path: str | Path = DEFAULT_CALIBRATED_PATH,
    feature_names_path: str | Path = DEFAULT_FEATURE_NAMES_PATH,
) -> TrainingResult:
    _validate_training_data(data)

    split = int(len(data.X) * TRAIN_FRACTION)
    X_train, X_test = data.X.iloc[:split], data.X.iloc[split:]
    y_train, y_test = data.y.iloc[:split], data.y.iloc[split:]
    _validate_split(y_train, y_test)

    # Train raw XGBoost
    xgb = build_xgboost()
    xgb.fit(X_train, y_train)

    # Walk-forward CV
    cv_result = walk_forward_cv(data.X, data.y)

    # Model comparison
    comparison = compare_models(X_train, X_test, y_train, y_test, xgb)

    # SHAP on test set
    shap_summary = compute_shap_summary(xgb, X_test if len(X_test) > 0 else X_train)

    # Platt calibration — trains fresh XGBoost with 3-fold CV internally for calibration
    calibrated = calibrate(X_train, y_train)
    cal_metrics = calibration_metrics(xgb, calibrated, X_test, y_test)

    # Final evaluation using calibrated model
    y_pred = np.array(calibrated.predict(X_test))
    cal_proba = calibrated.predict_proba(X_test)[:, 1]
    accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    roc_auc = _safe_roc_auc(y_test, cal_proba)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_analysis = confidence_analysis(y_test, y_pred, cal_proba)

    importance = feature_importance_frame(xgb, data.feature_names)

    # Persist
    out_model = Path(model_path)
    out_cal = Path(calibrated_path)
    out_names = Path(feature_names_path)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    xgb.save_model(out_model)
    joblib.dump(calibrated, out_cal)
    out_names.write_text(json.dumps(data.feature_names))

    result = TrainingResult(
        dataset_size=len(data.y),
        label_balance=_label_balance(data.y),
        train_size=len(y_train),
        test_size=len(y_test),
        accuracy=accuracy,
        roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
        classification_report=report,
        top_features=_top_feature_records(importance),
        shap_summary=shap_summary,
        confidence_analysis=conf_analysis,
        walk_forward_cv=cv_result,
        model_comparison=comparison,
        calibration=cal_metrics,
        model_path=str(out_model),
        calibrated_model_path=str(out_cal),
    )
    print_training_summary(result)
    return result


# ---------------------------------------------------------------------------
# Printing / serialisation
# ---------------------------------------------------------------------------

def print_training_summary(result: TrainingResult) -> None:
    print(f"\n{'='*60}")
    print(f"Dataset:        {result.dataset_size} samples  "
          f"(train={result.train_size}, test={result.test_size})")
    print(f"Label balance:  {result.label_balance}")
    print(f"\nAccuracy (calibrated):  {result.accuracy:.4f}")
    print(f"ROC AUC (calibrated):   "
          f"{result.roc_auc:.4f}" if result.roc_auc is not None else "ROC AUC: N/A")

    cv = result.walk_forward_cv
    print(f"\nWalk-forward CV ({cv.n_folds} folds): "
          f"{cv.mean_accuracy:.4f} ± {cv.std_accuracy:.4f}")
    print(f"  Fold accuracies: {cv.fold_accuracies}")

    cmp = result.model_comparison
    print(f"\nModel comparison:")
    print(f"  XGBoost:             acc={cmp.xgboost_accuracy:.4f}  auc={cmp.xgboost_roc_auc}")
    print(f"  Logistic regression: acc={cmp.logistic_accuracy:.4f}  auc={cmp.logistic_roc_auc}")
    print(f"  Winner: {cmp.winner}")

    cal = result.calibration
    print(f"\nPlatt calibration (Brier score — lower is better):")
    print(f"  Uncalibrated: {cal.brier_score_uncalibrated:.6f}")
    print(f"  Calibrated:   {cal.brier_score_calibrated:.6f}  "
          f"(improvement: {cal.improvement:+.6f})")

    print(f"\nTop SHAP features:")
    for item in result.shap_summary[:10]:
        print(f"  {item['feature']:<45} {float(item['mean_abs_shap']):.6f}")

    print(f"\nSaved XGBoost model:    {result.model_path}")
    print(f"Saved calibrated model: {result.calibrated_model_path}")
    print("=" * 60)


def result_to_dict(result: TrainingResult) -> dict[str, Any]:
    return asdict(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_roc_auc(y_true: pd.Series, probabilities: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def _label_balance(y: pd.Series) -> dict[str, float | int]:
    total = len(y)
    positives = int((y == 1).sum())
    return {
        "positive": round(positives / total, 4) if total else 0.0,
        "negative": round((total - positives) / total, 4) if total else 0.0,
        "positive_count": positives,
        "negative_count": total - positives,
    }


def _top_feature_records(importance: pd.DataFrame, limit: int = 15) -> list[dict[str, float | str]]:
    return [
        {"feature": str(row.feature), "importance": round(float(row.importance), 6)}
        for row in importance.head(limit).itertuples(index=False)
    ]


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the XGBoost model.")
    parser.add_argument("--api-url", default=DEFAULT_EXPORT_URL)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_xgboost_model(api_url=args.api_url, model_path=args.model_path, timeout=args.timeout)


if __name__ == "__main__":
    main()
