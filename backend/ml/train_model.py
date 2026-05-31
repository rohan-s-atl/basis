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
DEFAULT_DECISION_THRESHOLD_PATH = _MODELS_DIR / "decision_threshold.json"

TRAIN_FRACTION = 0.8
VALIDATION_FRACTION = 0.15
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
class DecisionThreshold:
    threshold: float
    accuracy: float
    default_accuracy: float
    positive_rate: float


@dataclass(frozen=True)
class SegmentModelResult:
    segment: str
    samples: int
    accuracy: float
    roc_auc: float | None
    threshold: float


@dataclass(frozen=True)
class TrainingResult:
    dataset_size: int
    label_balance: dict[str, float | int]
    train_size: int
    validation_size: int
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
    decision_threshold: DecisionThreshold
    segment_models: list[SegmentModelResult]
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

def build_xgboost(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )


def build_logistic() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight="balanced")),
    ])


def _compute_scale_pos_weight(y: pd.Series) -> float:
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return 1.0
    return round(n_neg / n_pos, 4)


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
            model = build_xgboost(scale_pos_weight=_compute_scale_pos_weight(y_tr))
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

def calibrate(X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float = 1.0) -> CalibratedClassifierCV:
    calibrated = CalibratedClassifierCV(build_xgboost(scale_pos_weight=scale_pos_weight), method="sigmoid", cv=3)
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


def optimize_decision_threshold(
    y_true: pd.Series,
    positive_probabilities: np.ndarray,
) -> DecisionThreshold:
    default_pred = (positive_probabilities >= 0.5).astype(int)
    default_accuracy = float(accuracy_score(y_true, default_pred))
    best_threshold = 0.5
    best_accuracy = default_accuracy
    best_positive_rate = float(default_pred.mean()) if len(default_pred) else 0.0

    # Keep the search away from degenerate all-up/all-down cutoffs.
    for threshold in np.arange(0.30, 0.701, 0.005):
        pred = (positive_probabilities >= threshold).astype(int)
        positive_rate = float(pred.mean()) if len(pred) else 0.0
        if positive_rate < 0.05 or positive_rate > 0.95:
            continue
        acc = float(accuracy_score(y_true, pred))
        if acc > best_accuracy or (acc == best_accuracy and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_threshold = float(threshold)
            best_accuracy = acc
            best_positive_rate = positive_rate

    return DecisionThreshold(
        threshold=round(best_threshold, 4),
        accuracy=round(best_accuracy, 4),
        default_accuracy=round(default_accuracy, 4),
        positive_rate=round(best_positive_rate, 4),
    )


def train_segment_models(data: TrainingData, *, base_dir: Path, min_samples: int = 30) -> list[SegmentModelResult]:
    """Train semi-separate calibrated models by horizon when the segment is large enough."""
    horizon_feature = "derived_horizon_days"
    if horizon_feature not in data.X.columns:
        return []

    results: list[SegmentModelResult] = []
    horizons = sorted({int(value) for value in data.X[horizon_feature].dropna().tolist()})
    for horizon in horizons:
        mask = data.X[horizon_feature].astype(int) == horizon
        X_segment = data.X.loc[mask]
        y_segment = data.y.loc[mask]
        if len(y_segment) < min_samples or y_segment.nunique() < 2:
            continue

        split = int(len(X_segment) * 0.8)
        X_train, X_test = X_segment.iloc[:split], X_segment.iloc[split:]
        y_train, y_test = y_segment.iloc[:split], y_segment.iloc[split:]
        if y_train.nunique() < 2 or y_test.empty or y_test.nunique() < 2:
            continue
        if int(y_train.value_counts().min()) < 3:
            continue

        model = calibrate(X_train, y_train, scale_pos_weight=_compute_scale_pos_weight(y_train))
        proba = model.predict_proba(X_test)[:, 1]
        threshold = optimize_decision_threshold(y_test, proba)
        pred = (proba >= threshold.threshold).astype(int)

        segment_dir = base_dir / f"{horizon}d"
        segment_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, segment_dir / "calibrated_model.pkl")
        (segment_dir / "feature_names.json").write_text(json.dumps(data.feature_names))
        (segment_dir / "decision_threshold.json").write_text(json.dumps(asdict(threshold)))

        results.append(
            SegmentModelResult(
                segment=f"{horizon}d",
                samples=len(y_segment),
                accuracy=round(float(accuracy_score(y_test, pred)), 4),
                roc_auc=round(_safe_roc_auc(y_test, proba), 4) if _safe_roc_auc(y_test, proba) is not None else None,
                threshold=threshold.threshold,
            )
        )

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
    dataset: dict[str, Any] | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    calibrated_path: str | Path = DEFAULT_CALIBRATED_PATH,
    feature_names_path: str | Path = DEFAULT_FEATURE_NAMES_PATH,
    decision_threshold_path: str | Path = DEFAULT_DECISION_THRESHOLD_PATH,
    timeout: float = 30.0,
) -> TrainingResult:
    data = training_data_from_payload(dataset) if dataset is not None else load_training_data(
        api_url,
        timeout=timeout,
    )
    return train_xgboost_from_data(
        data,
        model_path=model_path,
        calibrated_path=calibrated_path,
        feature_names_path=feature_names_path,
        decision_threshold_path=decision_threshold_path,
    )


def train_xgboost_from_payload(
    payload: dict[str, Any],
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    calibrated_path: str | Path = DEFAULT_CALIBRATED_PATH,
    feature_names_path: str | Path = DEFAULT_FEATURE_NAMES_PATH,
    decision_threshold_path: str | Path = DEFAULT_DECISION_THRESHOLD_PATH,
) -> TrainingResult:
    return train_xgboost_from_data(
        training_data_from_payload(payload),
        model_path=model_path,
        calibrated_path=calibrated_path,
        feature_names_path=feature_names_path,
        decision_threshold_path=decision_threshold_path,
    )


def train_xgboost_from_data(
    data: TrainingData,
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    calibrated_path: str | Path = DEFAULT_CALIBRATED_PATH,
    feature_names_path: str | Path = DEFAULT_FEATURE_NAMES_PATH,
    decision_threshold_path: str | Path = DEFAULT_DECISION_THRESHOLD_PATH,
) -> TrainingResult:
    _validate_training_data(data)

    train_end = int(len(data.X) * (1.0 - VALIDATION_FRACTION * 2))
    validation_end = int(len(data.X) * (1.0 - VALIDATION_FRACTION))
    X_train, X_validation, X_test = (
        data.X.iloc[:train_end],
        data.X.iloc[train_end:validation_end],
        data.X.iloc[validation_end:],
    )
    y_train, y_validation, y_test = (
        data.y.iloc[:train_end],
        data.y.iloc[train_end:validation_end],
        data.y.iloc[validation_end:],
    )
    _validate_split(y_train, y_validation, y_test)

    # Compute class weight from training labels to counteract imbalance
    scale_pos_weight = _compute_scale_pos_weight(y_train)

    # Train raw XGBoost
    xgb = build_xgboost(scale_pos_weight=scale_pos_weight)
    xgb.fit(X_train, y_train)

    # Walk-forward CV
    cv_result = walk_forward_cv(data.X, data.y)

    # Model comparison
    comparison = compare_models(X_train, X_test, y_train, y_test, xgb)

    # SHAP on test set
    shap_summary = compute_shap_summary(xgb, X_test if len(X_test) > 0 else X_train)

    # Platt calibration — trains fresh XGBoost with 3-fold CV internally for calibration
    calibrated = calibrate(X_train, y_train, scale_pos_weight=scale_pos_weight)
    cal_metrics = calibration_metrics(xgb, calibrated, X_test, y_test)

    # Tune threshold on validation, then report accuracy on untouched test data.
    validation_proba = calibrated.predict_proba(X_validation)[:, 1]
    threshold = optimize_decision_threshold(y_validation, validation_proba)
    cal_proba = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (cal_proba >= threshold.threshold).astype(int)
    accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    roc_auc = _safe_roc_auc(y_test, cal_proba)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_analysis = confidence_analysis(y_test, y_pred, cal_proba)

    importance = feature_importance_frame(xgb, data.feature_names)

    # Persist
    out_model = Path(model_path)
    out_cal = Path(calibrated_path)
    out_names = Path(feature_names_path)
    out_threshold = Path(decision_threshold_path)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    xgb.save_model(out_model)
    joblib.dump(calibrated, out_cal)
    out_names.write_text(json.dumps(data.feature_names))
    out_threshold.write_text(json.dumps(asdict(threshold)))
    segment_models = train_segment_models(data, base_dir=out_model.parent / "horizons")

    result = TrainingResult(
        dataset_size=len(data.y),
        label_balance=_label_balance(data.y),
        train_size=len(y_train),
        validation_size=len(y_validation),
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
        decision_threshold=threshold,
        segment_models=segment_models,
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
          f"(train={result.train_size}, validation={result.validation_size}, test={result.test_size})")
    print(f"Label balance:  {result.label_balance}")
    print(f"\nAccuracy (calibrated):  {result.accuracy:.4f}")
    print(f"Decision threshold:     {result.decision_threshold.threshold:.4f} "
          f"(default acc={result.decision_threshold.default_accuracy:.4f})")
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

    if result.segment_models:
        print(f"\nSegment models:")
        for item in result.segment_models:
            print(f"  {item.segment:<8} samples={item.samples:<5} acc={item.accuracy:.4f} threshold={item.threshold:.4f}")

    print(f"\nSaved XGBoost model:    {result.model_path}")
    print(f"Saved calibrated model: {result.calibrated_model_path}")
    print("=" * 60)


def result_to_dict(result: TrainingResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["calibrated_accuracy"] = result.accuracy
    payload["calibrated_roc_auc"] = result.roc_auc
    payload["deployment_accuracy"] = result.accuracy
    payload["accuracy_metric"] = "threshold_optimized_calibrated_accuracy"
    return payload


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


def _validate_split(y_train: pd.Series, y_validation: pd.Series, y_test: pd.Series) -> None:
    if y_train.empty or y_validation.empty or y_test.empty:
        raise ValueError("Train/validation/test split produced an empty partition.")
    if y_train.nunique() < 2:
        raise ValueError("Training split must contain at least two label classes.")
    if y_validation.nunique() < 2:
        raise ValueError("Validation split must contain at least two label classes.")
    if y_test.nunique() < 2:
        raise ValueError("Test split must contain at least two label classes.")


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
    result = train_xgboost_model(
        api_url=args.api_url,
        model_path=args.model_path,
        timeout=args.timeout,
    )
    _record_cli_training_run(result)


def _record_cli_training_run(result: TrainingResult) -> None:
    try:
        from app.db.init_db import init_db
        from app.db.session import SessionLocal
        from app.services.training_run_service import save_training_run

        init_db()
        db = SessionLocal()
        try:
            save_training_run(db, result_to_dict(result), triggered_by="cli")
        finally:
            db.close()
    except Exception as exc:
        print(f"Training run history not recorded: {exc}")


if __name__ == "__main__":
    main()
