from __future__ import annotations

import logging
from typing import Any

from app.services.baseline_scoring import BASELINE_MODEL_VERSION, BaselinePrediction, PredictionDirection
from app.services.feature_service import flatten_feature_snapshot

logger = logging.getLogger(__name__)

ML_MODEL_VERSION = "xgboost-v1"


def score_with_ml_model(
    event_features: dict[str, Any],
    market_features: dict[str, Any],
    derived_features: dict[str, Any],
    fallback: BaselinePrediction,
) -> tuple[BaselinePrediction, str]:
    """Return (prediction, model_version). Uses calibrated model when available, falls back to baseline."""
    from ml.model_store import get_calibrated_model, get_feature_names, get_model

    feature_names = get_feature_names()
    calibrated = get_calibrated_model()
    raw = get_model()

    predictor = calibrated or raw
    if predictor is None or feature_names is None:
        return fallback, BASELINE_MODEL_VERSION

    try:
        features = flatten_feature_snapshot(event_features, market_features, derived_features)
        X = [[float(features.get(f, 0.0)) for f in feature_names]]

        proba = predictor.predict_proba(X)[0]
        p_up = float(proba[1])

        direction: PredictionDirection = "up" if p_up >= 0.5 else "down"
        confidence = round(max(p_up, 1.0 - p_up), 4)
        score = round(p_up * 2.0 - 1.0, 6)

        return BaselinePrediction(
            predicted_direction=direction,
            confidence=confidence,
            score=score,
        ), ML_MODEL_VERSION

    except Exception as exc:
        logger.warning("ML inference failed, falling back to baseline: %s", exc)
        return fallback, BASELINE_MODEL_VERSION


def explain_prediction(
    event_features: dict[str, Any],
    market_features: dict[str, Any],
    derived_features: dict[str, Any],
) -> list[dict[str, float | str]]:
    """Return top SHAP contributors for a single prediction. Returns [] if model unavailable."""
    from ml.model_store import get_feature_names, get_model

    raw = get_model()
    feature_names = get_feature_names()
    if raw is None or feature_names is None:
        return []

    try:
        import numpy as np
        import shap

        features = flatten_feature_snapshot(event_features, market_features, derived_features)
        X_row = [[float(features.get(f, 0.0)) for f in feature_names]]

        import pandas as pd
        X_df = pd.DataFrame(X_row, columns=feature_names)

        explainer = shap.TreeExplainer(raw)
        shap_values = explainer.shap_values(X_df)
        contributions = shap_values[0] if hasattr(shap_values, "__len__") else shap_values

        return sorted(
            [
                {"feature": f, "shap_value": round(float(v), 6)}
                for f, v in zip(feature_names, contributions)
            ],
            key=lambda x: abs(float(x["shap_value"])),
            reverse=True,
        )[:10]

    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        return []
