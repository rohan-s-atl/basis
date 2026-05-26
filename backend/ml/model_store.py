from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent / "models"
_RAW_MODEL_PATH = _MODELS_DIR / "xgboost_model.json"
_CALIBRATED_MODEL_PATH = _MODELS_DIR / "calibrated_model.pkl"
_FEATURE_NAMES_PATH = _MODELS_DIR / "feature_names.json"
_DECISION_THRESHOLD_PATH = _MODELS_DIR / "decision_threshold.json"

_cached_raw: XGBClassifier | None = None
_cached_raw_mtime: float | None = None

_cached_calibrated: object | None = None
_cached_calibrated_mtime: float | None = None

_cached_feature_names: list[str] | None = None
_cached_feature_names_mtime: float | None = None

_cached_decision_threshold: float | None = None
_cached_decision_threshold_mtime: float | None = None


def get_model(path: Path = _RAW_MODEL_PATH) -> XGBClassifier | None:
    global _cached_raw, _cached_raw_mtime
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _cached_raw is not None and mtime == _cached_raw_mtime:
        return _cached_raw
    try:
        model = XGBClassifier()
        model.load_model(path)
        _cached_raw = model
        _cached_raw_mtime = mtime
        logger.info("Loaded XGBoost model from %s", path)
    except Exception as exc:
        logger.warning("Failed to load XGBoost model: %s", exc)
        return None
    return _cached_raw


def get_calibrated_model(path: Path = _CALIBRATED_MODEL_PATH) -> object | None:
    global _cached_calibrated, _cached_calibrated_mtime
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _cached_calibrated is not None and mtime == _cached_calibrated_mtime:
        return _cached_calibrated
    try:
        model = joblib.load(path)
        _cached_calibrated = model
        _cached_calibrated_mtime = mtime
        logger.info("Loaded calibrated model from %s", path)
    except Exception as exc:
        logger.warning("Failed to load calibrated model: %s", exc)
        return None
    return _cached_calibrated


def get_feature_names(path: Path = _FEATURE_NAMES_PATH) -> list[str] | None:
    global _cached_feature_names, _cached_feature_names_mtime
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _cached_feature_names is not None and mtime == _cached_feature_names_mtime:
        return _cached_feature_names
    try:
        _cached_feature_names = json.loads(path.read_text())
        _cached_feature_names_mtime = mtime
    except Exception as exc:
        logger.warning("Failed to load feature names: %s", exc)
        return None
    return _cached_feature_names


def get_decision_threshold(path: Path = _DECISION_THRESHOLD_PATH) -> float:
    global _cached_decision_threshold, _cached_decision_threshold_mtime
    if not path.exists():
        return 0.5
    mtime = path.stat().st_mtime
    if _cached_decision_threshold is not None and mtime == _cached_decision_threshold_mtime:
        return _cached_decision_threshold
    try:
        payload = json.loads(path.read_text())
        threshold = float(payload.get("threshold", 0.5))
        _cached_decision_threshold = min(0.95, max(0.05, threshold))
        _cached_decision_threshold_mtime = mtime
    except Exception as exc:
        logger.warning("Failed to load decision threshold: %s", exc)
        return 0.5
    return _cached_decision_threshold


def invalidate_cache() -> None:
    global _cached_raw, _cached_raw_mtime
    global _cached_calibrated, _cached_calibrated_mtime
    global _cached_feature_names, _cached_feature_names_mtime
    global _cached_decision_threshold, _cached_decision_threshold_mtime
    _cached_raw = _cached_raw_mtime = None
    _cached_calibrated = _cached_calibrated_mtime = None
    _cached_feature_names = _cached_feature_names_mtime = None
    _cached_decision_threshold = _cached_decision_threshold_mtime = None
