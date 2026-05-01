import math
from datetime import datetime
from typing import Any

from app.db.models import Event, EventRecord

_SEVERITY_SCORE = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.8,
    "critical": 1.0,
}

_IMPACT_SENTIMENT = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
}

_ASSET_CLASS_HINTS = {
    "energy": {"XLE", "USO"},
    "commodities": {"GLD", "DBC", "USO"},
    "technology": {"QQQ", "XLK"},
    "financials": {"XLF"},
    "airlines": {"JETS"},
    "broad_market": {"SPY", "QQQ"},
    "rates": {"TLT", "TIP"},
}


def build_feature_vector(
    event: EventRecord,
    *,
    symbol: str,
    entry_price: float,
    exit_price: float,
    return_pct: float,
    sectors: list[str],
) -> dict[str, float | str | bool]:
    symbol_upper = symbol.upper()
    severity_score = _SEVERITY_SCORE.get(event.severity, 0.25)
    direction_score = _direction_score(event.impact_direction)
    sector_match_count = sum(1 for sector in sectors if symbol_upper in _ASSET_CLASS_HINTS.get(sector, set()))
    asset_specificity = min(1.0, sector_match_count / max(1, len(sectors)))
    volatility_proxy = abs(return_pct)

    return {
        "symbol": symbol_upper,
        "event_type": event.event_type,
        "confidence": round(event.confidence, 4),
        "severity_score": severity_score,
        "direction_score": direction_score,
        "sector_count": len(sectors),
        "asset_specificity": round(asset_specificity, 4),
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "return_pct": round(return_pct, 4),
        "volatility_proxy": round(volatility_proxy, 4),
        "is_sector_etf": symbol_upper.startswith("XL") or symbol_upper in {"XLE", "XLF", "XLK", "JETS"},
        "is_macro_etf": symbol_upper in {"SPY", "QQQ", "GLD", "USO", "DBC", "TLT", "TIP"},
    }


def score_signal_quality(features: dict[str, float | str | bool]) -> float:
    """Small interpretable scoring model for ranking signal quality.

    This is intentionally lightweight: it behaves like a logistic model but keeps
    coefficients visible and easy to tune before a trained model is introduced.
    """
    confidence = float(features["confidence"])
    severity_score = float(features["severity_score"])
    asset_specificity = float(features["asset_specificity"])
    volatility_proxy = min(float(features["volatility_proxy"]) / 5, 1.0)
    is_macro_etf = 1.0 if features["is_macro_etf"] else 0.0

    z = (
        -1.15
        + 1.75 * confidence
        + 0.9 * severity_score
        + 0.65 * asset_specificity
        + 0.35 * is_macro_etf
        - 0.45 * volatility_proxy
    )
    return round(1 / (1 + math.exp(-z)), 4)


def severity_to_score(severity: str | float | int | None) -> float:
    if isinstance(severity, (float, int)):
        return _clamp(float(severity), 0.0, 1.0)
    return _SEVERITY_SCORE.get(str(severity or "").lower(), 0.25)


def sentiment_from_classification(classification: dict[str, Any]) -> float:
    direction = str(classification.get("impact_direction", "neutral")).lower()
    confidence = _clamp(float(classification.get("confidence", 0.0)), 0.0, 1.0)
    return round(_IMPACT_SENTIMENT.get(direction, 0.0) * confidence, 4)


def build_event_features(event: Event) -> dict[str, float | str]:
    return {
        "event_type": event.event_type,
        "region": event.region,
        "sentiment": round(event.sentiment, 4),
        "severity": round(event.severity, 4),
        "source": event.source,
        "model_version": event.model_version,
        "event_timestamp": _datetime_to_iso(event.timestamp),
    }


def build_market_features(
    *,
    asset: str,
    price: float,
    history: list[float] | None = None,
) -> dict[str, float | str | int]:
    clean_history = [float(value) for value in history or [] if value is not None]
    return {
        "asset": asset.upper(),
        "price": round(float(price), 4),
        "volatility": round(_volatility_from_history(clean_history), 6),
        "history_points": len(clean_history),
    }


def build_derived_features(
    *,
    event_features: dict[str, Any],
    market_features: dict[str, Any],
    score: float,
    horizon: str,
) -> dict[str, float | str]:
    sentiment = float(event_features.get("sentiment", 0.0))
    severity = float(event_features.get("severity", 0.0))
    volatility = float(market_features.get("volatility", 0.0))
    return {
        "baseline_score": round(score, 6),
        "horizon": horizon,
        "sentiment_x_severity": round(sentiment * severity, 6),
        "severity_x_volatility": round(severity * volatility, 6),
    }


def flatten_feature_snapshot(
    event_features: dict[str, Any],
    market_features: dict[str, Any],
    derived_features: dict[str, Any],
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for prefix, values in (
        ("event", event_features),
        ("market", market_features),
        ("derived", derived_features),
    ):
        for key, value in values.items():
            flattened[f"{prefix}_{key}"] = value
    return flattened


def _direction_score(direction: str) -> float:
    if direction == "positive":
        return 1.0
    if direction == "negative":
        return -1.0
    return 0.0


def _volatility_from_history(history: list[float]) -> float:
    if len(history) < 2:
        return 0.0

    returns = []
    for previous, current in zip(history, history[1:]):
        if previous == 0:
            continue
        returns.append((current - previous) / previous)

    if not returns:
        return 0.0

    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    return math.sqrt(variance)


def _datetime_to_iso(value: datetime) -> str:
    return value.isoformat()


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
