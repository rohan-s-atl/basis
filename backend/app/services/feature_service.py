import math
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

_EVENT_TYPE_ENCODING = {
    "general_market": 0,
    "geopolitical_conflict": 1,
    "inflation": 2,
    "interest_rate_change": 3,
    "economic_data": 4,
    "corporate_earnings": 5,
    "supply_shock": 6,
}

_ASSET_CLASS_ENCODING = {
    "broad_market": 0,
    "energy": 1,
    "commodities": 2,
    "technology": 3,
    "financials": 4,
    "airlines": 5,
    "rates": 6,
    "unknown": 7,
}

LEAKY_DERIVED_FEATURES = {
    "historical_accuracy_of_event_type",
    "rolling_accuracy_of_asset_predictions",
    "event_asset_avg_return",
    "event_asset_accuracy",
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


def build_event_features(event: Event) -> dict[str, float | int]:
    timestamp = event.timestamp
    features: dict[str, float | int] = {
        "event_type_encoded": encode_event_type(event.event_type),
        "source_encoded": _stable_string_encoding(event.source),
        "sentiment": round(event.sentiment, 4),
        "severity": round(event.severity, 4),
        "event_timestamp_unix": int(timestamp.timestamp()),
        "event_day_of_week": int(timestamp.weekday()),
        "event_hour_utc": int(timestamp.hour),
        "surprise_magnitude": 0.0,
        "text_embedding_norm": 0.0,
        "text_embedding_pos_sim": 0.5,
        "text_embedding_neg_sim": 0.5,
        "text_embedding_vol_sim": 0.5,
    }

    raw_embedding = getattr(event, "text_embedding", None)
    if raw_embedding:
        try:
            import json
            from app.services.embedding_service import anchor_similarities, embedding_norm
            emb = json.loads(raw_embedding)
            features["text_embedding_norm"] = embedding_norm(emb)
            sims = anchor_similarities(emb)
            features["text_embedding_pos_sim"] = sims["pos_sim"]
            features["text_embedding_neg_sim"] = sims["neg_sim"]
            features["text_embedding_vol_sim"] = sims["vol_sim"]
        except Exception:
            pass

    return features


def build_market_features(
    *,
    asset: str,
    price: float,
    history: list[float] | None = None,
    sector_return_5d: float | None = None,
    sector_return_10d: float | None = None,
    beta_to_spy: float = 1.0,
    volume_ratio: float = 1.0,
    dollar_trend: float = 0.0,
    oil_trend: float = 0.0,
    spy_return_20d: float = 0.0,
) -> dict[str, float | int]:
    clean_history = [float(value) for value in history or [] if value is not None]
    asset_class = asset_class_for_symbol(asset)
    rolling_volatility = _rolling_volatility(clean_history, window=5)
    return_5d = _return_over_period(clean_history, 5)
    return_10d = _return_over_period(clean_history, 10)
    return_20d = _return_over_period(clean_history, 20)
    effective_sector_return_5d = return_5d if sector_return_5d is None else sector_return_5d
    effective_sector_return_10d = return_10d if sector_return_10d is None else sector_return_10d
    return {
        "asset_encoded": encode_asset(asset),
        "asset_class_encoded": encode_asset_class(asset_class),
        "price": round(float(price), 4),
        "return_1d": round(_return_over_period(clean_history, 1), 6),
        "return_5d": round(return_5d, 6),
        "return_10d": round(return_10d, 6),
        "return_20d": round(return_20d, 6),
        "spy_return_20d": round(spy_return_20d, 6),
        "sector_return_5d": round(effective_sector_return_5d, 6),
        "sector_return_10d": round(effective_sector_return_10d, 6),
        "relative_strength_5d": round(return_5d - effective_sector_return_5d, 6),
        "rolling_volatility": round(rolling_volatility, 6),
        "volatility": round(rolling_volatility, 6),
        "asset_momentum_20d": round(return_20d, 6),
        "asset_beta_to_spy": round(beta_to_spy, 6),
        "volume_ratio": round(volume_ratio, 6),
        "dollar_trend": round(dollar_trend, 6),
        "oil_trend": round(oil_trend, 6),
        "history_points": len(clean_history),
    }


def build_derived_features(
    *,
    event_features: dict[str, Any],
    market_features: dict[str, Any],
    score: float,
    horizon: str,
    sector_sensitivity: float,
    historical_accuracy_of_event_type: float,
    rolling_accuracy_of_asset_predictions: float,
    event_asset_avg_return: float,
    event_asset_accuracy: float,
    vix_level: float = 20.0,
    vix_regime_encoded: int = 1,
    spy_trend: float = 0.0,
    rate_level: float = 4.0,
    market_regime_encoded: int = 1,
    asset_exposure_sign: float = 1.0,
    asset_exposure_confidence: float = 0.5,
) -> dict[str, float | int]:
    sentiment = float(event_features.get("sentiment", 0.0))
    severity = float(event_features.get("severity", 0.0))
    event_type_encoded = int(event_features.get("event_type_encoded", 0))
    asset_class_encoded = int(market_features.get("asset_class_encoded", _ASSET_CLASS_ENCODING["unknown"]))
    volatility = float(market_features.get("rolling_volatility", 0.0))
    return {
        "baseline_score": round(score, 6),
        "horizon_days": _horizon_to_days(horizon),
        "sentiment_x_severity": round(sentiment * severity, 6),
        "sentiment_x_sector_sensitivity": round(sentiment * sector_sensitivity, 6),
        "event_type_asset_class_interaction": event_type_encoded * asset_class_encoded,
        "historical_accuracy_of_event_type": round(historical_accuracy_of_event_type, 6),
        "rolling_accuracy_of_asset_predictions": round(rolling_accuracy_of_asset_predictions, 6),
        "event_asset_avg_return": round(event_asset_avg_return, 6),
        "event_asset_accuracy": round(event_asset_accuracy, 6),
        "severity_x_volatility": round(severity * volatility, 6),
        "vix_level": round(vix_level, 4),
        "vix_regime_encoded": vix_regime_encoded,
        "spy_trend": round(spy_trend, 6),
        "rate_level": round(rate_level, 4),
        "market_regime_encoded": market_regime_encoded,
        "asset_exposure_sign": round(asset_exposure_sign, 4),
        "asset_exposure_confidence": round(asset_exposure_confidence, 4),
        "recent_macro_cluster_count": 0,
        "repeated_event_type": 0,
        "event_novelty": 1.0,
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
            if prefix == "derived" and key in LEAKY_DERIVED_FEATURES:
                continue
            if not isinstance(value, (int, float, bool)):
                continue
            flattened[f"{prefix}_{key}"] = _numeric_feature_value(value)
    return flattened


def encode_event_type(event_type: str) -> int:
    return _EVENT_TYPE_ENCODING.get(event_type, _EVENT_TYPE_ENCODING["general_market"])


def asset_class_for_symbol(asset: str) -> str:
    symbol = asset.upper()
    for asset_class, symbols in _ASSET_CLASS_HINTS.items():
        if symbol in symbols:
            return asset_class
    if symbol.startswith("XL"):
        return "broad_market"
    return "unknown"


def sector_symbols_for_asset(asset: str) -> list[str]:
    asset_class = asset_class_for_symbol(asset)
    symbols = sorted(_ASSET_CLASS_HINTS.get(asset_class, set()))
    if symbols:
        return symbols
    return [asset.upper()]


def encode_asset_class(asset_class: str) -> int:
    return _ASSET_CLASS_ENCODING.get(asset_class, _ASSET_CLASS_ENCODING["unknown"])


def encode_asset(asset: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(asset.upper()))


def sector_sensitivity_for_asset(asset: str, sectors: list[str]) -> float:
    if not sectors:
        return 0.0

    symbol = asset.upper()
    matched = sum(1 for sector in sectors if symbol in _ASSET_CLASS_HINTS.get(sector, set()))
    if matched:
        return round(matched / len(sectors), 6)

    asset_class = asset_class_for_symbol(symbol)
    return 0.5 if asset_class in sectors else 0.0


def average_return_over_histories(histories: list[list[float]], *, periods: int) -> float:
    returns = [_return_over_period(history, periods) for history in histories if history]
    if not returns:
        return 0.0
    return sum(returns) / len(returns)


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


def _rolling_volatility(history: list[float], *, window: int) -> float:
    if len(history) < 2:
        return 0.0
    windowed_history = history[-(window + 1):]
    return _volatility_from_history(windowed_history)


def _return_over_period(history: list[float], periods: int) -> float:
    if len(history) <= periods:
        return 0.0
    start_price = history[-(periods + 1)]
    end_price = history[-1]
    if start_price == 0:
        return 0.0
    return (end_price - start_price) / start_price


def _horizon_to_days(horizon: str) -> int:
    try:
        return int(horizon.lower().replace("d", "").split("-")[0])
    except (ValueError, AttributeError):
        return 1


def _numeric_feature_value(value: Any) -> float | int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    return 0.0


def _stable_string_encoding(value: str | None) -> int:
    text = str(value or "unknown").lower()
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % 10_000


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
