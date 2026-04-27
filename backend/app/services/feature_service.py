import math

from app.db.models import EventRecord

_SEVERITY_SCORE = {
    "low": 0.25,
    "medium": 0.5,
    "high": 0.8,
    "critical": 1.0,
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


def _direction_score(direction: str) -> float:
    if direction == "positive":
        return 1.0
    if direction == "negative":
        return -1.0
    return 0.0
