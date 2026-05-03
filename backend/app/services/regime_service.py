from __future__ import annotations

import logging

from app.services.cache_service import cache
from app.services.market_service import fetch_price

logger = logging.getLogger(__name__)

_REGIME_CACHE_TTL = 3600  # 1 hour — VIX doesn't tick minute-to-minute


def fetch_market_regime() -> dict[str, float | int]:
    cached = cache.get("market_regime")
    if cached is not None:
        return cached

    result: dict[str, float | int] = {
        "vix_level": 20.0,
        "vix_regime_encoded": 1,
        "spy_trend": 0.0,
        "rate_level": 4.0,
        "market_regime_encoded": 1,
    }

    try:
        vix_data = fetch_price("^VIX")
        vix = float(vix_data["price"])
        result["vix_level"] = round(vix, 2)
        result["vix_regime_encoded"] = _encode_vix(vix)
    except Exception as exc:
        logger.warning("VIX fetch failed: %s", exc)

    try:
        spy_data = fetch_price("SPY")
        spy_history = [float(p) for p in spy_data.get("history", [])]
        result["spy_trend"] = round(_trend(spy_history, window=20), 6)
    except Exception as exc:
        logger.warning("SPY trend fetch failed: %s", exc)

    try:
        tnx_data = fetch_price("^TNX")
        result["rate_level"] = round(float(tnx_data["price"]), 3)
    except Exception as exc:
        logger.warning("TNX fetch failed: %s", exc)

    result["market_regime_encoded"] = _encode_regime(
        float(result["vix_level"]),
        float(result["spy_trend"]),
    )

    cache.set("market_regime", result, _REGIME_CACHE_TTL)
    return result


def _encode_vix(vix: float) -> int:
    if vix < 15:
        return 0  # calm
    if vix < 25:
        return 1  # normal
    if vix < 35:
        return 2  # stressed
    return 3  # crisis


def _encode_regime(vix: float, spy_trend: float) -> int:
    if vix < 20 and spy_trend > 0:
        return 0  # risk-on
    if vix > 30 or spy_trend < -0.05:
        return 2  # risk-off
    return 1  # neutral


def _trend(history: list[float], window: int) -> float:
    if len(history) < 2:
        return 0.0
    recent = history[-1]
    window_slice = history[-window:] if len(history) >= window else history
    ma = sum(window_slice) / len(window_slice)
    return (recent - ma) / ma if ma != 0 else 0.0
