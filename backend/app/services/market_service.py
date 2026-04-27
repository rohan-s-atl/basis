import yfinance as yf

from app.core.config import settings
from app.services.cache_service import cache

_RANGE_CONFIG = {
    "1d": ("1d", "5m"),
    "5d": ("5d", "15m"),
    "1m": ("1mo", "1d"),
    "6m": ("6mo", "1d"),
    "ytd": ("ytd", "1d"),
    "1y": ("1y", "1d"),
}


def fetch_price(symbol: str) -> dict[str, object]:
    normalized_symbol = symbol.upper()
    cache_key = f"price:{normalized_symbol}"
    cached_price = cache.get(cache_key)
    if cached_price is not None:
        return cached_price

    ticker = yf.Ticker(normalized_symbol)
    history = ticker.history(period="5d")

    if history.empty:
        raise ValueError(f"No market data found for symbol: {normalized_symbol}")

    close_prices = history["Close"].dropna()
    if close_prices.empty:
        raise ValueError(f"No price data found for symbol: {symbol}")

    price = float(close_prices.iloc[-1])
    price_history = [float(value) for value in close_prices.tolist()]

    market_data = {
        "symbol": normalized_symbol,
        "price": price,
        "history": price_history,
    }

    cache.set(cache_key, market_data, settings.price_cache_ttl_seconds)
    return market_data


def fetch_history(symbol: str, range_key: str = "1d") -> dict[str, object]:
    normalized_symbol = symbol.upper()
    normalized_range = range_key.lower()
    period, interval = _RANGE_CONFIG.get(normalized_range, _RANGE_CONFIG["1d"])
    cache_key = f"history:{normalized_symbol}:{normalized_range}"
    cached_history = cache.get(cache_key)
    if cached_history is not None:
        return cached_history

    ticker = yf.Ticker(normalized_symbol)
    history = ticker.history(period=period, interval=interval, auto_adjust=False)

    if history.empty:
        raise ValueError(f"No historical market data found for symbol: {normalized_symbol}")

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    cleaned = history.dropna(subset=["Close"])
    if cleaned.empty:
        raise ValueError(f"No close prices found for symbol: {normalized_symbol}")

    points = []
    for timestamp, row in cleaned.iterrows():
        close = float(row["Close"])
        open_price = float(row["Open"]) if not _is_nan(row["Open"]) else close
        high = float(row["High"]) if not _is_nan(row["High"]) else close
        low = float(row["Low"]) if not _is_nan(row["Low"]) else close
        volume = int(row["Volume"]) if "Volume" in required_columns and not _is_nan(row["Volume"]) else 0

        points.append(
            {
                "timestamp": timestamp.isoformat(),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    if not points:
        raise ValueError(f"No usable historical points found for symbol: {normalized_symbol}")

    closes = [point["close"] for point in points]
    volumes = [point["volume"] for point in points]
    info = {}
    try:
        info = ticker.fast_info or {}
    except Exception:
        info = {}

    history_data = {
        "symbol": normalized_symbol,
        "range": normalized_range,
        "interval": interval,
        "currency": _safe_info_get(info, "currency"),
        "previous_close": _safe_float(_safe_info_get(info, "previous_close")) or points[0]["close"],
        "open": points[0]["open"],
        "high": max(point["high"] for point in points),
        "low": min(point["low"] for point in points),
        "close": closes[-1],
        "volume": sum(volumes),
        "average_volume": int(sum(volumes) / len(volumes)) if volumes else None,
        "points": points,
    }

    cache.set(cache_key, history_data, settings.price_cache_ttl_seconds)
    return history_data


def _is_nan(value: object) -> bool:
    try:
        return value != value
    except Exception:
        return True


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_info_get(info: object, key: str) -> object:
    try:
        if isinstance(info, dict):
            return info.get(key)
        return getattr(info, key)
    except Exception:
        return None
