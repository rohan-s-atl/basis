from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from datetime import date, datetime, timedelta

import yfinance as yf


@dataclass(frozen=True)
class HistoricalPriceWindow:
    symbol: str
    entry_date: date
    entry_price: float
    pre_event_history: list[float]
    future_closes: dict[int, float]


@dataclass(frozen=True)
class HistoricalRegimeSnapshot:
    vix_level: float
    vix_regime_encoded: int
    spy_trend: float
    rate_level: float
    market_regime_encoded: int


HORIZONS = (1, 3, 5)


def fetch_price_window(
    symbol: str,
    event_timestamp: datetime,
    *,
    lookback_days: int = 30,
    forward_days: int = 10,
) -> HistoricalPriceWindow:
    normalized_symbol = symbol.upper()
    event_date = event_timestamp.date()
    start = event_date - timedelta(days=lookback_days)
    end = event_date + timedelta(days=forward_days)
    closes = _fetch_daily_closes(normalized_symbol, start, end)

    trading_dates = sorted(closes)
    entry_index = next(
        (index for index, trading_date in enumerate(trading_dates) if trading_date >= event_date),
        None,
    )
    if entry_index is None:
        raise ValueError(f"No entry price found for {normalized_symbol} on or after {event_date}")

    entry_date = trading_dates[entry_index]
    entry_price = closes[entry_date]
    pre_event_history = [closes[trading_date] for trading_date in trading_dates[: entry_index + 1]]
    future_closes: dict[int, float] = {}
    for horizon in HORIZONS:
        future_index = entry_index + horizon
        if future_index < len(trading_dates):
            future_closes[horizon] = closes[trading_dates[future_index]]

    return HistoricalPriceWindow(
        symbol=normalized_symbol,
        entry_date=entry_date,
        entry_price=entry_price,
        pre_event_history=pre_event_history,
        future_closes=future_closes,
    )


def fetch_historical_regime(event_timestamp: datetime) -> HistoricalRegimeSnapshot:
    return _fetch_historical_regime_for_date(event_timestamp.date())


@lru_cache(maxsize=512)
def _fetch_historical_regime_for_date(event_date: date) -> HistoricalRegimeSnapshot:
    start = event_date - timedelta(days=45)
    end = event_date + timedelta(days=5)
    vix = _close_on_or_before("^VIX", event_date, start, end, default=20.0)
    tnx = _close_on_or_before("^TNX", event_date, start, end, default=4.0)
    spy_closes = _fetch_daily_closes("SPY", start, end)
    spy_history = [
        close
        for trading_date, close in sorted(spy_closes.items())
        if trading_date <= event_date
    ]
    spy_trend = _trend(spy_history, window=20)

    return HistoricalRegimeSnapshot(
        vix_level=round(vix, 4),
        vix_regime_encoded=_encode_vix(vix),
        spy_trend=round(spy_trend, 6),
        rate_level=round(tnx / 10.0, 4),
        market_regime_encoded=_encode_market_regime(vix, spy_trend),
    )


def _fetch_daily_closes(symbol: str, start: date, end: date) -> dict[date, float]:
    ticker = yf.Ticker(symbol)
    history = ticker.history(start=str(start), end=str(end + timedelta(days=1)), auto_adjust=False)
    if history.empty:
        raise ValueError(f"No historical market data found for symbol: {symbol}")

    close_prices = history["Close"].dropna()
    if close_prices.empty:
        raise ValueError(f"No close prices found for symbol: {symbol}")

    return {timestamp.date(): float(close) for timestamp, close in close_prices.items()}


def _close_on_or_before(
    symbol: str,
    target_date: date,
    start: date,
    end: date,
    *,
    default: float,
) -> float:
    try:
        closes = _fetch_daily_closes(symbol, start, end)
    except Exception:
        return default

    eligible = [
        close
        for trading_date, close in sorted(closes.items())
        if trading_date <= target_date
    ]
    return eligible[-1] if eligible else default


def _trend(history: list[float], *, window: int) -> float:
    if len(history) <= window:
        return 0.0
    start_price = history[-(window + 1)]
    end_price = history[-1]
    if start_price == 0:
        return 0.0
    return (end_price - start_price) / start_price


def _encode_vix(vix: float) -> int:
    if vix < 15:
        return 0
    if vix < 25:
        return 1
    if vix < 35:
        return 2
    return 3


def _encode_market_regime(vix: float, spy_trend: float) -> int:
    if vix >= 25 or spy_trend < -0.03:
        return 2
    if vix < 18 and spy_trend > 0:
        return 0
    return 1
