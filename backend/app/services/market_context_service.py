from __future__ import annotations

import math
from dataclasses import dataclass

import yfinance as yf


@dataclass(frozen=True)
class MarketContext:
    symbol: str
    price: float
    close_history: list[float]
    return_20d: float
    return_60d: float
    rolling_volatility_20d: float
    beta_to_spy: float
    volume_ratio: float
    spy_return_20d: float
    dollar_trend_20d: float
    oil_trend_20d: float


def fetch_market_context(symbol: str, *, period: str = "6mo") -> MarketContext:
    normalized = symbol.upper()
    asset_history = _ticker_history(normalized, period=period)
    spy_history = _ticker_history("SPY", period=period)
    dollar_history = _ticker_history("UUP", period=period, required=False)
    oil_history = _ticker_history("USO", period=period, required=False)

    closes = asset_history["closes"]
    if not closes:
        raise ValueError(f"No market context found for {normalized}")

    return MarketContext(
        symbol=normalized,
        price=closes[-1],
        close_history=closes,
        return_20d=_return_over_period(closes, 20),
        return_60d=_return_over_period(closes, 60),
        rolling_volatility_20d=_rolling_volatility(closes, window=20),
        beta_to_spy=_beta(_daily_returns(closes), _daily_returns(spy_history["closes"])),
        volume_ratio=_volume_ratio(asset_history["volumes"], window=20),
        spy_return_20d=_return_over_period(spy_history["closes"], 20),
        dollar_trend_20d=_return_over_period(dollar_history["closes"], 20),
        oil_trend_20d=_return_over_period(oil_history["closes"], 20),
    )


def _ticker_history(symbol: str, *, period: str, required: bool = True) -> dict[str, list[float]]:
    try:
        history = yf.Ticker(symbol).history(period=period, auto_adjust=False)
        if history.empty:
            raise ValueError(f"No history for {symbol}")
        closes = [float(value) for value in history["Close"].dropna().tolist()]
        volumes = [float(value) for value in history["Volume"].dropna().tolist()] if "Volume" in history else []
        return {"closes": closes, "volumes": volumes}
    except Exception:
        if required:
            raise
        return {"closes": [], "volumes": []}


def _daily_returns(values: list[float]) -> list[float]:
    returns: list[float] = []
    for previous, current in zip(values, values[1:]):
        if previous:
            returns.append((current - previous) / previous)
    return returns


def _return_over_period(values: list[float], periods: int) -> float:
    if len(values) <= periods:
        return 0.0
    start = values[-(periods + 1)]
    if start == 0:
        return 0.0
    return (values[-1] - start) / start


def _rolling_volatility(values: list[float], *, window: int) -> float:
    returns = _daily_returns(values[-(window + 1):])
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    return math.sqrt(variance)


def _beta(asset_returns: list[float], benchmark_returns: list[float]) -> float:
    n = min(len(asset_returns), len(benchmark_returns), 60)
    if n < 10:
        return 1.0
    asset = asset_returns[-n:]
    benchmark = benchmark_returns[-n:]
    benchmark_mean = sum(benchmark) / n
    asset_mean = sum(asset) / n
    variance = sum((value - benchmark_mean) ** 2 for value in benchmark)
    if variance == 0:
        return 1.0
    covariance = sum((a - asset_mean) * (b - benchmark_mean) for a, b in zip(asset, benchmark))
    return covariance / variance


def _volume_ratio(volumes: list[float], *, window: int) -> float:
    if len(volumes) <= window:
        return 1.0
    recent = volumes[-1]
    average = sum(volumes[-(window + 1):-1]) / window
    if average <= 0:
        return 1.0
    return recent / average
