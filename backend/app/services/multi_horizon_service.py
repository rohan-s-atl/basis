from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

import yfinance as yf
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import FeatureSnapshot, MultiHorizonOutcome, Prediction
from app.db.session import SessionLocal
from app.services.outcome_label_service import benchmark_metrics, directional_label, return_bucket

logger = logging.getLogger(__name__)

HORIZONS = [1, 3, 5]


def compute_multi_horizon_outcomes(
    *,
    db: Session | None = None,
    limit: int = 200,
    noise_threshold: float | None = None,
) -> dict[str, int]:
    owns_session = db is None
    session = db or SessionLocal()
    threshold = settings.outcome_noise_threshold if noise_threshold is None else noise_threshold
    computed = 0
    skipped = 0

    try:
        predictions = _predictions_missing_horizons(session, limit=limit)
        if not predictions:
            return {"computed": 0, "skipped": 0}

        # Group by symbol — one yfinance call per symbol covers all its predictions
        by_symbol: dict[str, list[Prediction]] = {}
        for pred in predictions:
            by_symbol.setdefault(pred.asset, []).append(pred)

        for symbol, symbol_preds in by_symbol.items():
            dates = [p.timestamp.date() for p in symbol_preds]
            start_date = min(dates)
            end_date = max(dates) + timedelta(days=10)

            try:
                hist = _fetch_history(symbol, start_date, end_date)
            except Exception as exc:
                logger.warning("History fetch failed for %s: %s", symbol, exc)
                skipped += len(symbol_preds) * len(HORIZONS)
                continue
            benchmark_hist = _fetch_benchmark_history(start_date, end_date)

            for pred in symbol_preds:
                snapshot = pred.feature_snapshot
                if snapshot is None:
                    skipped += len(HORIZONS)
                    continue

                entry_price = _entry_price(snapshot)
                if entry_price is None or entry_price <= 0:
                    skipped += len(HORIZONS)
                    continue

                existing = {
                    o.horizon_days
                    for o in session.query(MultiHorizonOutcome.horizon_days)
                    .filter(MultiHorizonOutcome.prediction_id == pred.id)
                    .all()
                }

                for horizon_days in HORIZONS:
                    if horizon_days in existing:
                        continue

                    target = pred.timestamp.date() + timedelta(days=horizon_days)
                    exit_price = _price_at(hist, target)
                    if exit_price is None:
                        skipped += 1
                        continue

                    raw_return = (exit_price - entry_price) / entry_price
                    label = directional_label(
                        pred.predicted_direction,
                        raw_return,
                        noise_threshold=threshold,
                    )
                    benchmark_return = _benchmark_return(
                        pred,
                        snapshot,
                        benchmark_hist,
                        target=target,
                        raw_return=raw_return,
                    )
                    benchmark = benchmark_metrics(
                        predicted_direction=pred.predicted_direction,
                        raw_return=raw_return,
                        benchmark_return=benchmark_return,
                        noise_threshold=threshold,
                    )

                    session.add(
                        MultiHorizonOutcome(
                            prediction_id=pred.id,
                            horizon_days=horizon_days,
                            entry_price=round(entry_price, 8),
                            exit_price=round(exit_price, 8),
                            raw_return=round(raw_return, 8),
                            return_magnitude=round(abs(raw_return), 8),
                            return_bucket=return_bucket(raw_return),
                            benchmark_return=benchmark["benchmark_return"],
                            excess_return=benchmark["excess_return"],
                            benchmark_label=benchmark["benchmark_label"],
                            label=label,
                            threshold_used=threshold,
                            computed_at=datetime.now(UTC),
                        )
                    )
                    computed += 1

        session.commit()
        return {"computed": computed, "skipped": skipped}
    except Exception:
        session.rollback()
        raise
    finally:
        if owns_session:
            session.close()


def _predictions_missing_horizons(db: Session, limit: int) -> list[Prediction]:
    # Only process predictions old enough that the longest horizon has closed.
    # Add 3 trading-day buffer to handle weekends/holidays.
    min_age_cutoff = datetime.now(UTC) - timedelta(days=max(HORIZONS) + 3)
    complete = (
        db.query(MultiHorizonOutcome.prediction_id)
        .group_by(MultiHorizonOutcome.prediction_id)
        .having(func.count(MultiHorizonOutcome.horizon_days) >= len(HORIZONS))
        .subquery()
    )
    return (
        db.query(Prediction)
        .filter(
            Prediction.id.notin_(complete),
            Prediction.timestamp <= min_age_cutoff,
        )
        .order_by(Prediction.timestamp.asc())
        .limit(limit)
        .all()
    )


def _fetch_history(symbol: str, start: date, end: date) -> dict[date, float]:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=str(start), end=str(end + timedelta(days=1)))
    if hist.empty:
        raise ValueError(f"No data for {symbol}")
    return {idx.date(): float(row["Close"]) for idx, row in hist.iterrows()}


def _fetch_benchmark_history(start: date, end: date) -> dict[date, float]:
    try:
        return _fetch_history("SPY", start, end)
    except Exception as exc:
        logger.debug("Benchmark history fetch failed: %s", exc)
        return {}


def _price_at(hist: dict[date, float], target: date) -> float | None:
    for offset in range(5):
        price = hist.get(target + timedelta(days=offset))
        if price is not None:
            return price
    return None


def _entry_price(snapshot: FeatureSnapshot) -> float | None:
    try:
        return float((snapshot.market_features or {}).get("price"))
    except (TypeError, ValueError):
        return None


def _benchmark_return(
    pred: Prediction,
    snapshot: FeatureSnapshot,
    benchmark_hist: dict[date, float],
    *,
    target: date,
    raw_return: float,
) -> float | None:
    if pred.asset.upper() == "SPY":
        return raw_return

    try:
        entry_price = float((snapshot.derived_features or {}).get("benchmark_price"))
    except (TypeError, ValueError):
        return None
    if entry_price <= 0:
        return None

    exit_price = _price_at(benchmark_hist, target)
    if exit_price is None:
        return None
    return (exit_price - entry_price) / entry_price
