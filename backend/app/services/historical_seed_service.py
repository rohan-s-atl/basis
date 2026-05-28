from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Event, FeatureSnapshot, MultiHorizonOutcome, Prediction
from app.db.session import SessionLocal
from app.services.baseline_scoring import score_baseline_prediction
from app.services.feature_service import (
    build_derived_features,
    build_event_features,
    build_market_features,
    sector_sensitivity_for_asset,
)
from app.services.historical_market_service import HORIZONS, fetch_price_window
from app.services.historical_market_service import fetch_historical_regime
from app.services.outcome_label_service import benchmark_metrics, market_direction_label, return_bucket


@dataclass(frozen=True)
class HistoricalSeedResult:
    events_inserted: int
    events_reused: int
    predictions_inserted: int
    outcomes_inserted: int
    skipped_rows: int
    errors: list[str]


def seed_historical_training_data_from_csv(
    file_path: str | Path,
    *,
    db: Session | None = None,
    noise_threshold: float | None = None,
    refresh_existing: bool = False,
) -> HistoricalSeedResult:
    rows = _read_rows(file_path)
    return seed_historical_training_data(
        rows,
        db=db,
        noise_threshold=noise_threshold,
        refresh_existing=refresh_existing,
    )


def seed_historical_training_data(
    rows: list[dict[str, str]],
    *,
    db: Session | None = None,
    noise_threshold: float | None = None,
    refresh_existing: bool = False,
) -> HistoricalSeedResult:
    owns_session = db is None
    session = db or SessionLocal()
    threshold = settings.outcome_noise_threshold if noise_threshold is None else noise_threshold
    events_inserted = 0
    events_reused = 0
    predictions_inserted = 0
    outcomes_inserted = 0
    skipped_rows = 0
    errors: list[str] = []

    try:
        for row_number, row in enumerate(rows, start=2):
            try:
                normalized = _normalize_row(row)
                event, inserted = _get_or_create_event(
                    session,
                    normalized,
                    refresh_existing=refresh_existing,
                )
                events_inserted += int(inserted)
                events_reused += int(not inserted)

                event_features = build_event_features(event)
                regime = fetch_historical_regime(normalized["timestamp"])
                benchmark = _benchmark_window(normalized["timestamp"])
                for symbol in normalized["mapped_assets"]:
                    window = fetch_price_window(symbol, normalized["timestamp"])
                    market_features = build_market_features(
                        asset=symbol,
                        price=window.entry_price,
                        history=window.pre_event_history,
                        spy_return_20d=regime.spy_trend,
                    )
                    baseline = score_baseline_prediction(
                        sentiment=float(event_features["sentiment"]),
                        severity=float(event_features["severity"]),
                        volatility=float(market_features["rolling_volatility"]),
                    )
                    derived_features = build_derived_features(
                        event_features=event_features,
                        market_features=market_features,
                        score=baseline.score,
                        horizon="1d",
                        sector_sensitivity=sector_sensitivity_for_asset(
                            symbol,
                            normalized["affected_sectors"],
                        ),
                        historical_accuracy_of_event_type=0.5,
                        rolling_accuracy_of_asset_predictions=0.5,
                        event_asset_avg_return=0.0,
                        event_asset_accuracy=0.5,
                        vix_level=regime.vix_level,
                        vix_regime_encoded=regime.vix_regime_encoded,
                        spy_trend=regime.spy_trend,
                        rate_level=regime.rate_level,
                        market_regime_encoded=regime.market_regime_encoded,
                    )
                    if benchmark is not None:
                        derived_features["benchmark_price"] = benchmark.entry_price

                    prediction, prediction_inserted = _get_or_create_prediction(
                        session,
                        event=event,
                        symbol=symbol,
                        predicted_direction=_direction_from_impact(
                            normalized["impact_direction"],
                            baseline.predicted_direction,
                        ),
                        confidence=normalized["confidence"],
                        timestamp=normalized["timestamp"],
                    )
                    predictions_inserted += int(prediction_inserted)
                    _get_or_create_snapshot(
                        session,
                        prediction=prediction,
                        event_features=event_features,
                        market_features=market_features,
                        derived_features={
                            **derived_features,
                            "prediction_model_version": "historical_seed",
                        },
                        refresh_existing=refresh_existing,
                    )
                    outcomes_inserted += _create_missing_outcomes(
                        session,
                        prediction=prediction,
                        entry_price=window.entry_price,
                        future_closes=window.future_closes,
                        benchmark_future_closes=benchmark.future_closes if benchmark is not None else {},
                        benchmark_entry_price=benchmark.entry_price if benchmark is not None else None,
                        threshold=threshold,
                        refresh_existing=refresh_existing,
                    )
            except Exception as exc:
                skipped_rows += 1
                errors.append(f"row {row_number}: {exc}")

        session.commit()
        return HistoricalSeedResult(
            events_inserted=events_inserted,
            events_reused=events_reused,
            predictions_inserted=predictions_inserted,
            outcomes_inserted=outcomes_inserted,
            skipped_rows=skipped_rows,
            errors=errors,
        )
    except Exception:
        session.rollback()
        raise
    finally:
        if owns_session:
            session.close()


def _read_rows(file_path: str | Path) -> list[dict[str, str]]:
    with Path(file_path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_row(row: dict[str, str]) -> dict[str, Any]:
    timestamp = _parse_timestamp(_required(row, "timestamp"))
    title = _required(row, "title")
    description = _required(row, "description")
    return {
        "timestamp": timestamp,
        "title": title,
        "description": description,
        "raw_text": f"{title}\n\n{description}",
        "event_type": _required(row, "event_type"),
        "impact_direction": _required(row, "impact_direction").lower(),
        "severity": _severity_to_score(_required(row, "severity")),
        "confidence": float(_required(row, "confidence")),
        "affected_sectors": _split_list(_required(row, "affected_sectors"), upper=False),
        "mapped_assets": _split_list(_required(row, "mapped_assets"), upper=True),
    }


def _get_or_create_event(
    db: Session,
    row: dict[str, Any],
    *,
    refresh_existing: bool,
) -> tuple[Event, bool]:
    matches = (
        db.query(Event)
        .filter(
            Event.raw_text == row["raw_text"],
            Event.source == "historical_seed",
        )
        .order_by(Event.created_at.asc())
        .all()
    )
    event = matches[0] if matches else None
    if event is not None:
        if refresh_existing and len(matches) > 1:
            for duplicate in matches[1:]:
                db.delete(duplicate)
            db.flush()
        event.timestamp = row["timestamp"]
        event.event_type = row["event_type"]
        event.sentiment = _sentiment_from_impact(row["impact_direction"], row["confidence"])
        event.severity = row["severity"]
        return event, False

    event = Event(
        timestamp=row["timestamp"],
        event_type=row["event_type"],
        region="global",
        sentiment=_sentiment_from_impact(row["impact_direction"], row["confidence"]),
        severity=row["severity"],
        raw_text=row["raw_text"],
        source="historical_seed",
        model_version="historical_seed",
    )
    db.add(event)
    db.flush()
    return event, True


def _get_or_create_prediction(
    db: Session,
    *,
    event: Event,
    symbol: str,
    predicted_direction: str,
    confidence: float,
    timestamp: datetime,
) -> tuple[Prediction, bool]:
    normalized_symbol = symbol.upper()
    prediction = (
        db.query(Prediction)
        .filter(
            Prediction.event_id == event.id,
            Prediction.asset == normalized_symbol,
            Prediction.horizon == "multi",
        )
        .one_or_none()
    )
    if prediction is not None:
        prediction.predicted_direction = predicted_direction
        prediction.confidence = confidence
        prediction.timestamp = timestamp
        return prediction, False

    prediction = Prediction(
        event_id=event.id,
        asset=normalized_symbol,
        predicted_direction=predicted_direction,
        confidence=confidence,
        horizon="multi",
        model_version="historical_seed",
        timestamp=timestamp,
    )
    db.add(prediction)
    db.flush()
    return prediction, True


def _get_or_create_snapshot(
    db: Session,
    *,
    prediction: Prediction,
    event_features: dict[str, Any],
    market_features: dict[str, Any],
    derived_features: dict[str, Any],
    refresh_existing: bool,
) -> FeatureSnapshot:
    if prediction.feature_snapshot is not None:
        if refresh_existing:
            prediction.feature_snapshot.event_features = event_features
            prediction.feature_snapshot.market_features = market_features
            prediction.feature_snapshot.derived_features = derived_features
        return prediction.feature_snapshot

    snapshot = FeatureSnapshot(
        prediction_id=prediction.id,
        event_features=event_features,
        market_features=market_features,
        derived_features=derived_features,
    )
    db.add(snapshot)
    db.flush()
    return snapshot


def _create_missing_outcomes(
    db: Session,
    *,
    prediction: Prediction,
    entry_price: float,
    future_closes: dict[int, float],
    benchmark_future_closes: dict[int, float],
    benchmark_entry_price: float | None,
    threshold: float,
    refresh_existing: bool,
) -> int:
    if refresh_existing:
        db.query(MultiHorizonOutcome).filter(
            MultiHorizonOutcome.prediction_id == prediction.id
        ).delete()
        db.flush()

    existing = {
        int(horizon)
        for (horizon,) in (
            db.query(MultiHorizonOutcome.horizon_days)
            .filter(MultiHorizonOutcome.prediction_id == prediction.id)
            .all()
        )
    }
    inserted = 0
    for horizon in HORIZONS:
        if horizon in existing or horizon not in future_closes:
            continue
        exit_price = future_closes[horizon]
        raw_return = (exit_price - entry_price) / entry_price
        label = market_direction_label(
            raw_return,
            noise_threshold=threshold,
        )
        benchmark_return = _benchmark_return_for_horizon(
            benchmark_future_closes,
            horizon=horizon,
            benchmark_entry_price=benchmark_entry_price,
            asset=prediction.asset,
            raw_return=raw_return,
        )
        benchmark = benchmark_metrics(
            raw_return=raw_return,
            benchmark_return=benchmark_return,
            noise_threshold=threshold,
        )
        db.add(
            MultiHorizonOutcome(
                prediction_id=prediction.id,
                horizon_days=horizon,
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
            )
        )
        inserted += 1
    return inserted


def _required(row: dict[str, str], key: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"missing {key}")
    return value


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _split_list(value: str, *, upper: bool) -> list[str]:
    parts = [part.strip() for part in value.replace(",", ";").split(";") if part.strip()]
    if upper:
        return [part.upper() for part in parts]
    return [part.lower() for part in parts]


def _severity_to_score(value: str) -> float:
    scores = {"low": 0.25, "medium": 0.5, "high": 0.8, "critical": 1.0}
    return scores.get(value.lower(), 0.25)


def _sentiment_from_impact(impact_direction: str, confidence: float) -> float:
    if impact_direction == "positive":
        return round(confidence, 4)
    if impact_direction == "negative":
        return round(-confidence, 4)
    return 0.0


def _direction_from_impact(impact_direction: str, fallback: str) -> str:
    if impact_direction == "positive":
        return "up"
    if impact_direction == "negative":
        return "down"
    return fallback


def _benchmark_window(event_timestamp: datetime):
    try:
        return fetch_price_window("SPY", event_timestamp)
    except Exception:
        return None


def _benchmark_return_for_horizon(
    benchmark_future_closes: dict[int, float],
    *,
    horizon: int,
    benchmark_entry_price: float | None,
    asset: str,
    raw_return: float,
) -> float | None:
    if asset.upper() == "SPY":
        return raw_return
    exit_price = benchmark_future_closes.get(horizon)
    if exit_price is None or benchmark_entry_price is None or benchmark_entry_price <= 0:
        return None
    return (exit_price - benchmark_entry_price) / benchmark_entry_price
