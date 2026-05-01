import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session, object_session

from app.core.config import settings
from app.db.models import Event, FeatureSnapshot, Prediction
from app.db.session import SessionLocal
from app.services.baseline_scoring import score_baseline_prediction
from app.services.event_classifier import classify_event
from app.services.feature_service import (
    build_derived_features,
    build_event_features,
    build_market_features,
    sentiment_from_classification,
    severity_to_score,
)
from app.services.mapping_service import map_article_to_assets
from app.services.market_service import fetch_price

BASELINE_MODEL_VERSION = "baseline-rule-v1"
logger = logging.getLogger(__name__)


def run_prediction_pipeline(
    event: dict[str, Any] | Event,
    *,
    db: Session | None = None,
    classification: dict[str, Any] | None = None,
    mapped_assets: Iterable[str] | None = None,
    model_version: str = BASELINE_MODEL_VERSION,
) -> list[Prediction]:
    owns_session = db is None
    session = db or SessionLocal()
    try:
        persisted_event, article, effective_classification = _persist_event(
            session,
            event,
            classification=classification,
            model_version=model_version,
        )
        symbols = list(mapped_assets or _assets_for_event(article, effective_classification))
        predictions = _store_predictions_for_assets(
            session,
            persisted_event,
            symbols=symbols,
            model_version=model_version,
        )
        session.commit()
        for prediction in predictions:
            session.refresh(prediction)
        return predictions
    except Exception:
        session.rollback()
        raise
    finally:
        if owns_session:
            session.close()


def prediction_horizon(event_type: str) -> str:
    if event_type in {"geopolitical_conflict", "supply_shock", "corporate_earnings"}:
        return "1d"
    if event_type in {"inflation", "interest_rate_change", "economic_data"}:
        return "3d"
    return "1d"


def _persist_event(
    db: Session,
    event: dict[str, Any] | Event,
    *,
    classification: dict[str, Any] | None,
    model_version: str,
) -> tuple[Event, dict[str, Any], dict[str, Any]]:
    if isinstance(event, Event):
        persisted_event = event if object_session(event) is db else db.merge(event)
        db.flush()
        return persisted_event, _article_from_event(persisted_event), _classification_from_event(persisted_event)

    article = dict(event)
    effective_classification = classification or classify_event(article)
    timestamp = _parse_timestamp(article.get("publishedAt") or article.get("timestamp"))
    raw_text = _raw_text(article)
    source = str(article.get("source") or article.get("url") or "news")

    persisted = (
        db.query(Event)
        .filter(
            Event.timestamp == timestamp,
            Event.raw_text == raw_text,
            Event.source == source,
        )
        .one_or_none()
    )
    if persisted is None:
        persisted = Event(
            timestamp=timestamp,
            raw_text=raw_text,
            source=source,
        )
        db.add(persisted)

    persisted.event_type = str(effective_classification["event_type"])
    persisted.region = str(article.get("region") or "global")
    persisted.sentiment = sentiment_from_classification(effective_classification)
    persisted.severity = severity_to_score(effective_classification.get("severity"))
    persisted.model_version = str(article.get("model_version") or settings.openai_model or model_version)

    db.flush()
    return persisted, article, effective_classification


def _store_predictions_for_assets(
    db: Session,
    event: Event,
    *,
    symbols: Iterable[str],
    model_version: str,
) -> list[Prediction]:
    stored: list[Prediction] = []
    event_features = build_event_features(event)
    horizon = prediction_horizon(event.event_type)

    for raw_symbol in symbols:
        symbol = str(raw_symbol).upper()
        if not symbol:
            continue
        if _prediction_exists(db, event, symbol, horizon):
            continue

        try:
            market_data = fetch_price(symbol)
        except Exception as exc:
            logger.warning("Skipping persisted prediction for %s: %s", symbol, exc)
            continue
        price = float(market_data["price"])
        history = [float(value) for value in market_data.get("history", [])]
        market_features = build_market_features(asset=symbol, price=price, history=history)
        baseline = score_baseline_prediction(
            sentiment=float(event_features["sentiment"]),
            severity=float(event_features["severity"]),
            volatility=float(market_features["volatility"]),
        )
        derived_features = build_derived_features(
            event_features=event_features,
            market_features=market_features,
            score=baseline.score,
            horizon=horizon,
        )

        prediction = Prediction(
            event_id=event.id,
            asset=symbol,
            predicted_direction=baseline.predicted_direction,
            confidence=baseline.confidence,
            horizon=horizon,
            timestamp=datetime.now(UTC),
        )
        db.add(prediction)
        db.flush()

        db.add(
            FeatureSnapshot(
                prediction_id=prediction.id,
                event_features=event_features,
                market_features=market_features,
                derived_features={
                    **derived_features,
                    "prediction_model_version": model_version,
                },
            )
        )
        stored.append(prediction)

    return stored


def _prediction_exists(db: Session, event: Event, symbol: str, horizon: str) -> bool:
    return (
        db.query(Prediction.id)
        .filter(
            Prediction.event_id == event.id,
            Prediction.asset == symbol,
            Prediction.horizon == horizon,
        )
        .first()
        is not None
    )


def _assets_for_event(article: dict[str, Any], classification: dict[str, Any]) -> list[str]:
    return map_article_to_assets(article, classification)


def _raw_text(article: dict[str, Any]) -> str:
    title = str(article.get("title") or "").strip()
    description = str(article.get("description") or "").strip()
    return f"{title}\n\n{description}".strip()


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(UTC)


def _article_from_event(event: Event) -> dict[str, Any]:
    return {
        "title": event.raw_text.splitlines()[0] if event.raw_text else "",
        "description": event.raw_text,
        "publishedAt": event.timestamp.isoformat(),
        "region": event.region,
        "source": event.source,
        "model_version": event.model_version,
    }


def _classification_from_event(event: Event) -> dict[str, Any]:
    if event.sentiment > 0:
        impact_direction = "positive"
    elif event.sentiment < 0:
        impact_direction = "negative"
    else:
        impact_direction = "neutral"

    return {
        "event_type": event.event_type,
        "affected_sectors": [],
        "impact_direction": impact_direction,
        "confidence": min(abs(event.sentiment), 1.0),
        "severity": event.severity,
        "reasoning": "persisted event",
    }
