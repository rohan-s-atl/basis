import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session, object_session

from app.core.config import settings
from app.db.models import Event, FeatureSnapshot, MultiHorizonOutcome, Outcome, Prediction
from app.db.session import SessionLocal
from app.services.baseline_scoring import score_baseline_prediction
from app.services.ml_scorer import score_with_ml_model
from app.services.event_classifier import classify_event
from app.services.feature_service import (
    average_return_over_histories,
    build_derived_features,
    build_event_features,
    build_market_features,
    sector_sensitivity_for_asset,
    sector_symbols_for_asset,
    sentiment_from_classification,
    severity_to_score,
)
from app.services.mapping_service import map_article_to_assets
from app.services.market_service import fetch_price

from app.services.baseline_scoring import BASELINE_MODEL_VERSION
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
            sectors=[str(sector) for sector in effective_classification.get("affected_sectors", [])],
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

    # Step 1: exact-match deduplication
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
        # Step 2: semantic deduplication via embedding similarity
        from datetime import timedelta
        import json as _json
        from app.services.embedding_service import embed_text, is_semantic_duplicate

        new_embedding = embed_text(raw_text)
        if new_embedding:
            cutoff = datetime.now(UTC) - timedelta(hours=48)
            recent = (
                db.query(Event)
                .filter(Event.created_at >= cutoff, Event.text_embedding.isnot(None))
                .all()
            )
            candidate_embeddings = []
            for candidate in recent:
                try:
                    candidate_embeddings.append(_json.loads(candidate.text_embedding))
                except Exception:
                    continue

            if is_semantic_duplicate(new_embedding, candidate_embeddings):
                logger.info("Semantic duplicate detected — skipping new event for: %.60s", raw_text)
                # Return the most recent matching event to reuse its predictions
                persisted = recent[-1] if recent else None

        if persisted is None:
            persisted = Event(
                timestamp=timestamp,
                raw_text=raw_text,
                source=source,
            )
            db.add(persisted)
            if new_embedding:
                persisted.text_embedding = _json.dumps(new_embedding)

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
    sectors: list[str],
    model_version: str,
) -> list[Prediction]:
    stored: list[Prediction] = []
    event_features = build_event_features(event)
    horizon = prediction_horizon(event.event_type)

    try:
        from app.services.regime_service import fetch_market_regime
        regime = fetch_market_regime()
    except Exception as exc:
        logger.warning("Regime fetch failed, using defaults: %s", exc)
        regime = {}
    benchmark_price = _current_benchmark_price()

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
        sector_return_5d, sector_return_10d = _sector_return_features(symbol, history)
        market_features = build_market_features(
            asset=symbol,
            price=price,
            history=history,
            sector_return_5d=sector_return_5d,
            sector_return_10d=sector_return_10d,
        )
        baseline = score_baseline_prediction(
            sentiment=float(event_features["sentiment"]),
            severity=float(event_features["severity"]),
            volatility=float(market_features["rolling_volatility"]),
        )
        prediction_timestamp = datetime.now(UTC)
        derived_features = build_derived_features(
            event_features=event_features,
            market_features=market_features,
            score=baseline.score,
            horizon=horizon,
            sector_sensitivity=sector_sensitivity_for_asset(symbol, sectors),
            historical_accuracy_of_event_type=_historical_accuracy_of_event_type(
                db,
                event.event_type,
                before=prediction_timestamp,
            ),
            rolling_accuracy_of_asset_predictions=_rolling_accuracy_of_asset_predictions(
                db,
                symbol,
                before=prediction_timestamp,
            ),
            event_asset_avg_return=_event_asset_avg_return(
                db,
                symbol,
                event.event_type,
                before=prediction_timestamp,
            ),
            event_asset_accuracy=_event_asset_accuracy(
                db,
                symbol,
                event.event_type,
                before=prediction_timestamp,
            ),
            vix_level=float(regime.get("vix_level", 20.0)),
            vix_regime_encoded=int(regime.get("vix_regime_encoded", 1)),
            spy_trend=float(regime.get("spy_trend", 0.0)),
            rate_level=float(regime.get("rate_level", 4.0)),
            market_regime_encoded=int(regime.get("market_regime_encoded", 1)),
        )
        if benchmark_price is not None:
            derived_features["benchmark_price"] = benchmark_price

        final_prediction, effective_model_version = score_with_ml_model(
            event_features,
            market_features,
            derived_features,
            fallback=baseline,
        )

        prediction = Prediction(
            event_id=event.id,
            asset=symbol,
            predicted_direction=final_prediction.predicted_direction,
            confidence=final_prediction.confidence,
            horizon=horizon,
            model_version=effective_model_version,
            timestamp=prediction_timestamp,
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
                    "prediction_model_version": effective_model_version,
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


def _current_benchmark_price() -> float | None:
    try:
        benchmark = fetch_price("SPY")
        return float(benchmark["price"])
    except Exception as exc:
        logger.debug("Benchmark entry price fetch failed: %s", exc)
        return None


def _historical_accuracy_of_event_type(db: Session, event_type: str, *, before: datetime) -> float:
    labels = _historical_labels(db, before=before, event_type=event_type)
    return _accuracy_or_prior(labels)


def _rolling_accuracy_of_asset_predictions(
    db: Session,
    asset: str,
    *,
    before: datetime,
    window: int = 50,
) -> float:
    labels = _historical_labels(db, before=before, asset=asset.upper(), limit=window)
    return _accuracy_or_prior(labels)


def _accuracy_or_prior(labels: list[int]) -> float:
    if not labels:
        return 0.5
    return sum(labels) / len(labels)


def _sector_return_features(asset: str, asset_history: list[float]) -> tuple[float, float]:
    histories: list[list[float]] = []
    for sector_symbol in sector_symbols_for_asset(asset):
        if sector_symbol == asset.upper():
            histories.append(asset_history)
            continue
        try:
            sector_market_data = fetch_price(sector_symbol)
            histories.append([float(value) for value in sector_market_data.get("history", [])])
        except Exception as exc:
            logger.debug("Sector return fetch failed for %s: %s", sector_symbol, exc)

    if not histories:
        histories = [asset_history]

    return (
        average_return_over_histories(histories, periods=5),
        average_return_over_histories(histories, periods=10),
    )


def _event_asset_avg_return(
    db: Session,
    asset: str,
    event_type: str,
    *,
    before: datetime,
) -> float:
    returns = _historical_returns(db, before=before, asset=asset.upper(), event_type=event_type)
    if returns:
        return sum(returns) / len(returns)
    return _global_avg_return(db, before=before)


def _event_asset_accuracy(
    db: Session,
    asset: str,
    event_type: str,
    *,
    before: datetime,
) -> float:
    labels = _historical_labels(db, before=before, asset=asset.upper(), event_type=event_type)
    if labels:
        return sum(labels) / len(labels)
    return _global_accuracy(db, before=before)


def _global_avg_return(db: Session, *, before: datetime) -> float:
    returns = _historical_returns(db, before=before)
    if not returns:
        return 0.0
    return sum(returns) / len(returns)


def _global_accuracy(db: Session, *, before: datetime) -> float:
    labels = _historical_labels(db, before=before)
    return _accuracy_or_prior(labels)


def _historical_labels(
    db: Session,
    *,
    before: datetime,
    asset: str | None = None,
    event_type: str | None = None,
    limit: int | None = None,
) -> list[int]:
    records = [
        (timestamp, int(label))
        for timestamp, label in _historical_metric_records(
            db,
            before=before,
            metric="label",
            asset=asset,
            event_type=event_type,
        )
    ]
    records.sort(key=lambda row: row[0], reverse=True)
    if limit is not None:
        records = records[:limit]
    return [label for _, label in records]


def _historical_returns(
    db: Session,
    *,
    before: datetime,
    asset: str | None = None,
    event_type: str | None = None,
) -> list[float]:
    return [
        float(value)
        for _, value in _historical_metric_records(
            db,
            before=before,
            metric="return",
            asset=asset,
            event_type=event_type,
        )
    ]


def _historical_metric_records(
    db: Session,
    *,
    before: datetime,
    metric: str,
    asset: str | None,
    event_type: str | None,
) -> list[tuple[datetime, int | float]]:
    multi_value = MultiHorizonOutcome.label if metric == "label" else MultiHorizonOutcome.raw_return
    single_value = Outcome.filtered_label if metric == "label" else Outcome.raw_return

    multi_query = (
        db.query(Prediction.timestamp, multi_value)
        .join(MultiHorizonOutcome, MultiHorizonOutcome.prediction_id == Prediction.id)
        .join(Event, Event.id == Prediction.event_id)
        .filter(
            Prediction.timestamp < before,
            MultiHorizonOutcome.label.isnot(None),
        )
    )
    single_query = (
        db.query(Prediction.timestamp, single_value)
        .join(Outcome, Outcome.prediction_id == Prediction.id)
        .join(Event, Event.id == Prediction.event_id)
        .filter(
            Prediction.timestamp < before,
            Outcome.filtered_label.isnot(None),
            ~Prediction.id.in_(_multi_labeled_prediction_ids(db)),
        )
    )

    if asset is not None:
        multi_query = multi_query.filter(Prediction.asset == asset.upper())
        single_query = single_query.filter(Prediction.asset == asset.upper())
    if event_type is not None:
        multi_query = multi_query.filter(Event.event_type == event_type)
        single_query = single_query.filter(Event.event_type == event_type)

    return [*multi_query.all(), *single_query.all()]


def _multi_labeled_prediction_ids(db: Session) -> list[Any]:
    return [
        prediction_id
        for (prediction_id,) in (
            db.query(MultiHorizonOutcome.prediction_id)
            .filter(MultiHorizonOutcome.label.isnot(None))
            .distinct()
            .all()
        )
    ]


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
