from datetime import UTC, datetime

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Event, FeatureSnapshot, MultiHorizonOutcome, Outcome, Prediction
from app.db.session import Base
from app.services.historical_market_service import HistoricalPriceWindow, HistoricalRegimeSnapshot
from app.services.historical_seed_service import seed_historical_training_data
from app.services.model_evaluation_service import get_model_evaluation
from app.services.prediction_pipeline import (
    _event_asset_accuracy,
    _event_asset_avg_return,
    _historical_accuracy_of_event_type,
    _rolling_accuracy_of_asset_predictions,
)
from app.services.training_data_service import export_training_dataset, get_confidence_analysis
from app.services.training_run_service import get_model_health


def _session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal()


def test_export_training_dataset_is_chronological() -> None:
    db_session = _session()
    try:
        newer = _prediction_with_outcome(
            db_session,
            timestamp=datetime(2024, 1, 2, tzinfo=UTC),
            label=1,
        )
        older = _prediction_with_outcome(
            db_session,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            label=0,
        )
        db_session.commit()

        dataset = export_training_dataset(db_session)
    finally:
        db_session.close()

    assert newer.timestamp > older.timestamp
    assert dataset["labels"] == [0, 1]


def test_historical_seed_creates_idempotent_multi_horizon_training_rows(monkeypatch) -> None:
    db_session = _session()

    def fake_window(symbol, event_timestamp):
        return HistoricalPriceWindow(
            symbol=symbol.upper(),
            entry_date=event_timestamp.date(),
            entry_price=100.0,
            pre_event_history=[98.0, 99.0, 100.0],
            future_closes={1: 99.0, 3: 101.0, 5: 100.1},
        )

    monkeypatch.setattr(
        "app.services.historical_seed_service.fetch_price_window",
        fake_window,
    )
    monkeypatch.setattr(
        "app.services.historical_seed_service.fetch_historical_regime",
        lambda timestamp: HistoricalRegimeSnapshot(
            vix_level=18.5,
            vix_regime_encoded=1,
            spy_trend=0.02,
            rate_level=4.25,
            market_regime_encoded=0,
        ),
    )
    rows = [
        {
            "timestamp": "2024-01-01T08:30:00Z",
            "title": "Inflation runs hot",
            "description": "A hot CPI release pressured risk assets.",
            "event_type": "inflation",
            "impact_direction": "negative",
            "severity": "high",
            "confidence": "0.85",
            "affected_sectors": "broad_market;technology;rates",
            "mapped_assets": "SPY",
        }
    ]

    try:
        first = seed_historical_training_data(rows, db=db_session, noise_threshold=0.002)
        second = seed_historical_training_data(rows, db=db_session, noise_threshold=0.002)
        outcomes = (
            db_session.query(MultiHorizonOutcome)
            .order_by(MultiHorizonOutcome.horizon_days.asc())
            .all()
        )
        dataset = export_training_dataset(db_session)
        snapshot = db_session.query(FeatureSnapshot).one()
    finally:
        db_session.close()

    assert first.events_inserted == 1
    assert first.predictions_inserted == 1
    assert first.outcomes_inserted == 3
    assert first.skipped_rows == 0
    assert second.events_inserted == 0
    assert second.events_reused == 1
    assert second.predictions_inserted == 0
    assert second.outcomes_inserted == 0
    assert [outcome.label for outcome in outcomes] == [1, 0, None]
    assert dataset["labels"] == [1, 0]
    assert "derived_horizon_days" in dataset["feature_names"]
    assert snapshot.derived_features["vix_level"] == 18.5
    assert snapshot.derived_features["market_regime_encoded"] == 0


def test_training_metrics_combine_multi_horizon_and_single_horizon_rows() -> None:
    db_session = _session()
    try:
        single = _prediction_with_outcome(
            db_session,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            label=1,
            confidence=0.55,
        )
        multi_prediction = _prediction_with_multi_outcomes(
            db_session,
            timestamp=datetime(2024, 1, 2, tzinfo=UTC),
            labels=[0, 1],
            confidence=0.85,
        )
        db_session.add(
            Outcome(
                prediction_id=multi_prediction.id,
                actual_return=0.02,
                raw_return=0.02,
                return_magnitude=0.02,
                label=1,
                filtered_label=1,
                threshold_used=0.002,
            )
        )
        db_session.commit()

        dataset = export_training_dataset(db_session)
        buckets = get_confidence_analysis(db_session)
        health = get_model_health(db_session)
        evaluation = get_model_evaluation(db_session)
    finally:
        db_session.close()

    assert single.timestamp < multi_prediction.timestamp
    assert dataset["labels"] == [1, 0, 1]
    assert "derived_horizon_days" in dataset["feature_names"]
    assert sum(bucket["samples"] for bucket in buckets.values()) == 3
    assert buckets["0.5-0.6"]["samples"] == 1
    assert buckets["0.8+"]["samples"] == 2
    assert health["rolling_accuracy"] == {"accuracy": 0.6667, "samples": 3, "window": 30}
    assert evaluation["overall"]["samples"] == 3
    assert evaluation["overall"]["accuracy"] == 0.6667
    assert evaluation["baselines"]["always_up"]["samples"] == 3
    assert evaluation["by_horizon"]["1"]["samples"] == 2
    assert evaluation["by_model_version"]["unknown"]["samples"] == 3


def test_live_history_features_read_multi_horizon_outcomes() -> None:
    db_session = _session()
    try:
        timestamp = datetime(2024, 1, 1, tzinfo=UTC)
        _prediction_with_multi_outcomes(
            db_session,
            timestamp=timestamp,
            labels=[1, 0],
            confidence=0.85,
            raw_returns=[0.03, -0.01],
        )
        db_session.commit()
        before = datetime(2024, 1, 3, tzinfo=UTC)

        event_type_accuracy = _historical_accuracy_of_event_type(
            db_session,
            "inflation",
            before=before,
        )
        rolling_asset_accuracy = _rolling_accuracy_of_asset_predictions(
            db_session,
            "SPY",
            before=before,
        )
        event_asset_accuracy = _event_asset_accuracy(
            db_session,
            "SPY",
            "inflation",
            before=before,
        )
        avg_return = _event_asset_avg_return(
            db_session,
            "SPY",
            "inflation",
            before=before,
        )
    finally:
        db_session.close()

    assert event_type_accuracy == 0.5
    assert rolling_asset_accuracy == 0.5
    assert event_asset_accuracy == 0.5
    assert avg_return == pytest.approx(0.01)


def _prediction_with_outcome(
    db_session,
    *,
    timestamp: datetime,
    label: int,
    confidence: float = 0.75,
) -> Prediction:
    event = Event(
        timestamp=timestamp,
        event_type="inflation",
        region="global",
        sentiment=0.5,
        severity=0.8,
        raw_text=f"event {timestamp.isoformat()}",
        source="test",
        model_version="test",
    )
    db_session.add(event)
    db_session.flush()

    prediction = Prediction(
        event_id=event.id,
        asset="SPY",
        predicted_direction="up",
        confidence=confidence,
        horizon="1d",
        timestamp=timestamp,
    )
    db_session.add(prediction)
    db_session.flush()

    db_session.add(
        FeatureSnapshot(
            prediction_id=prediction.id,
            event_features={"event_type_encoded": 2, "event_timestamp_unix": int(timestamp.timestamp())},
            market_features={"price": 100.0, "return_1d": 0.01},
            derived_features={"baseline_score": 0.7},
        )
    )
    db_session.add(
        Outcome(
            prediction_id=prediction.id,
            actual_return=0.01,
            raw_return=0.01,
            return_magnitude=0.01,
            label=label,
            filtered_label=label,
            threshold_used=0.002,
        )
    )
    return prediction


def _prediction_with_multi_outcomes(
    db_session,
    *,
    timestamp: datetime,
    labels: list[int],
    confidence: float = 0.75,
    raw_returns: list[float] | None = None,
) -> Prediction:
    event = Event(
        timestamp=timestamp,
        event_type="inflation",
        region="global",
        sentiment=0.5,
        severity=0.8,
        raw_text=f"multi event {timestamp.isoformat()}",
        source="test",
        model_version="test",
    )
    db_session.add(event)
    db_session.flush()

    prediction = Prediction(
        event_id=event.id,
        asset="SPY",
        predicted_direction="up",
        confidence=confidence,
        horizon="multi",
        timestamp=timestamp,
    )
    db_session.add(prediction)
    db_session.flush()

    db_session.add(
        FeatureSnapshot(
            prediction_id=prediction.id,
            event_features={"event_type_encoded": 2, "event_timestamp_unix": int(timestamp.timestamp())},
            market_features={"price": 100.0, "return_1d": 0.01},
            derived_features={"baseline_score": 0.7},
        )
    )
    returns = raw_returns or [0.01 for _ in labels]
    for index, label in enumerate(labels):
        raw_return = returns[index]
        db_session.add(
            MultiHorizonOutcome(
                prediction_id=prediction.id,
                horizon_days=index + 1,
                entry_price=100.0,
                exit_price=100.0 * (1.0 + raw_return),
                raw_return=raw_return,
                return_magnitude=abs(raw_return),
                label=label,
                threshold_used=0.002,
            )
        )
    return prediction
