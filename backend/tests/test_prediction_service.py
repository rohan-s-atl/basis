import json
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Event, EventRecord, FeatureSnapshot, Prediction
from app.db.session import Base
from app.services.prediction_service import generate_predictions


def test_generate_predictions_ranks_asset_signals() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()

    db.add(
        EventRecord(
            article_hash="pred123",
            title="Energy supply shock lifts oil risk",
            description="Oil markets react to constrained supply.",
            published_at="2026-04-27T00:00:00Z",
            event_type="supply_shock",
            affected_sectors=json.dumps(["energy", "commodities"]),
            impact_direction="positive",
            confidence=0.85,
            severity="high",
            reasoning="Supply constraints may lift energy assets.",
            mapped_assets=json.dumps(["XLE"]),
            assets_json=json.dumps([{"symbol": "XLE", "price": 100.0}]),
            price_at_event=100.0,
            created_at=datetime.now(UTC),
        )
    )
    db.commit()

    try:
        payload = generate_predictions(db)
    finally:
        db.close()

    assert payload["count"] == 1
    assert payload["total_considered"] == 1
    assert payload["weak_filtered"] == 0
    prediction = payload["predictions"][0]
    assert prediction["symbol"] == "XLE"
    assert prediction["impact_direction"] == "positive"
    assert prediction["expected_move_pct"] > 0
    assert prediction["probability"] > 0.5
    assert prediction["ranking_score"] > prediction["probability"] * 0.4
    assert prediction["is_actionable"] is True
    assert prediction["filter_reason"] is None
    assert prediction["drivers"]


def test_generate_predictions_applies_min_quality_to_stored_predictions(monkeypatch) -> None:
    monkeypatch.setattr("app.services.prediction_service.explain_prediction", lambda *args: [])

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()

    event = Event(
        timestamp=datetime.now(UTC),
        event_type="general_market",
        sentiment=0.1,
        severity=0.5,
        raw_text="Stored signal\nMarket update",
        source="test",
        model_version="test",
    )
    prediction = Prediction(
        event=event,
        asset="SPY",
        predicted_direction="up",
        confidence=0.6,
        horizon="1d",
        model_version="xgboost-v1",
        timestamp=datetime.now(UTC),
    )
    snapshot = FeatureSnapshot(
        prediction=prediction,
        event_features={},
        market_features={"rolling_volatility": 0.01},
        derived_features={"prediction_model_version": "xgboost-v1"},
    )
    db.add(snapshot)
    db.commit()

    try:
        default_payload = generate_predictions(db)
        relaxed_payload = generate_predictions(db, min_quality=0)
    finally:
        db.close()

    assert default_payload["count"] == 0
    assert default_payload["weak_filtered"] == 1
    assert relaxed_payload["count"] == 1
    assert relaxed_payload["weak_filtered"] == 0
    assert relaxed_payload["predictions"][0]["is_actionable"] is True


@pytest.mark.parametrize(
    ("asset", "event_type", "horizon", "expected_reason"),
    [
        ("GLD", "general_market", "1d", "gated_asset"),
        ("SPY", "interest_rate_change", "1d", "gated_event_type"),
        ("SPY", "general_market", "3d", "gated_horizon"),
    ],
)
def test_generate_predictions_gates_weak_live_segments(
    monkeypatch,
    asset: str,
    event_type: str,
    horizon: str,
    expected_reason: str,
) -> None:
    monkeypatch.setattr("app.services.prediction_service.explain_prediction", lambda *args: [])

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()

    event = Event(
        timestamp=datetime.now(UTC),
        event_type=event_type,
        sentiment=0.8,
        severity=0.8,
        raw_text="Weak segment signal\nMarket update",
        source="test",
        model_version="test",
    )
    prediction = Prediction(
        event=event,
        asset=asset,
        predicted_direction="up",
        confidence=0.9,
        horizon=horizon,
        model_version="xgboost-v1",
        timestamp=datetime.now(UTC),
    )
    snapshot = FeatureSnapshot(
        prediction=prediction,
        event_features={},
        market_features={"rolling_volatility": 0.02},
        derived_features={"prediction_model_version": "xgboost-v1"},
    )
    db.add(snapshot)
    db.commit()

    try:
        default_payload = generate_predictions(db)
        weak_payload = generate_predictions(db, include_weak=True)
    finally:
        db.close()

    assert default_payload["count"] == 0
    assert default_payload["weak_filtered"] == 1
    assert weak_payload["count"] == 1
    weak_prediction = weak_payload["predictions"][0]
    assert weak_prediction["is_actionable"] is False
    assert weak_prediction["filter_reason"] == expected_reason
