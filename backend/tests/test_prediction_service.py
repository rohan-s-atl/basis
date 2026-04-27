import json
from datetime import UTC, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import EventRecord
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
    prediction = payload["predictions"][0]
    assert prediction["symbol"] == "XLE"
    assert prediction["impact_direction"] == "positive"
    assert prediction["expected_move_pct"] > 0
    assert prediction["probability"] > 0.5
    assert prediction["drivers"]
