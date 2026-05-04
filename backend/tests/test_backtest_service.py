import json
from datetime import UTC, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import EventRecord
from app.db.session import Base
from app.services.backtest_service import get_portfolio_simulation, run_backtest


def test_run_backtest_builds_signal_outcome(monkeypatch) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db_session = TestingSessionLocal()

    event = EventRecord(
        article_hash="abc123",
        title="Oil prices jump on supply risk",
        description="Energy markets react to conflict.",
        published_at="2026-04-27T00:00:00Z",
        event_type="geopolitical_conflict",
        affected_sectors=json.dumps(["energy", "commodities"]),
        impact_direction="positive",
        confidence=0.82,
        severity="high",
        reasoning="Supply risk can lift energy prices.",
        mapped_assets=json.dumps(["XLE"]),
        assets_json=json.dumps([{"symbol": "XLE", "price": 100.0}]),
        price_at_event=100.0,
        created_at=datetime.now(UTC),
    )
    db_session.add(event)
    db_session.commit()

    monkeypatch.setattr(
        "app.services.backtest_service.fetch_price",
        lambda symbol: {"symbol": symbol, "price": 104.0, "history": [100.0, 104.0]},
    )

    try:
        summary = run_backtest(db_session)
        portfolio = get_portfolio_simulation(db_session)
    finally:
        db_session.close()

    assert summary["total_signals"] == 1
    assert summary["correct"] == 1
    assert summary["accuracy_pct"] == 100.0
    assert summary["top_signals"][0]["symbol"] == "XLE"
    assert summary["top_signals"][0]["return_pct"] == 4.0
    assert summary["top_signals"][0]["ml_score"] > 0.5
    assert portfolio["signals"] == 1
    assert portfolio["total_return_pct"] > 0
    assert len(portfolio["points"]) == 2
