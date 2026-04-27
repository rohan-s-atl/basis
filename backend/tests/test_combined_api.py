from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_db
from app.main import app


def test_combined_endpoint_merges_classification_and_assets(monkeypatch) -> None:
    def fake_ingest_events(db) -> list[dict[str, object]]:
        return [
            {
                "title": "Inflation cools in latest economic data",
                "description": "Markets rally as prices moderate.",
                "event_type": "inflation",
                "affected_sectors": ["broad_market"],
                "impact_direction": "positive",
                "confidence": 0.82,
                "severity": "medium",
                "reasoning": "Cooling inflation can support risk assets.",
                "assets": [{"symbol": "SPY", "price": 100.0}],
            },
        ]

    monkeypatch.setattr("app.api.combined.ingest_events", fake_ingest_events)

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app)
    response = client.get("/combined")
    app.dependency_overrides.clear()

    assert response.status_code == 200

    payload = response.json()
    assert payload[0]["event_type"] == "inflation"
    assert payload[0]["affected_sectors"] == ["broad_market"]
    assert payload[0]["impact_direction"] == "positive"
    assert payload[0]["confidence"] == 0.82
    assert payload[0]["severity"] == "medium"
    assert payload[0]["assets"]
    assert {"symbol": "SPY", "price": 100.0} in payload[0]["assets"]
