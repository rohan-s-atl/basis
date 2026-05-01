from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import FeatureSnapshot, Outcome, Prediction
from app.db.session import Base
from app.services.outcome_service import compute_outcomes
from app.services.prediction_pipeline import run_prediction_pipeline
from app.services.training_data_service import export_training_dataset


def test_prediction_pipeline_stores_features_and_exports_labeled_dataset(monkeypatch) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db_session = TestingSessionLocal()

    article = {
        "title": "Inflation data lifts rate expectations",
        "description": "Markets price in tighter policy after inflation runs hot.",
        "publishedAt": "2026-04-27T00:00:00Z",
    }
    classification = {
        "event_type": "inflation",
        "affected_sectors": ["broad_market"],
        "impact_direction": "positive",
        "confidence": 0.8,
        "severity": "high",
        "reasoning": "Hot inflation can move macro assets.",
    }

    monkeypatch.setattr(
        "app.services.prediction_pipeline.fetch_price",
        lambda symbol: {"symbol": symbol, "price": 100.0, "history": [99.0, 100.0]},
    )

    try:
        predictions = run_prediction_pipeline(
            article,
            db=db_session,
            classification=classification,
            mapped_assets=["SPY"],
        )

        assert len(predictions) == 1
        prediction = db_session.query(Prediction).one()
        snapshot = db_session.query(FeatureSnapshot).one()
        assert prediction.asset == "SPY"
        assert prediction.predicted_direction == "up"
        assert snapshot.market_features["price"] == 100.0

        monkeypatch.setattr(
            "app.services.outcome_service.fetch_price",
            lambda symbol: {"symbol": symbol, "price": 110.0, "history": [100.0, 110.0]},
        )
        result = compute_outcomes(db=db_session)

        outcome = db_session.query(Outcome).one()
        dataset = export_training_dataset(db_session)
    finally:
        db_session.close()

    assert result == {"computed": 1, "skipped": 0}
    assert outcome.label == 1
    assert outcome.actual_return == 0.1
    assert dataset == [
        {
            "features": {
                "event_event_type": "inflation",
                "event_region": "global",
                "event_sentiment": 0.8,
                "event_severity": 0.8,
                "event_source": "news",
                "event_model_version": "gpt-4o-mini",
                "event_event_timestamp": "2026-04-27T00:00:00+00:00",
                "market_asset": "SPY",
                "market_price": 100.0,
                "market_volatility": 0.0,
                "market_history_points": 2,
                "derived_baseline_score": 0.72,
                "derived_horizon": "3d",
                "derived_sentiment_x_severity": 0.64,
                "derived_severity_x_volatility": 0.0,
                "derived_prediction_model_version": "baseline-rule-v1",
            },
            "label": 1,
        }
    ]
