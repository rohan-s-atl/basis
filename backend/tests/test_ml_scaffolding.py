from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import FeatureSnapshot, MultiHorizonOutcome, Outcome, Prediction
from app.db.session import Base
from app.services.embedding_service import most_similar_duplicate_index
from app.services.outcome_service import compute_outcomes
from app.services.prediction_pipeline import run_prediction_pipeline
from app.services.training_data_service import (
    export_training_dataset,
    get_confidence_analysis,
    get_dataset_stats,
    get_train_test_split,
    validate_dataset,
)


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
        "provider": "alpha_vantage",
        "providers": "alpha_vantage,marketaux",
        "provider_sentiment": "Bullish",
        "related": "SPY,AAPL",
        "source_count": 2,
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
    monkeypatch.setattr(
        "app.services.prediction_pipeline.score_with_ml_model",
        lambda event_features, market_features, derived_features, fallback: (fallback, "baseline-v1"),
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
        assert prediction.predicted_direction == "down"
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

    assert result == {"computed": 1, "skipped": 0, "filtered": 0}
    assert outcome.label == 1
    assert outcome.filtered_label == 0
    assert outcome.actual_return == 0.1
    assert outcome.return_magnitude == 0.1
    assert dataset["labels"] == [1]
    feature_map = dict(zip(dataset["feature_names"], dataset["features"][0]))
    assert feature_map["event_event_type_encoded"] == 2
    assert feature_map["event_news_provider_count"] == 2
    assert feature_map["event_news_source_count"] == 2
    assert feature_map["event_news_symbol_match"] == 1
    assert feature_map["event_news_provider_sentiment_score"] == 1.0
    assert feature_map["event_news_sentiment_alignment"] > 0
    assert feature_map["market_return_1d"] > 0
    assert feature_map["market_return_5d"] == 0.0
    assert feature_map["market_return_10d"] == 0.0
    assert feature_map["market_sector_return_5d"] == 0.0
    assert feature_map["market_sector_return_10d"] == 0.0
    assert feature_map["market_relative_strength_5d"] == 0.0
    assert "derived_sentiment_x_sector_sensitivity" in feature_map
    assert "derived_event_type_asset_class_interaction" in feature_map
    assert "derived_historical_accuracy_of_event_type" not in feature_map
    assert "derived_rolling_accuracy_of_asset_predictions" not in feature_map
    assert "derived_event_asset_avg_return" not in feature_map
    assert "derived_event_asset_accuracy" not in feature_map
    assert "derived_event_novelty" in feature_map
    assert "market_asset_momentum_20d" in feature_map
    assert "outcome_return_magnitude" not in feature_map
    assert all(isinstance(value, (int, float)) for value in dataset["features"][0])

    split = get_train_test_split(db_session)
    buckets = get_confidence_analysis(db_session)
    report = validate_dataset(db_session)
    stats = get_dataset_stats(db_session)

    assert split["train"]["labels"] == []
    assert split["test"]["labels"] == [1]
    assert sum(bucket["samples"] for bucket in buckets.values()) == 1
    assert report["num_samples"] == 1
    assert stats["num_samples"] == 1


def test_compute_outcomes_filters_noise_from_training_dataset(monkeypatch) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db_session = TestingSessionLocal()

    monkeypatch.setattr(
        "app.services.prediction_pipeline.fetch_price",
        lambda symbol: {"symbol": symbol, "price": 100.0, "history": [99.0, 100.0]},
    )

    try:
        run_prediction_pipeline(
            {
                "title": "Technology shares drift",
                "description": "The move is small and directionally weak.",
                "publishedAt": "2026-04-27T00:00:00Z",
            },
            db=db_session,
            classification={
                "event_type": "general_market",
                "affected_sectors": ["technology"],
                "impact_direction": "positive",
                "confidence": 0.6,
                "severity": "medium",
                "reasoning": "Small market move.",
            },
            mapped_assets=["QQQ"],
        )
        monkeypatch.setattr(
            "app.services.outcome_service.fetch_price",
            lambda symbol: {"symbol": symbol, "price": 100.1, "history": [100.0, 100.1]},
        )

        result = compute_outcomes(db=db_session, noise_threshold=0.002)
        dataset = export_training_dataset(db_session)
        pending = db_session.query(Prediction).outerjoin(Outcome, Outcome.prediction_id == Prediction.id).filter(Outcome.id.is_(None)).count()
    finally:
        db_session.close()

    assert result == {"computed": 1, "skipped": 0, "filtered": 1}
    assert pending == 1
    assert dataset["features"] == []
    assert dataset["labels"] == []
    assert dataset["feature_names"] == []
    assert dataset["metadata"]["num_samples"] == 0


def test_training_export_combines_labels_in_chronological_order(monkeypatch) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db_session = TestingSessionLocal()

    monkeypatch.setattr(
        "app.services.prediction_pipeline.fetch_price",
        lambda symbol: {"symbol": symbol, "price": 100.0, "history": [99.0, 100.0]},
    )

    base_time = datetime(2026, 4, 27, tzinfo=UTC)
    try:
        for index, symbol in enumerate(["SPY", "QQQ", "GLD"]):
            run_prediction_pipeline(
                {
                    "title": f"Macro event {index}",
                    "description": "Markets react to macro conditions.",
                    "publishedAt": "2026-04-27T00:00:00Z",
                },
                db=db_session,
                classification={
                    "event_type": "general_market",
                    "affected_sectors": ["broad_market"],
                    "impact_direction": "positive",
                    "confidence": 0.7,
                    "severity": "medium",
                    "reasoning": "Market moving event.",
                },
                mapped_assets=[symbol],
            )
            prediction = db_session.query(Prediction).filter(Prediction.asset == symbol).one()
            prediction.timestamp = base_time + timedelta(hours=index)
            if symbol == "QQQ":
                db_session.add(
                    MultiHorizonOutcome(
                        prediction_id=prediction.id,
                        horizon_days=3,
                        entry_price=100.0,
                        exit_price=102.0,
                        raw_return=0.02,
                        return_magnitude=0.02,
                        label=1,
                        threshold_used=0.0001,
                    )
                )
            else:
                db_session.add(
                    Outcome(
                        prediction_id=prediction.id,
                        actual_return=0.01,
                        raw_return=0.01,
                        return_magnitude=0.01,
                        label=index % 2,
                        filtered_label=index % 2,
                        threshold_used=0.0001,
                    )
                )
        db_session.commit()
        dataset = export_training_dataset(db_session)
        split = get_train_test_split(db_session)
    finally:
        db_session.close()

    assert dataset["metadata"]["num_samples"] == 3
    assert dataset["labels"] == [0, 1, 0]
    assert "derived_horizon_days" in dataset["feature_names"]
    assert split["train"]["labels"] == [0, 1]
    assert split["test"]["labels"] == [0]


def test_most_similar_duplicate_index_returns_matching_candidate() -> None:
    assert most_similar_duplicate_index(
        [1.0, 0.0],
        [[0.0, 1.0], [0.99, 0.01]],
    ) == 1
