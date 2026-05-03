from datetime import UTC, datetime
from uuid import UUID, uuid4
from typing import Optional

from sqlalchemy import Boolean, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class EventRecord(Base):
    __tablename__ = "event_records"
    __table_args__ = (UniqueConstraint("article_hash", name="uq_event_article_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    article_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    affected_sectors: Mapped[str] = mapped_column(Text, nullable=False)
    impact_direction: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    mapped_assets: Mapped[str] = mapped_column(Text, nullable=False)
    assets_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    price_at_event: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    price_at_evaluation: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=None)
    signal_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True, default=None)
    evaluated_at: Mapped[Optional[datetime]] = mapped_column(nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class MarketSnapshot(Base):
    __tablename__ = "market_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    article_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


class SignalBacktest(Base):
    __tablename__ = "signal_backtests"
    __table_args__ = (
        UniqueConstraint("article_hash", "symbol", "horizon", name="uq_signal_backtest_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    article_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    horizon: Mapped[str] = mapped_column(String(16), index=True, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    affected_sectors: Mapped[str] = mapped_column(Text, nullable=False)
    expected_direction: Mapped[str] = mapped_column(String(32), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    return_pct: Mapped[float] = mapped_column(Float, nullable=False)
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    ml_score: Mapped[float] = mapped_column(Float, nullable=False)
    feature_vector: Mapped[str] = mapped_column(Text, nullable=False)
    observations: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    evaluated_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


class Event(Base):
    __tablename__ = "events"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    timestamp: Mapped[datetime] = mapped_column(nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    region: Mapped[str] = mapped_column(String(64), nullable=False, default="global")
    sentiment: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    severity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False, default="unknown")
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    text_embedding: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    # SHA-256 of "{timestamp}|{raw_text}|{source}" — unique dedup key
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, unique=True, index=True, default=None)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    event_id: Mapped[UUID] = mapped_column(ForeignKey("events.id"), nullable=False, index=True)
    asset: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    predicted_direction: Mapped[str] = mapped_column(String(8), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    horizon: Mapped[str] = mapped_column(String(16), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC), nullable=False)

    event: Mapped[Event] = relationship(back_populates="predictions")
    feature_snapshot: Mapped[Optional["FeatureSnapshot"]] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
        uselist=False,
    )
    outcome: Mapped[Optional["Outcome"]] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
        uselist=False,
    )
    multi_horizon_outcomes: Mapped[list["MultiHorizonOutcome"]] = relationship(
        back_populates="prediction",
        cascade="all, delete-orphan",
    )


class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    prediction_id: Mapped[UUID] = mapped_column(
        ForeignKey("predictions.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    event_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    market_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    derived_features: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    prediction: Mapped[Prediction] = relationship(back_populates="feature_snapshot")


class Outcome(Base):
    __tablename__ = "outcomes"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    prediction_id: Mapped[UUID] = mapped_column(
        ForeignKey("predictions.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    actual_return: Mapped[float] = mapped_column(Float, nullable=False)
    raw_return: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    return_magnitude: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    label: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    filtered_label: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False, default=0.002)
    computed_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    prediction: Mapped[Prediction] = relationship(back_populates="outcome")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    trained_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(UTC), nullable=False, index=True
    )
    triggered_by: Mapped[str] = mapped_column(String(64), nullable=False, default="manual")
    dataset_size: Mapped[int] = mapped_column(Integer, nullable=False)
    train_size: Mapped[int] = mapped_column(Integer, nullable=False)
    test_size: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    brier_score_calibrated: Mapped[float] = mapped_column(Float, nullable=False)
    brier_improvement: Mapped[float] = mapped_column(Float, nullable=False)
    walk_forward_mean: Mapped[float] = mapped_column(Float, nullable=False)
    walk_forward_std: Mapped[float] = mapped_column(Float, nullable=False)
    walk_forward_folds: Mapped[int] = mapped_column(Integer, nullable=False)
    xgboost_roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    logistic_roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winner_model: Mapped[str] = mapped_column(String(64), nullable=False)
    top_features: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    label_balance: Mapped[str] = mapped_column(Text, nullable=False, default="{}")


class MultiHorizonOutcome(Base):
    __tablename__ = "multi_horizon_outcomes"
    __table_args__ = (
        UniqueConstraint("prediction_id", "horizon_days", name="uq_mho_pred_horizon"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    prediction_id: Mapped[UUID] = mapped_column(
        ForeignKey("predictions.id"),
        nullable=False,
        index=True,
    )
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    raw_return: Mapped[float] = mapped_column(Float, nullable=False)
    return_magnitude: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False, default=0.002)
    computed_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    prediction: Mapped[Prediction] = relationship(back_populates="multi_horizon_outcomes")
