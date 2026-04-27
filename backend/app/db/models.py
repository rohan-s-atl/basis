from datetime import UTC, datetime
from typing import Optional

from sqlalchemy import Boolean, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

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
