from sqlalchemy.orm import Session

from app.db.models import SignalBacktest


def upsert_signal_backtest(
    db: Session,
    *,
    article_hash: str,
    symbol: str,
    horizon: str,
    event_type: str,
    affected_sectors: str,
    expected_direction: str,
    entry_price: float,
    exit_price: float,
    return_pct: float,
    correct: bool,
    confidence: float,
    severity: str,
    ml_score: float,
    feature_vector: str,
) -> SignalBacktest:
    record = (
        db.query(SignalBacktest)
        .filter(
            SignalBacktest.article_hash == article_hash,
            SignalBacktest.symbol == symbol,
            SignalBacktest.horizon == horizon,
        )
        .one_or_none()
    )

    if record is None:
        record = SignalBacktest(
            article_hash=article_hash,
            symbol=symbol.upper(),
            horizon=horizon,
        )
        db.add(record)

    record.event_type = event_type
    record.affected_sectors = affected_sectors
    record.expected_direction = expected_direction
    record.entry_price = entry_price
    record.exit_price = exit_price
    record.return_pct = return_pct
    record.correct = correct
    record.confidence = confidence
    record.severity = severity
    record.ml_score = ml_score
    record.feature_vector = feature_vector

    db.commit()
    db.refresh(record)
    return record


def list_signal_backtests(db: Session, limit: int = 500) -> list[SignalBacktest]:
    return (
        db.query(SignalBacktest)
        .order_by(SignalBacktest.evaluated_at.desc())
        .limit(limit)
        .all()
    )
