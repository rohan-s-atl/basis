import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import FeatureSnapshot, Outcome, Prediction
from app.db.session import SessionLocal
from app.services.market_service import fetch_price

logger = logging.getLogger(__name__)


def compute_outcomes(
    *,
    db: Session | None = None,
    limit: int = 500,
    noise_threshold: float | None = None,
) -> dict[str, int]:
    owns_session = db is None
    session = db or SessionLocal()
    threshold = settings.outcome_noise_threshold if noise_threshold is None else noise_threshold
    computed = 0
    skipped = 0
    filtered = 0

    try:
        predictions = _predictions_without_outcomes(session, limit=limit)
        for prediction in predictions:
            snapshot = prediction.feature_snapshot
            if snapshot is None:
                skipped += 1
                continue

            entry_price = _snapshot_entry_price(snapshot)
            if entry_price is None or entry_price <= 0:
                skipped += 1
                continue

            try:
                current_market = fetch_price(prediction.asset)
                current_price = float(current_market["price"])
            except Exception as exc:
                logger.warning("Outcome market fetch failed for %s: %s", prediction.asset, exc)
                skipped += 1
                continue

            actual_return = (current_price - entry_price) / entry_price
            label = _label_prediction(
                prediction.predicted_direction,
                actual_return,
                noise_threshold=threshold,
            )
            if label is None:
                filtered += 1
            session.add(
                Outcome(
                    prediction_id=prediction.id,
                    actual_return=round(actual_return, 8),
                    raw_return=round(actual_return, 8),
                    label=label,
                    filtered_label=label,
                    threshold_used=threshold,
                    computed_at=datetime.now(UTC),
                )
            )
            computed += 1

        session.commit()
        return {"computed": computed, "skipped": skipped, "filtered": filtered}
    except Exception:
        session.rollback()
        raise
    finally:
        if owns_session:
            session.close()


def _predictions_without_outcomes(db: Session, *, limit: int) -> list[Prediction]:
    return (
        db.query(Prediction)
        .outerjoin(Outcome, Outcome.prediction_id == Prediction.id)
        .filter(Outcome.id.is_(None))
        .order_by(Prediction.timestamp.asc())
        .limit(limit)
        .all()
    )


def _snapshot_entry_price(snapshot: FeatureSnapshot) -> float | None:
    market_features: dict[str, Any] = snapshot.market_features or {}
    try:
        return float(market_features.get("price"))
    except (TypeError, ValueError):
        return None


def _label_prediction(
    predicted_direction: str,
    actual_return: float,
    *,
    noise_threshold: float,
) -> int | None:
    if abs(actual_return) < noise_threshold:
        return None
    if predicted_direction == "up" and actual_return > 0:
        return 1
    if predicted_direction == "down" and actual_return < 0:
        return 1
    return 0
