import json
import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.repositories.event_repository import list_unevaluated_events, update_signal_outcome
from app.services.market_service import fetch_price

logger = logging.getLogger(__name__)

MIN_AGE_HOURS = 24


def evaluate_signals(db: Session) -> int:
    """
    Find events older than MIN_AGE_HOURS whose direction hasn't been evaluated yet.
    Fetch current prices for their mapped assets, compare to price_at_event,
    and record whether the predicted direction was correct.
    Returns the number of signals evaluated.
    """
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=MIN_AGE_HOURS)
    pending = list_unevaluated_events(db, before=cutoff)

    if not pending:
        return 0

    count = 0
    for record in pending:
        try:
            mapped_assets = json.loads(record.mapped_assets or "[]")
        except Exception as exc:
            logger.warning("Skipping record %s: failed to parse mapped_assets: %s", record.id, exc)
            continue

        if not mapped_assets or record.price_at_event is None:
            continue

        prices: list[float] = []
        for symbol in mapped_assets[:4]:
            try:
                market_data = fetch_price(symbol)
                prices.append(float(market_data["price"]))
            except Exception as exc:
                logger.warning("Failed to fetch price for %s during signal evaluation: %s", symbol, exc)
                continue

        if not prices:
            continue

        current_avg = sum(prices) / len(prices)
        price_change = current_avg - record.price_at_event

        direction = record.impact_direction
        if direction == "positive":
            signal_correct = price_change > 0
        elif direction == "negative":
            signal_correct = price_change < 0
        else:
            continue

        update_signal_outcome(
            db,
            record_id=record.id,
            price_at_evaluation=current_avg,
            signal_correct=signal_correct,
        )
        count += 1
        logger.info(
            "Evaluated signal for '%s': predicted=%s, correct=%s (Δ%.2f)",
            record.title[:60],
            direction,
            signal_correct,
            price_change,
        )

    return count
