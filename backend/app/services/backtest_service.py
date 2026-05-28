import json
import logging
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models import Event, MultiHorizonOutcome, Outcome, Prediction, SignalBacktest
from app.repositories.backtest_repository import list_signal_backtests, upsert_signal_backtest
from app.repositories.event_repository import list_recent_events
from app.services.feature_service import build_feature_vector, score_signal_quality
from app.services.market_service import fetch_price

logger = logging.getLogger(__name__)

_ACTIONABLE_RETURN_THRESHOLD_PCT = 0.05
_INITIAL_EQUITY = 10_000.0
_SIGNAL_ALLOCATION = 0.10


@dataclass(frozen=True)
class CanonicalBacktestRecord:
    symbol: str
    horizon: str
    event_type: str
    expected_direction: str
    entry_price: float
    exit_price: float
    return_pct: float
    benchmark_return_pct: float
    correct: bool
    confidence: float
    severity: str
    ml_score: float
    evaluated_at: object | None


def run_backtest(db: Session, limit: int = 100) -> dict:
    canonical = _canonical_backtest_records(db, limit=limit)
    if canonical:
        return _build_canonical_summary(canonical, skipped=0)

    evaluated: list[SignalBacktest] = []
    skipped = 0

    for event in list_recent_events(db, limit=limit):
        if event.impact_direction == "neutral":
            skipped += 1
            continue

        sectors = _parse_json_list(event.affected_sectors)
        assets = _parse_assets(event.assets_json, event.mapped_assets)

        if not assets:
            skipped += 1
            continue

        for asset in assets:
            symbol = str(asset.get("symbol", "")).upper()
            entry_price = _safe_float(asset.get("price")) or event.price_at_event
            if not symbol or not entry_price:
                skipped += 1
                continue

            try:
                current = fetch_price(symbol)
                exit_price = float(current["price"])
            except Exception as exc:
                logger.warning("Failed to fetch price for %s during backtest: %s", symbol, exc)
                skipped += 1
                continue

            return_pct = ((exit_price - float(entry_price)) / float(entry_price)) * 100
            correct = _is_signal_correct(event.impact_direction, return_pct)
            features = build_feature_vector(
                event,
                symbol=symbol,
                entry_price=float(entry_price),
                exit_price=exit_price,
                return_pct=return_pct,
                sectors=sectors,
            )
            ml_score = score_signal_quality(features)

            evaluated.append(
                upsert_signal_backtest(
                    db,
                    article_hash=event.article_hash,
                    symbol=symbol,
                    horizon="mark_to_market",
                    event_type=event.event_type,
                    affected_sectors=json.dumps(sectors),
                    expected_direction=event.impact_direction,
                    entry_price=float(entry_price),
                    exit_price=exit_price,
                    return_pct=return_pct,
                    correct=correct,
                    confidence=event.confidence,
                    severity=event.severity,
                    ml_score=ml_score,
                    feature_vector=json.dumps(features),
                )
            )

    return build_backtest_summary(list_signal_backtests(db), skipped=skipped)


def get_backtest_summary(db: Session) -> dict:
    canonical = _canonical_backtest_records(db)
    if canonical:
        return _build_canonical_summary(canonical, skipped=0)
    return build_backtest_summary(list_signal_backtests(db), skipped=0)


def get_portfolio_simulation(db: Session) -> dict:
    canonical = _canonical_backtest_records(db)
    if canonical:
        return _portfolio_from_canonical(canonical)
    records = list_signal_backtests(db)
    actionable_records = [record for record in records if _is_actionable(record)]
    ordered = sorted(
        actionable_records,
        key=lambda record: (
            record.evaluated_at is None,
            record.evaluated_at.isoformat() if record.evaluated_at else str(record.id),
        ),
    )
    all_ordered = sorted(
        records,
        key=lambda record: (
            record.evaluated_at is None,
            record.evaluated_at.isoformat() if record.evaluated_at else str(record.id),
        ),
    )

    equity = _INITIAL_EQUITY
    points = [{
        "index": 0,
        "label": "Start",
        "equity": round(equity, 2),
        "return_pct": 0.0,
        "benchmark_equity": round(_INITIAL_EQUITY, 2),
        "benchmark_return_pct": 0.0,
    }]

    for index, record in enumerate(ordered, start=1):
        signal_return = _signal_return_pct(record)
        equity *= 1.0 + (_SIGNAL_ALLOCATION * signal_return / 100.0)
        points.append({
            "index": index,
            "label": record.evaluated_at.isoformat() if record.evaluated_at else str(index),
            "equity": round(equity, 2),
            "return_pct": round(((equity / _INITIAL_EQUITY) - 1.0) * 100.0, 3),
            "benchmark_equity": round(_INITIAL_EQUITY, 2),
            "benchmark_return_pct": 0.0,
        })

    benchmark_points = _benchmark_curve(len(points))
    for point, benchmark in zip(points, benchmark_points):
        point["benchmark_equity"] = benchmark["equity"]
        point["benchmark_return_pct"] = benchmark["return_pct"]

    final_return = points[-1]["return_pct"] if points else 0.0
    benchmark_return = points[-1]["benchmark_return_pct"] if points else 0.0
    wins = sum(1 for record in ordered if _signal_return_pct(record) > 0)

    return {
        "initial_equity": round(_INITIAL_EQUITY, 2),
        "final_equity": points[-1]["equity"] if points else round(_INITIAL_EQUITY, 2),
        "total_return_pct": final_return,
        "benchmark_return_pct": benchmark_return,
        "excess_return_pct": round(final_return - benchmark_return, 3),
        "signals": len(all_ordered),
        "actionable_signals": len(ordered),
        "flat_signals": len(all_ordered) - len(ordered),
        "return_threshold_pct": _ACTIONABLE_RETURN_THRESHOLD_PCT,
        "win_rate_pct": round((wins / len(ordered)) * 100.0, 1) if ordered else 0.0,
        "allocation_pct": round(_SIGNAL_ALLOCATION * 100.0, 1),
        "points": points,
    }


def _canonical_backtest_records(db: Session, limit: int | None = None) -> list[CanonicalBacktestRecord]:
    multi_query = (
        db.query(Prediction, Event, MultiHorizonOutcome)
        .join(Event, Event.id == Prediction.event_id)
        .join(MultiHorizonOutcome, MultiHorizonOutcome.prediction_id == Prediction.id)
        .filter(MultiHorizonOutcome.benchmark_label.isnot(None) | MultiHorizonOutcome.label.isnot(None))
        .order_by(Prediction.timestamp.asc())
    )
    if limit is not None:
        multi_query = multi_query.limit(limit)
    multi_rows = [
        CanonicalBacktestRecord(
            symbol=prediction.asset,
            horizon=f"{outcome.horizon_days}d",
            event_type=event.event_type,
            expected_direction="positive" if prediction.predicted_direction == "up" else "negative",
            entry_price=outcome.entry_price,
            exit_price=outcome.exit_price,
            return_pct=round(outcome.raw_return * 100.0, 4),
            benchmark_return_pct=round(float(outcome.benchmark_return or 0.0) * 100.0, 4),
            correct=_direction_matches_label(prediction.predicted_direction, _target_label(outcome.benchmark_label, outcome.label)),
            confidence=prediction.confidence,
            severity=_severity_label(event.severity),
            ml_score=prediction.confidence,
            evaluated_at=outcome.computed_at,
        )
        for prediction, event, outcome in multi_query.all()
    ]
    if multi_rows:
        return multi_rows

    single_query = (
        db.query(Prediction, Event, Outcome)
        .join(Event, Event.id == Prediction.event_id)
        .join(Outcome, Outcome.prediction_id == Prediction.id)
        .filter(Outcome.benchmark_label.isnot(None) | Outcome.label.isnot(None))
        .order_by(Prediction.timestamp.asc())
    )
    if limit is not None:
        single_query = single_query.limit(limit)
    return [
        CanonicalBacktestRecord(
            symbol=prediction.asset,
            horizon=prediction.horizon,
            event_type=event.event_type,
            expected_direction="positive" if prediction.predicted_direction == "up" else "negative",
            entry_price=100.0,
            exit_price=100.0 * (1.0 + outcome.raw_return),
            return_pct=round(outcome.raw_return * 100.0, 4),
            benchmark_return_pct=round(float(outcome.benchmark_return or 0.0) * 100.0, 4),
            correct=_direction_matches_label(prediction.predicted_direction, _target_label(outcome.benchmark_label, outcome.label)),
            confidence=prediction.confidence,
            severity=_severity_label(event.severity),
            ml_score=prediction.confidence,
            evaluated_at=outcome.computed_at,
        )
        for prediction, event, outcome in single_query.all()
    ]


def _build_canonical_summary(records: list[CanonicalBacktestRecord], skipped: int = 0) -> dict:
    total = len(records)
    actionable = [record for record in records if abs(record.return_pct) >= _ACTIONABLE_RETURN_THRESHOLD_PCT]
    correct = sum(1 for record in actionable if record.correct)
    avg_return = sum(record.return_pct for record in records) / total if total else 0.0
    avg_score = sum(record.ml_score for record in records) / total if total else 0.0
    return {
        "total_signals": total,
        "actionable_signals": len(actionable),
        "flat_signals": total - len(actionable),
        "correct": correct,
        "accuracy_pct": round((correct / len(actionable)) * 100, 1) if actionable else 0.0,
        "avg_return_pct": round(avg_return, 3),
        "avg_ml_score": round(avg_score, 4),
        "skipped": skipped,
        "by_event_type": _canonical_group_accuracy(records, "event_type"),
        "by_symbol": _canonical_group_accuracy(records, "symbol"),
        "by_severity": _canonical_group_accuracy(records, "severity"),
        "top_signals": [_serialize_canonical(record) for record in sorted(records, key=lambda r: r.ml_score, reverse=True)[:10]],
        "recent": [_serialize_canonical(record) for record in records[-10:]],
    }


def _portfolio_from_canonical(records: list[CanonicalBacktestRecord]) -> dict:
    ordered = [record for record in records if abs(record.return_pct) >= _ACTIONABLE_RETURN_THRESHOLD_PCT]
    equity = _INITIAL_EQUITY
    benchmark_equity = _INITIAL_EQUITY
    points = [{
        "index": 0,
        "label": "Start",
        "equity": round(equity, 2),
        "return_pct": 0.0,
        "benchmark_equity": round(benchmark_equity, 2),
        "benchmark_return_pct": 0.0,
    }]
    for index, record in enumerate(ordered, start=1):
        signal_return = record.return_pct if record.expected_direction == "positive" else -record.return_pct
        equity *= 1.0 + (_SIGNAL_ALLOCATION * signal_return / 100.0)
        benchmark_equity *= 1.0 + (_SIGNAL_ALLOCATION * record.benchmark_return_pct / 100.0)
        points.append({
            "index": index,
            "label": record.evaluated_at.isoformat() if hasattr(record.evaluated_at, "isoformat") else str(index),
            "equity": round(equity, 2),
            "return_pct": round(((equity / _INITIAL_EQUITY) - 1.0) * 100.0, 3),
            "benchmark_equity": round(benchmark_equity, 2),
            "benchmark_return_pct": round(((benchmark_equity / _INITIAL_EQUITY) - 1.0) * 100.0, 3),
        })
    wins = sum(1 for record in ordered if record.correct)
    total_return = points[-1]["return_pct"] if points else 0.0
    benchmark_return = points[-1]["benchmark_return_pct"] if points else 0.0
    return {
        "initial_equity": round(_INITIAL_EQUITY, 2),
        "final_equity": points[-1]["equity"] if points else round(_INITIAL_EQUITY, 2),
        "total_return_pct": total_return,
        "benchmark_return_pct": benchmark_return,
        "excess_return_pct": round(total_return - benchmark_return, 3),
        "signals": len(records),
        "actionable_signals": len(ordered),
        "flat_signals": len(records) - len(ordered),
        "return_threshold_pct": _ACTIONABLE_RETURN_THRESHOLD_PCT,
        "win_rate_pct": round((wins / len(ordered)) * 100.0, 1) if ordered else 0.0,
        "allocation_pct": round(_SIGNAL_ALLOCATION * 100.0, 1),
        "points": points,
    }


def build_backtest_summary(records: list[SignalBacktest], skipped: int = 0) -> dict:
    total = len(records)
    actionable = [record for record in records if abs(record.return_pct) >= _ACTIONABLE_RETURN_THRESHOLD_PCT]
    flat = total - len(actionable)
    correct = sum(1 for record in actionable if record.correct)
    avg_return = sum(record.return_pct for record in records) / total if total else 0.0
    avg_ml_score = sum(record.ml_score for record in records) / total if total else 0.0

    by_event_type = _group_accuracy(records, "event_type")
    by_symbol = _group_accuracy(records, "symbol")
    by_severity = _group_accuracy(records, "severity")

    return {
        "total_signals": total,
        "actionable_signals": len(actionable),
        "flat_signals": flat,
        "correct": correct,
        "accuracy_pct": round((correct / len(actionable)) * 100, 1) if actionable else 0.0,
        "avg_return_pct": round(avg_return, 3),
        "avg_ml_score": round(avg_ml_score, 4),
        "skipped": skipped,
        "by_event_type": by_event_type,
        "by_symbol": by_symbol,
        "by_severity": by_severity,
        "top_signals": [_serialize_record(record) for record in sorted(records, key=lambda r: r.ml_score, reverse=True)[:10]],
        "recent": [_serialize_record(record) for record in records[:10]],
    }


def _group_accuracy(records: list[SignalBacktest], attr: str) -> dict[str, dict]:
    groups: dict[str, list[SignalBacktest]] = defaultdict(list)
    for record in records:
        groups[str(getattr(record, attr))].append(record)

    return {
        key: {
            "total": len(items),
            "actionable": len([item for item in items if abs(item.return_pct) >= _ACTIONABLE_RETURN_THRESHOLD_PCT]),
            "flat": len([item for item in items if abs(item.return_pct) < _ACTIONABLE_RETURN_THRESHOLD_PCT]),
            "correct": sum(
                1
                for item in items
                if abs(item.return_pct) >= _ACTIONABLE_RETURN_THRESHOLD_PCT and item.correct
            ),
            "accuracy_pct": _group_accuracy_pct(items),
            "avg_return_pct": round(sum(item.return_pct for item in items) / len(items), 3),
            "avg_ml_score": round(sum(item.ml_score for item in items) / len(items), 4),
        }
        for key, items in groups.items()
    }


def _canonical_group_accuracy(records: list[CanonicalBacktestRecord], attr: str) -> dict[str, dict]:
    groups: dict[str, list[CanonicalBacktestRecord]] = defaultdict(list)
    for record in records:
        groups[str(getattr(record, attr))].append(record)
    return {
        key: {
            "total": len(items),
            "actionable": len([item for item in items if abs(item.return_pct) >= _ACTIONABLE_RETURN_THRESHOLD_PCT]),
            "flat": len([item for item in items if abs(item.return_pct) < _ACTIONABLE_RETURN_THRESHOLD_PCT]),
            "correct": sum(
                1
                for item in items
                if abs(item.return_pct) >= _ACTIONABLE_RETURN_THRESHOLD_PCT and item.correct
            ),
            "accuracy_pct": _canonical_group_accuracy_pct(items),
            "avg_return_pct": round(sum(item.return_pct for item in items) / len(items), 3),
            "avg_ml_score": round(sum(item.ml_score for item in items) / len(items), 4),
        }
        for key, items in groups.items()
    }


def _serialize_record(record: SignalBacktest) -> dict:
    return {
        "symbol": record.symbol,
        "horizon": record.horizon,
        "event_type": record.event_type,
        "expected_direction": record.expected_direction,
        "entry_price": record.entry_price,
        "exit_price": record.exit_price,
        "return_pct": round(record.return_pct, 3),
        "correct": record.correct,
        "outcome_status": _outcome_status(record),
        "confidence": record.confidence,
        "severity": record.severity,
        "ml_score": record.ml_score,
        "evaluated_at": record.evaluated_at.isoformat() if record.evaluated_at else None,
    }


def _serialize_canonical(record: CanonicalBacktestRecord) -> dict:
    return {
        "symbol": record.symbol,
        "horizon": record.horizon,
        "event_type": record.event_type,
        "expected_direction": record.expected_direction,
        "entry_price": round(record.entry_price, 4),
        "exit_price": round(record.exit_price, 4),
        "return_pct": round(record.return_pct, 3),
        "correct": record.correct,
        "outcome_status": "flat" if abs(record.return_pct) < _ACTIONABLE_RETURN_THRESHOLD_PCT else "correct" if record.correct else "incorrect",
        "confidence": record.confidence,
        "severity": record.severity,
        "ml_score": record.ml_score,
        "evaluated_at": record.evaluated_at.isoformat() if hasattr(record.evaluated_at, "isoformat") else None,
    }


def _parse_assets(assets_json: str, mapped_assets: str) -> list[dict]:
    try:
        assets = json.loads(assets_json) if assets_json else []
        if isinstance(assets, list) and assets:
            return [asset for asset in assets if isinstance(asset, dict)]
    except Exception:
        pass

    return [{"symbol": symbol} for symbol in _parse_json_list(mapped_assets)]


def _parse_json_list(value: str) -> list[str]:
    try:
        parsed = json.loads(value) if value else []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        pass
    return []


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_signal_correct(direction: str, return_pct: float) -> bool:
    if direction == "positive":
        return return_pct > 0
    if direction == "negative":
        return return_pct < 0
    return abs(return_pct) < 0.25


def _outcome_status(record: SignalBacktest) -> str:
    if not _is_actionable(record):
        return "flat"
    return "correct" if record.correct else "incorrect"


def _is_actionable(record: SignalBacktest) -> bool:
    return abs(record.return_pct) >= _ACTIONABLE_RETURN_THRESHOLD_PCT


def _signal_return_pct(record: SignalBacktest) -> float:
    if record.expected_direction == "negative":
        return -float(record.return_pct)
    return float(record.return_pct)


def _benchmark_curve(length: int) -> list[dict[str, float]]:
    if length <= 0:
        return []

    try:
        spy = fetch_price("SPY")
        history = [float(value) for value in spy.get("history", []) if value is not None]
    except Exception as exc:
        logger.warning("SPY benchmark fetch failed: %s", exc)
        history = []

    if len(history) < 2:
        return [
            {"equity": round(_INITIAL_EQUITY, 2), "return_pct": 0.0}
            for _ in range(length)
        ]

    sampled: list[float] = []
    for index in range(length):
        history_index = round(index * (len(history) - 1) / max(length - 1, 1))
        sampled.append(history[history_index])

    start = sampled[0]
    if start <= 0:
        return [
            {"equity": round(_INITIAL_EQUITY, 2), "return_pct": 0.0}
            for _ in range(length)
        ]

    return [
        {
            "equity": round(_INITIAL_EQUITY * (price / start), 2),
            "return_pct": round(((price / start) - 1.0) * 100.0, 3),
        }
        for price in sampled
    ]


def _group_accuracy_pct(items: list[SignalBacktest]) -> float:
    actionable = [item for item in items if abs(item.return_pct) >= _ACTIONABLE_RETURN_THRESHOLD_PCT]
    if not actionable:
        return 0.0

    correct = sum(1 for item in actionable if item.correct)
    return round((correct / len(actionable)) * 100, 1)


def _canonical_group_accuracy_pct(items: list[CanonicalBacktestRecord]) -> float:
    actionable = [item for item in items if abs(item.return_pct) >= _ACTIONABLE_RETURN_THRESHOLD_PCT]
    if not actionable:
        return 0.0
    correct = sum(1 for item in actionable if item.correct)
    return round((correct / len(actionable)) * 100, 1)


def _target_label(benchmark_label: int | None, raw_label: int | None) -> int:
    label = benchmark_label if benchmark_label is not None else raw_label
    if label is None:
        raise ValueError("backtest row missing outcome label")
    return int(label)


def _direction_matches_label(direction: str, label: int) -> bool:
    return (direction == "up" and label == 1) or (direction == "down" and label == 0)


def _severity_label(value: float) -> str:
    if value >= 0.9:
        return "critical"
    if value >= 0.7:
        return "high"
    if value >= 0.4:
        return "medium"
    return "low"
