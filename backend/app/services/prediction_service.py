import json

from sqlalchemy.orm import Session

from app.db.models import Event, EventRecord, FeatureSnapshot, MultiHorizonOutcome, Outcome, Prediction
from app.repositories.event_repository import list_recent_events
from app.services.baseline_scoring import BASELINE_MODEL_VERSION
from app.services.feature_service import build_feature_vector, score_signal_quality
from app.services.ml_scorer import explain_prediction
from app.services.model_evaluation_service import get_model_evaluation
from app.services.training_run_service import get_model_health
from app.services.mapping_service import asset_exposure_for

_SEVERITY_MOVE = {
    "low": 0.35,
    "medium": 0.8,
    "high": 1.45,
    "critical": 2.4,
}

_EVENT_MULTIPLIER = {
    "geopolitical_conflict": 1.35,
    "supply_shock": 1.25,
    "inflation": 1.1,
    "interest_rate_change": 1.15,
    "economic_data": 0.9,
    "corporate_earnings": 1.0,
    "general_market": 0.75,
}


MIN_SIGNAL_QUALITY = 0.62
GATED_ASSETS = {"GLD", "XLF"}
GATED_EVENT_TYPES = {"interest_rate_change"}
GATED_HORIZON_DAYS = {3}
CURRENT_MODEL_PREFIX = "xgboost-v1"


def generate_predictions(
    db: Session,
    limit: int = 50,
    *,
    min_quality: float = MIN_SIGNAL_QUALITY,
    include_weak: bool = False,
) -> dict:
    calibration_profile = _calibration_profile(db)
    stored = _stored_ml_predictions(
        db,
        limit=limit,
        min_quality=min_quality,
        calibration_profile=calibration_profile,
    )
    if stored:
        filtered = [
            item
            for item in stored
            if include_weak or item["is_actionable"]
        ]
        return {
            "model_version": _summary_model_version(filtered or stored),
            "count": len(filtered),
            "total_considered": len(stored),
            "weak_filtered": len(stored) - len(filtered),
            "min_quality": min_quality,
            "predictions": filtered[:25],
        }
    predictions = []
    weak_filtered = 0
    total_considered = 0

    for event in list_recent_events(db, limit=limit):
        sectors = _parse_json_list(event.affected_sectors)
        assets = _parse_assets(event.assets_json, event.mapped_assets)

        for asset in assets:
            symbol = str(asset.get("symbol", "")).upper()
            price = _safe_float(asset.get("price")) or event.price_at_event
            if not symbol or not price:
                continue
            total_considered += 1

            features = build_feature_vector(
                event,
                symbol=symbol,
                entry_price=float(price),
                exit_price=float(price),
                return_pct=0.0,
                sectors=sectors,
            )
            probability = score_signal_quality(features)
            exposure = asset_exposure_for(symbol, {
                "event_type": event.event_type,
                "affected_sectors": sectors,
            })
            move_mid = _expected_move(event, float(features["asset_specificity"]))
            ranking_score = _ranking_score(
                probability=probability,
                confidence=event.confidence,
                expected_move=move_mid,
                severity=event.severity,
            )
            horizon = _horizon(event)
            filter_reason = _filter_reason(
                symbol=symbol,
                event_type=event.event_type,
                horizon=horizon,
                probability=probability,
                confidence=event.confidence,
                expected_move=move_mid,
                min_quality=min_quality,
                segment_quality_reason=_segment_quality_reason(db, symbol, event.event_type, horizon),
                drift_reason=_drift_reason(db),
                calibration_reason=None,
            )
            if filter_reason and not include_weak:
                weak_filtered += 1
                continue
            direction_multiplier = _direction_multiplier(event.impact_direction) * float(exposure["sign"])
            if event.impact_direction == "neutral":
                direction_multiplier = 0
            impact_direction = "positive" if direction_multiplier > 0 else "negative" if direction_multiplier < 0 else "neutral"

            predictions.append(
                {
                    "symbol": symbol,
                    "title": event.title,
                    "event_type": event.event_type,
                    "affected_sectors": sectors,
                    "impact_direction": impact_direction,
                    "severity": event.severity,
                    "confidence": event.confidence,
                    "probability": probability,
                    "ranking_score": ranking_score,
                    "is_actionable": filter_reason is None,
                    "filter_reason": filter_reason,
                    "actionability": _actionability(filter_reason),
                    "confidence_tier": _confidence_tier(probability),
                    "horizon": horizon,
                    "expected_move_pct": round(move_mid * direction_multiplier, 2),
                    "expected_move_low_pct": round(max(0.1, move_mid * 0.55) * direction_multiplier, 2),
                    "expected_move_high_pct": round(move_mid * 1.55 * direction_multiplier, 2),
                    "expected_excess_return_pct": round(move_mid * direction_multiplier * 0.45, 2),
                    "why_this_matters": _why_this_matters(event.event_type, symbol, str(exposure["rationale"])),
                    "risk_factors": _risk_factors(filter_reason, event.event_type),
                    "gate_status": _gate_status(filter_reason),
                    "bull_case": _scenario_text(event, symbol, "bull"),
                    "base_case": _scenario_text(event, symbol, "base"),
                    "bear_case": _scenario_text(event, symbol, "bear"),
                    "drivers": _drivers(event, sectors, probability),
                    "model_version": "interpretable-v2",
                    "shap_contributions": [],
                }
            )

    predictions.sort(
        key=lambda item: (item["is_actionable"], item["ranking_score"], abs(item["expected_move_pct"])),
        reverse=True,
    )

    return {
        "model_version": "interpretable-v2",
        "count": len(predictions),
        "total_considered": total_considered,
        "weak_filtered": weak_filtered,
        "min_quality": min_quality,
        "predictions": predictions[:25],
    }


def _stored_ml_predictions(
    db: Session,
    limit: int,
    min_quality: float,
    calibration_profile: dict[str, dict[str, dict[str, float | int]]] | None,
) -> list[dict]:
    rows = _stored_prediction_rows(db, limit=limit)
    predictions: list[dict] = []
    for prediction, snapshot in rows:
        event = prediction.event
        raw_text = event.raw_text if event is not None else ""
        title = raw_text.splitlines()[0] if raw_text else f"{prediction.asset} prediction"
        direction = "positive" if prediction.predicted_direction == "up" else "negative"
        confidence = float(prediction.confidence)
        volatility = float((snapshot.market_features or {}).get("rolling_volatility", 0.0))
        expected_move = max(0.1, min(3.5, volatility * 200.0 or confidence * 1.2))
        signed_move = expected_move if direction == "positive" else -expected_move
        model_version = str((snapshot.derived_features or {}).get("prediction_model_version", "xgboost-v1"))
        ranking_score = _ranking_score(
            probability=confidence,
            confidence=confidence,
            expected_move=expected_move,
            severity=_severity_label(float(event.severity) if event is not None else 0.0),
        )
        calibration_reason = _calibration_filter_reason(calibration_profile, model_version, confidence)
        filter_reason = _filter_reason(
            symbol=prediction.asset,
            event_type=event.event_type if event is not None else "general_market",
            horizon=prediction.horizon,
            probability=confidence,
            confidence=confidence,
            expected_move=expected_move,
            min_quality=min_quality,
            segment_quality_reason=_segment_quality_reason(
                db,
                prediction.asset,
                event.event_type if event is not None else "general_market",
                prediction.horizon,
                model_version=model_version,
            ),
            drift_reason=_drift_reason(db),
            calibration_reason=calibration_reason,
            model_version=model_version,
        )
        shap = explain_prediction(
            snapshot.event_features,
            snapshot.market_features,
            snapshot.derived_features,
        )

        predictions.append({
            "symbol": prediction.asset,
            "title": title,
            "event_type": event.event_type if event is not None else "general_market",
            "affected_sectors": [],
            "impact_direction": direction,
            "severity": _severity_label(float(event.severity) if event is not None else 0.0),
            "confidence": confidence,
            "probability": confidence,
            "ranking_score": ranking_score,
            "is_actionable": filter_reason is None,
            "filter_reason": filter_reason,
            "actionability": _actionability(filter_reason),
            "confidence_tier": _confidence_tier(confidence),
            "horizon": prediction.horizon,
            "expected_move_pct": round(signed_move, 2),
            "expected_move_low_pct": round(signed_move * 0.55, 2),
            "expected_move_high_pct": round(signed_move * 1.55, 2),
            "expected_excess_return_pct": round(signed_move * 0.45, 2),
            "why_this_matters": _why_this_matters(
                event.event_type if event is not None else "general_market",
                prediction.asset,
                "stored ML signal",
            ),
            "risk_factors": _risk_factors(filter_reason, event.event_type if event is not None else "general_market"),
            "gate_status": _gate_status(filter_reason),
            "bull_case": _stored_scenario(prediction.asset, direction, "bull"),
            "base_case": f"{prediction.asset} follows the stored {prediction.horizon} ML signal.",
            "bear_case": _stored_scenario(prediction.asset, direction, "bear"),
            "drivers": _stored_drivers(model_version, confidence, shap),
            "model_version": model_version,
            "shap_contributions": shap,
        })

    predictions.sort(key=lambda item: (item["probability"], abs(item["expected_move_pct"])), reverse=True)
    return predictions


def _stored_prediction_rows(db: Session, *, limit: int) -> list[tuple[Prediction, FeatureSnapshot]]:
    query = (
        db.query(Prediction, FeatureSnapshot)
        .join(FeatureSnapshot, FeatureSnapshot.prediction_id == Prediction.id)
    )
    ml_rows = (
        query.filter(Prediction.model_version.like(f"{CURRENT_MODEL_PREFIX}%"))
        .order_by(Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )
    if ml_rows:
        return ml_rows
    return (
        query.order_by(Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )


def _summary_model_version(predictions: list[dict]) -> str:
    versions = [str(item.get("model_version", "")) for item in predictions if item.get("model_version")]
    if not versions:
        return "unknown"
    return versions[0]


def _severity_label(value: float) -> str:
    if value >= 0.9:
        return "critical"
    if value >= 0.7:
        return "high"
    if value >= 0.4:
        return "medium"
    return "low"


def _stored_scenario(symbol: str, direction: str, scenario: str) -> str:
    if scenario == "bull":
        if direction == "negative":
            return f"{symbol} benefits if the downside signal mean-reverts or macro pressure fades."
        return f"{symbol} extends higher if the model drivers persist and market regime stays supportive."
    if direction == "positive":
        return f"{symbol} weakens if the signal is already priced in or regime conditions deteriorate."
    return f"{symbol} falls further if model drivers intensify and risk appetite weakens."


def _stored_drivers(model_version: str, confidence: float, shap: list[dict]) -> list[str]:
    drivers = [
        f"{round(confidence * 100)}% stored model confidence",
        f"model version: {model_version}",
    ]
    drivers.extend(_driver_label(str(item["feature"])) for item in shap[:3])
    return drivers


def _driver_label(feature: str) -> str:
    labels = {
        "event_news_provider_count": "multiple news sources confirmed the catalyst",
        "event_news_source_count": "broad source coverage",
        "event_news_cross_provider_consensus": "cross-provider news agreement",
        "event_news_provider_weight": "higher-quality news source mix",
        "event_news_symbol_match": "article directly matched the asset",
        "event_news_symbol_specific": "company-specific news signal",
        "event_news_provider_sentiment_score": "provider sentiment signal",
        "event_news_sentiment_alignment": "news sentiment aligned with the event",
        "event_news_recency_hours": "fresh news catalyst",
        "event_sentiment": "event tone",
        "event_severity": "event severity",
        "derived_baseline_score": "baseline macro score",
        "derived_sentiment_x_severity": "tone adjusted for severity",
        "derived_sentiment_x_sector_sensitivity": "sector-sensitive event tone",
        "derived_rate_level": "rate backdrop",
        "derived_benchmark_price": "market benchmark level",
        "market_rolling_volatility": "recent volatility",
        "market_return_5d": "recent asset trend",
        "market_return_10d": "medium-term asset trend",
        "market_asset_class_encoded": "asset class behavior",
    }
    return labels.get(feature, feature.replace("_", " "))


def _expected_move(event: EventRecord, asset_specificity: float) -> float:
    base = _SEVERITY_MOVE.get(event.severity, 0.35)
    event_multiplier = _EVENT_MULTIPLIER.get(event.event_type, 0.85)
    confidence_multiplier = 0.55 + event.confidence
    specificity_multiplier = 0.75 + asset_specificity
    return base * event_multiplier * confidence_multiplier * specificity_multiplier


def _ranking_score(
    *,
    probability: float,
    confidence: float,
    expected_move: float,
    severity: str,
) -> float:
    severity_score = {
        "low": 0.25,
        "medium": 0.5,
        "high": 0.8,
        "critical": 1.0,
    }.get(severity, 0.25)
    move_score = min(abs(expected_move) / 3.0, 1.0)
    return round((probability * 0.45) + (confidence * 0.25) + (move_score * 0.2) + (severity_score * 0.1), 4)


def _filter_reason(
    *,
    symbol: str,
    event_type: str,
    horizon: str,
    probability: float,
    confidence: float,
    expected_move: float,
    min_quality: float,
    segment_quality_reason: str | None = None,
    drift_reason: str | None = None,
    calibration_reason: str | None = None,
    model_version: str | None = None,
) -> str | None:
    if model_version == BASELINE_MODEL_VERSION:
        return "stale_baseline_model"
    if drift_reason is not None:
        return drift_reason
    if calibration_reason is not None:
        return calibration_reason
    segment_reason = _segment_filter_reason(
        symbol=symbol,
        event_type=event_type,
        horizon=horizon,
    )
    if segment_reason is not None:
        return segment_reason
    if segment_quality_reason is not None:
        return segment_quality_reason
    if probability < min_quality:
        return "low_quality"
    if confidence < 0.55:
        return "low_confidence"
    if abs(expected_move) < 0.25:
        return "small_expected_move"
    return None


def _direction_multiplier(direction: str) -> float:
    if direction == "positive":
        return 1.0
    if direction == "negative":
        return -1.0
    return 0.0


def _actionability(filter_reason: str | None) -> str:
    if filter_reason is None:
        return "actionable"
    if filter_reason in {"small_expected_move", "low_quality", "low_confidence"}:
        return "watch"
    return "blocked"


def _confidence_tier(probability: float) -> str:
    if probability >= 0.75:
        return "high"
    if probability >= 0.62:
        return "medium"
    return "low"


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.6:
        return "0.5-0.6"
    if confidence < 0.7:
        return "0.6-0.7"
    if confidence < 0.8:
        return "0.7-0.8"
    return "0.8+"


def _gate_status(filter_reason: str | None) -> dict[str, str | bool | None]:
    return {
        "passed": filter_reason is None,
        "reason": filter_reason,
    }


def _why_this_matters(event_type: str, symbol: str, exposure_rationale: str) -> str:
    return f"{symbol} is linked to {event_type.replace('_', ' ')} through {exposure_rationale}."


def _risk_factors(filter_reason: str | None, event_type: str) -> list[str]:
    risks: list[str] = []
    if filter_reason is not None:
        risks.append(filter_reason.replace("_", " "))
    if event_type in {"geopolitical_conflict", "supply_shock"}:
        risks.append("headline reversal risk")
    if event_type in {"inflation", "interest_rate_change"}:
        risks.append("rates repricing risk")
    return risks or ["model uncertainty", "market regime shift"]


def _calibration_profile(db: Session) -> dict[str, dict[str, dict[str, float | int]]] | None:
    try:
        evaluation = get_model_evaluation(db, limit=50_000)
    except Exception:
        return None
    by_model = evaluation.get("calibration_by_model_version")
    if not isinstance(by_model, dict):
        return None
    return {
        str(model_version): calibration
        for model_version, calibration in by_model.items()
        if isinstance(calibration, dict)
    }


def _calibration_filter_reason(
    profile: dict[str, dict[str, dict[str, float | int]]] | None,
    model_version: str,
    confidence: float,
) -> str | None:
    if profile is None:
        return None
    buckets = profile.get(model_version)
    if buckets is None and model_version.startswith(CURRENT_MODEL_PREFIX):
        buckets = next(
            (value for key, value in profile.items() if key.startswith(CURRENT_MODEL_PREFIX)),
            None,
        )
    if buckets is None:
        return None
    bucket = buckets.get(_confidence_bucket(confidence))
    if not bucket:
        return None
    samples = int(bucket.get("samples", 0))
    observed_accuracy = float(bucket.get("observed_accuracy", 0.0))
    avg_excess_return = float(bucket.get("avg_excess_return", 0.0))
    if samples >= 30 and (observed_accuracy < 0.55 or avg_excess_return < -0.001):
        return "weak_calibration_bucket"
    return None


def _drift_reason(db: Session) -> str | None:
    try:
        health = get_model_health(db)
    except Exception:
        return None
    confidence = health.get("confidence_drift") or {}
    if health.get("drift_detected") and float(confidence.get("psi", 0.0)) >= 0.35:
        return "severe_drift"
    return None


def _segment_quality_reason(
    db: Session,
    symbol: str,
    event_type: str,
    horizon: str,
    *,
    model_version: str | None = None,
) -> str | None:
    rows = _segment_label_rows(
        db,
        symbol=symbol,
        event_type=event_type,
        horizon_days=_horizon_days(horizon),
        model_version=model_version,
    )
    if len(rows) < 8:
        return None

    model_acc = sum(_direction_matches_label(direction, label) for direction, label in rows) / len(rows)
    always_up = sum(1 for _, label in rows if label == 1) / len(rows)
    always_down = 1.0 - always_up
    best_baseline = max(always_up, always_down)

    if model_acc <= best_baseline:
        return "segment_not_beating_baseline"
    if max(always_up, always_down) < 0.58:
        return "contradictory_history"
    return None


def _segment_label_rows(
    db: Session,
    *,
    symbol: str,
    event_type: str,
    horizon_days: int | None,
    model_version: str | None = None,
) -> list[tuple[str, int]]:
    symbol = symbol.upper()
    multi = (
        db.query(Prediction.predicted_direction, MultiHorizonOutcome.benchmark_label, MultiHorizonOutcome.label)
        .join(Event, Event.id == Prediction.event_id)
        .join(MultiHorizonOutcome, MultiHorizonOutcome.prediction_id == Prediction.id)
        .filter(
            Prediction.asset == symbol,
            Event.event_type == event_type,
            MultiHorizonOutcome.benchmark_label.isnot(None) | MultiHorizonOutcome.label.isnot(None),
        )
    )
    if horizon_days is not None:
        multi = multi.filter(MultiHorizonOutcome.horizon_days == horizon_days)
    if model_version is not None:
        multi = multi.filter(Prediction.model_version == model_version)

    multi_rows = [
        (direction, _target_label(benchmark_label, label))
        for direction, benchmark_label, label in multi.limit(100).all()
    ]
    if multi_rows:
        return multi_rows

    single = (
        db.query(Prediction.predicted_direction, Outcome.benchmark_label, Outcome.label)
        .join(Event, Event.id == Prediction.event_id)
        .join(Outcome, Outcome.prediction_id == Prediction.id)
        .filter(
            Prediction.asset == symbol,
            Event.event_type == event_type,
            Outcome.benchmark_label.isnot(None) | Outcome.label.isnot(None),
        )
    )
    if model_version is not None:
        single = single.filter(Prediction.model_version == model_version)
    return [
        (direction, _target_label(benchmark_label, label))
        for direction, benchmark_label, label in single.limit(100).all()
    ]


def _target_label(benchmark_label: int | None, raw_label: int | None) -> int:
    label = benchmark_label if benchmark_label is not None else raw_label
    if label is None:
        raise ValueError("segment row missing outcome label")
    return int(label)


def _direction_matches_label(direction: str, label: int) -> int:
    return int((direction == "up" and label == 1) or (direction == "down" and label == 0))


def _segment_filter_reason(
    *,
    symbol: str,
    event_type: str,
    horizon: str,
) -> str | None:
    if symbol.upper() in GATED_ASSETS:
        return "gated_asset"
    if event_type in GATED_EVENT_TYPES:
        return "gated_event_type"
    if _horizon_days(horizon) in GATED_HORIZON_DAYS:
        return "gated_horizon"
    return None


def _horizon_days(value: str | None) -> int | None:
    if not value:
        return None
    normalized = value.strip().lower()
    first_part = normalized.split("-")[0].replace("d", "")
    try:
        return int(first_part)
    except ValueError:
        return None


def _horizon(event: EventRecord) -> str:
    if event.event_type in {"geopolitical_conflict", "supply_shock", "corporate_earnings"}:
        return "1d-5d"
    if event.event_type in {"inflation", "interest_rate_change", "economic_data"}:
        return "5d-20d"
    return "1d-10d"


def _drivers(event: EventRecord, sectors: list[str], probability: float) -> list[str]:
    drivers = [
        f"{event.severity} severity classification",
        f"{round(event.confidence * 100)}% model confidence",
        f"{round(probability * 100)}% signal quality score",
    ]
    if sectors:
        drivers.append(f"sector exposure: {', '.join(sectors[:3])}")
    return drivers


def _scenario_text(event: EventRecord, symbol: str, scenario: str) -> str:
    direction = event.impact_direction
    if scenario == "bull":
        if direction == "negative":
            return f"{symbol} stabilizes if the macro event fades or policy response offsets the shock."
        return f"{symbol} outperforms if the event transmission strengthens and liquidity remains supportive."
    if scenario == "bear":
        if direction == "positive":
            return f"{symbol} underperforms if the catalyst is already priced in or risk appetite weakens."
        return f"{symbol} remains pressured if the event escalates or spreads to adjacent sectors."
    return f"{symbol} follows the classified {event.event_type} signal over the {_horizon(event)} horizon."


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
