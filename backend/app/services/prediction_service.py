import json

from sqlalchemy.orm import Session

from app.db.models import EventRecord, FeatureSnapshot, Prediction
from app.repositories.event_repository import list_recent_events
from app.services.feature_service import build_feature_vector, score_signal_quality
from app.services.ml_scorer import explain_prediction

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


def generate_predictions(
    db: Session,
    limit: int = 50,
    *,
    min_quality: float = MIN_SIGNAL_QUALITY,
    include_weak: bool = False,
) -> dict:
    stored = _stored_ml_predictions(db, limit=limit)
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
            move_mid = _expected_move(event, float(features["asset_specificity"]))
            ranking_score = _ranking_score(
                probability=probability,
                confidence=event.confidence,
                expected_move=move_mid,
                severity=event.severity,
            )
            filter_reason = _filter_reason(
                probability=probability,
                confidence=event.confidence,
                expected_move=move_mid,
                min_quality=min_quality,
            )
            if filter_reason and not include_weak:
                weak_filtered += 1
                continue
            direction_multiplier = -1 if event.impact_direction == "negative" else 1
            if event.impact_direction == "neutral":
                direction_multiplier = 0

            predictions.append(
                {
                    "symbol": symbol,
                    "title": event.title,
                    "event_type": event.event_type,
                    "affected_sectors": sectors,
                    "impact_direction": event.impact_direction,
                    "severity": event.severity,
                    "confidence": event.confidence,
                    "probability": probability,
                    "ranking_score": ranking_score,
                    "is_actionable": filter_reason is None,
                    "filter_reason": filter_reason,
                    "horizon": _horizon(event),
                    "expected_move_pct": round(move_mid * direction_multiplier, 2),
                    "expected_move_low_pct": round(max(0.1, move_mid * 0.55) * direction_multiplier, 2),
                    "expected_move_high_pct": round(move_mid * 1.55 * direction_multiplier, 2),
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


def _stored_ml_predictions(db: Session, limit: int) -> list[dict]:
    rows = (
        db.query(Prediction, FeatureSnapshot)
        .join(FeatureSnapshot, FeatureSnapshot.prediction_id == Prediction.id)
        .order_by(Prediction.timestamp.desc())
        .limit(limit)
        .all()
    )
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
        filter_reason = _filter_reason(
            probability=confidence,
            confidence=confidence,
            expected_move=expected_move,
            min_quality=MIN_SIGNAL_QUALITY,
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
            "horizon": prediction.horizon,
            "expected_move_pct": round(signed_move, 2),
            "expected_move_low_pct": round(signed_move * 0.55, 2),
            "expected_move_high_pct": round(signed_move * 1.55, 2),
            "bull_case": _stored_scenario(prediction.asset, direction, "bull"),
            "base_case": f"{prediction.asset} follows the stored {prediction.horizon} ML signal.",
            "bear_case": _stored_scenario(prediction.asset, direction, "bear"),
            "drivers": _stored_drivers(model_version, confidence, shap),
            "model_version": model_version,
            "shap_contributions": shap,
        })

    predictions.sort(key=lambda item: (item["probability"], abs(item["expected_move_pct"])), reverse=True)
    return predictions


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
    drivers.extend(str(item["feature"]).replace("_", " ") for item in shap[:3])
    return drivers


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
    probability: float,
    confidence: float,
    expected_move: float,
    min_quality: float,
) -> str | None:
    if probability < min_quality:
        return "low_quality"
    if confidence < 0.55:
        return "low_confidence"
    if abs(expected_move) < 0.25:
        return "small_expected_move"
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
