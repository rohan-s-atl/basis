from dataclasses import dataclass
from typing import Literal

BASELINE_MODEL_VERSION = "baseline-rule-v1"
PredictionDirection = Literal["up", "down"]


@dataclass(frozen=True)
class BaselinePrediction:
    predicted_direction: PredictionDirection
    confidence: float
    score: float


def score_baseline_prediction(
    *,
    sentiment: float,
    severity: float,
    volatility: float,
) -> BaselinePrediction:
    """Interpretable pre-ML scoring rule used to log baseline predictions."""
    normalized_sentiment = _clamp(sentiment, -1.0, 1.0)
    normalized_severity = _clamp(severity, 0.0, 1.0)
    normalized_volatility = _clamp(volatility, 0.0, 1.0)
    sentiment_sign = -1.0 if normalized_sentiment < 0 else 1.0

    score = (
        0.60 * normalized_sentiment
        + sentiment_sign * 0.30 * normalized_severity
        - sentiment_sign * 0.10 * normalized_volatility
    )
    predicted_direction: PredictionDirection = "up" if score >= 0 else "down"
    confidence = _clamp(0.50 + abs(score) * 0.50, 0.50, 0.95)

    return BaselinePrediction(
        predicted_direction=predicted_direction,
        confidence=round(confidence, 4),
        score=round(score, 6),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
