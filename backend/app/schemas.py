from datetime import datetime

from pydantic import BaseModel


class Article(BaseModel):
    title: str
    description: str
    publishedAt: str


class AssetPrice(BaseModel):
    symbol: str
    price: float


class MarketData(AssetPrice):
    history: list[float]


class MarketHistoryPoint(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class MarketHistory(BaseModel):
    symbol: str
    range: str
    interval: str
    currency: str | None = None
    previous_close: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: int
    average_volume: int | None = None
    points: list[MarketHistoryPoint]


class EventClassification(BaseModel):
    event_type: str
    affected_sectors: list[str]
    impact_direction: str
    confidence: float
    severity: str
    reasoning: str


class CombinedArticle(EventClassification):
    title: str
    description: str
    assets: list[AssetPrice]


class StoredEvent(EventClassification):
    id: int
    article_hash: str
    title: str
    description: str
    publishedAt: str
    mapped_assets: list[str]
    created_at: datetime


class WatchlistRequest(BaseModel):
    symbols: list[str]


class WatchlistAssetImpact(BaseModel):
    symbol: str
    matched_events: int
    highest_severity: str
    net_direction: str
    reason: str


class WatchlistImpact(BaseModel):
    watchlist: list[str]
    impacted_assets: list[WatchlistAssetImpact]
    portfolio_risk_score: int


class EvaluatedSignal(BaseModel):
    title: str
    event_type: str
    impact_direction: str
    signal_correct: bool
    price_at_event: float | None
    price_at_evaluation: float | None
    evaluated_at: str | None


class SignalTypeStats(BaseModel):
    total: int
    correct: int
    accuracy_pct: float


class SignalAccuracy(BaseModel):
    total_evaluated: int
    correct: int
    accuracy_pct: float
    by_event_type: dict[str, SignalTypeStats]
    recent: list[EvaluatedSignal]


class BacktestGroupStats(BaseModel):
    total: int
    actionable: int
    flat: int
    correct: int
    accuracy_pct: float
    avg_return_pct: float
    avg_ml_score: float


class BacktestSignal(BaseModel):
    symbol: str
    horizon: str
    event_type: str
    expected_direction: str
    entry_price: float
    exit_price: float
    return_pct: float
    correct: bool
    outcome_status: str
    confidence: float
    severity: str
    ml_score: float
    evaluated_at: str | None


class BacktestSummary(BaseModel):
    total_signals: int
    actionable_signals: int
    flat_signals: int
    correct: int
    accuracy_pct: float
    avg_return_pct: float
    avg_ml_score: float
    skipped: int
    by_event_type: dict[str, BacktestGroupStats]
    by_symbol: dict[str, BacktestGroupStats]
    by_severity: dict[str, BacktestGroupStats]
    top_signals: list[BacktestSignal]
    recent: list[BacktestSignal]


class AssetPrediction(BaseModel):
    model_config = {"protected_namespaces": ()}

    symbol: str
    title: str
    event_type: str
    affected_sectors: list[str]
    impact_direction: str
    severity: str
    confidence: float
    probability: float
    horizon: str
    expected_move_pct: float
    expected_move_low_pct: float
    expected_move_high_pct: float
    bull_case: str
    base_case: str
    bear_case: str
    drivers: list[str]
    model_version: str


class PredictionSummary(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_version: str
    count: int
    predictions: list[AssetPrediction]


class TrainingExample(BaseModel):
    features: dict[str, object]
    label: int
