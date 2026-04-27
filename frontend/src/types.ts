export type ImpactDirection = "positive" | "negative" | "neutral";

export type Severity = "low" | "medium" | "high" | "critical";

export type ConfidenceLevel = "high" | "medium" | "low";

export type AssetPrice = {
  symbol: string;
  price?: number;
  history?: number[];
};

export type EventRecord = {
  id?: number;
  article_hash?: string;
  title: string;
  description: string;
  event_type: string;
  affected_sectors: string[];
  impact_direction: ImpactDirection;
  reasoning: string;
  confidence?: number;
  severity?: Severity;
  recommended_assets?: AssetPrice[];
  assets?: AssetPrice[];
  mapped_assets?: string[];
};

export type AssetImpactRow = {
  symbol: string;
  price?: number;
  direction: ImpactDirection;
  eventType: string;
  confidence: ConfidenceLevel;
  severity: Severity;
};

export type PriceQuote = {
  symbol: string;
  price: number;
  previousPrice?: number;
  history: number[];
};

export type MarketHistoryPoint = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type MarketHistory = {
  symbol: string;
  range: string;
  interval: string;
  currency?: string | null;
  previous_close?: number | null;
  open?: number | null;
  high?: number | null;
  low?: number | null;
  close?: number | null;
  volume: number;
  average_volume?: number | null;
  points: MarketHistoryPoint[];
};

export type WatchlistAssetImpact = {
  symbol: string;
  matched_events: number;
  highest_severity: Severity;
  net_direction: ImpactDirection;
  reason: string;
};

export type WatchlistImpact = {
  watchlist: string[];
  impacted_assets: WatchlistAssetImpact[];
  portfolio_risk_score: number;
};

export type EventCluster = {
  id: string;
  title: string;
  eventType: string;
  events: EventRecord[];
  sectors: string[];
  assets: string[];
  direction: ImpactDirection;
  severity: Severity;
  confidence: number;
};

export type AlertRule = {
  id: string;
  label: string;
  active: boolean;
  severity: Severity;
  description: string;
  matchedEvents: number;
  rule: string;
};

export type EvaluatedSignal = {
  title: string;
  event_type: string;
  impact_direction: ImpactDirection;
  signal_correct: boolean;
  price_at_event: number | null;
  price_at_evaluation: number | null;
  evaluated_at: string | null;
};

export type SignalTypeStats = {
  total: number;
  correct: number;
  accuracy_pct: number;
};

export type SignalAccuracy = {
  total_evaluated: number;
  correct: number;
  accuracy_pct: number;
  by_event_type: Record<string, SignalTypeStats>;
  recent: EvaluatedSignal[];
};

export type BacktestGroupStats = {
  total: number;
  actionable: number;
  flat: number;
  correct: number;
  accuracy_pct: number;
  avg_return_pct: number;
  avg_ml_score: number;
};

export type BacktestSignal = {
  symbol: string;
  horizon: string;
  event_type: string;
  expected_direction: ImpactDirection;
  entry_price: number;
  exit_price: number;
  return_pct: number;
  correct: boolean;
  outcome_status: "correct" | "incorrect" | "flat";
  confidence: number;
  severity: Severity;
  ml_score: number;
  evaluated_at: string | null;
};

export type BacktestSummary = {
  total_signals: number;
  actionable_signals: number;
  flat_signals: number;
  correct: number;
  accuracy_pct: number;
  avg_return_pct: number;
  avg_ml_score: number;
  skipped: number;
  by_event_type: Record<string, BacktestGroupStats>;
  by_symbol: Record<string, BacktestGroupStats>;
  by_severity: Record<string, BacktestGroupStats>;
  top_signals: BacktestSignal[];
  recent: BacktestSignal[];
};

export type AssetPrediction = {
  symbol: string;
  title: string;
  event_type: string;
  affected_sectors: string[];
  impact_direction: ImpactDirection;
  severity: Severity;
  confidence: number;
  probability: number;
  horizon: string;
  expected_move_pct: number;
  expected_move_low_pct: number;
  expected_move_high_pct: number;
  bull_case: string;
  base_case: string;
  bear_case: string;
  drivers: string[];
  model_version: string;
};

export type PredictionSummary = {
  model_version: string;
  count: number;
  predictions: AssetPrediction[];
};
