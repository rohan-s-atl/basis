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

export type PortfolioSimulationPoint = {
  index: number;
  label: string;
  equity: number;
  return_pct: number;
  benchmark_equity: number;
  benchmark_return_pct: number;
};

export type PortfolioSimulation = {
  initial_equity: number;
  final_equity: number;
  total_return_pct: number;
  benchmark_return_pct: number;
  excess_return_pct: number;
  signals: number;
  win_rate_pct: number;
  allocation_pct: number;
  points: PortfolioSimulationPoint[];
};

export type ShapContribution = {
  feature: string;
  shap_value: number;
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
  ranking_score: number;
  is_actionable: boolean;
  filter_reason: string | null;
  horizon: string;
  expected_move_pct: number;
  expected_move_low_pct: number;
  expected_move_high_pct: number;
  bull_case: string;
  base_case: string;
  bear_case: string;
  drivers: string[];
  model_version: string;
  shap_contributions: ShapContribution[];
};

export type PredictionSummary = {
  model_version: string;
  count: number;
  total_considered: number;
  weak_filtered: number;
  min_quality: number;
  predictions: AssetPrediction[];
};

export type FeatureImportance = {
  feature: string;
  importance: number;
};

export type LabelBalance = {
  positive: number;
  negative: number;
  positive_count: number;
  negative_count: number;
};

export type TrainingRun = {
  id: string;
  trained_at: string;
  triggered_by: string;
  dataset_size: number;
  train_size: number;
  test_size: number;
  accuracy: number;
  calibrated_accuracy?: number;
  deployment_accuracy?: number;
  accuracy_metric?: string;
  roc_auc: number | null;
  brier_score_calibrated: number;
  brier_improvement: number;
  walk_forward_mean: number;
  walk_forward_std: number;
  walk_forward_folds: number;
  xgboost_roc_auc: number | null;
  logistic_roc_auc: number | null;
  winner_model: string;
  comparison_winner?: string;
  top_features: FeatureImportance[];
  label_balance: LabelBalance;
};

export type TrainingDataStats = {
  num_samples: number;
  class_distribution: LabelBalance;
  feature_count: number;
};

export type TrainingDataValidation = {
  num_samples: number;
  class_balance: LabelBalance;
  num_features: number;
  issues: string[];
};

export type ModelHealth = {
  status: "healthy" | "drift_detected" | "no_model";
  rolling_accuracy: { accuracy: number; samples: number; window: number } | null;
  peak_accuracy: number | null;
  trained_accuracy?: number | null;
  deployment_accuracy?: number | null;
  accuracy_metric?: string;
  trained_roc_auc?: number | null;
  drift_detected: boolean;
  confidence_drift: {
    drift_detected: boolean;
    psi: number;
    threshold: number;
    training: number[];
    recent: number[];
    samples: number;
  };
  last_trained_at: string | null;
  dataset_size_at_training: number | null;
  retraining_threshold: number;
};

export type EvaluationMetric = {
  samples: number;
  accuracy: number;
  avg_return?: number;
};

export type ModelEvaluation = {
  sample_count: number;
  overall: EvaluationMetric;
  high_confidence: EvaluationMetric;
  benchmark_relative: {
    samples: number;
    accuracy: number;
  };
  baselines: Record<string, EvaluationMetric>;
  by_asset: Record<string, EvaluationMetric>;
  by_event_type: Record<string, EvaluationMetric>;
  by_horizon: Record<string, EvaluationMetric>;
  by_return_bucket: Record<string, EvaluationMetric>;
  by_model_version: Record<string, EvaluationMetric>;
  data_quality: {
    dataset_samples: number;
    feature_count: number;
    issues: string[];
    duplicate_event_groups: number;
    predictions_missing_snapshots: number;
    unlabeled_predictions: number;
  };
  recommendations: string[];
};

export type MarketRegime = {
  vix_level: number;
  vix_regime_encoded: number;
  spy_trend: number;
  rate_level: number;
  market_regime_encoded: number;
};
