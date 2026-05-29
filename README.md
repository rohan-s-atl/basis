# Basis

Basis is a full-stack AI investor decision engine that turns real-world financial news, company headlines, and economic releases into structured market events, ranked asset signals, holding-level decision briefs, labeled outcomes, trust controls, and an interactive investor decision workspace.

## Links

Basis: https://basis-intel.vercel.app/

The system is built around a continuous decision loop:

```text
Multi-source news -> event classification -> asset mapping -> feature engineering
     -> signal ranking -> decision gates -> outcome labeling -> model evaluation -> retraining
```

Basis treats market information as decision data. Inflation releases, rate decisions, geopolitical events, supply shocks, energy disruptions, earnings surprises, company-specific headlines, and broad risk-on or risk-off shifts are converted into a normalized event schema. Those events are mapped to affected assets, scored by a decision pipeline, gated for actionability, tracked against future market movement, and folded back into the model lifecycle.

## Why Basis Exists

Markets absorb information quickly, but the reasoning chain behind a market move is often scattered across articles, data releases, asset reactions, portfolio exposure, and model outputs. Basis brings that chain into one place so investors can decide what deserves attention, what should be monitored, and what should be ignored for now.

The goal is to make macro intelligence actionable without hiding the evidence. A user can move from a source event to its classification, mapped assets, signal confidence, feature snapshot, model drivers, gate status, realized outcome, holding exposure, historical performance, market regime context, and retraining history without losing the thread.

Basis is designed for studying how market events become machine-learning examples and investor decisions. Each event is not only a news item; it becomes a structured record with context, features, signals, gates, outcomes, and evaluation metadata. Over time, the system builds a supervised learning dataset from the same workflow that powers the live application.

## Product Description

Basis is organized as a routed decision workspace rather than a single dashboard. The interface connects macro events, affected assets, ranked signals, holdings, portfolio simulations, model health, market regime data, training data quality, and a lightweight in-app guide into one navigable application.

Core surfaces include:

| Surface | Purpose |
|---|---|
| Command | Decision overview with macro pulse, latest risks, model readiness, top signal, and current market state |
| Events | Searchable feed of classified macro events |
| Event Detail | Article context, classification reasoning, and mapped assets |
| Assets | Asset impact table with linked event and prediction context |
| Asset Detail | Related events, prices, signal history, holding fallback views, and accuracy |
| Signals | Ranked forecasts with confidence, horizon, expected move, source event, gate status, and model version |
| Prediction Detail | Decision case file with feature snapshot, gate status, and model contributors |
| Holdings | Watchlist and holding-level exposure, catalysts, risk score, macro view, and portfolio briefing |
| Portfolio | Signal-following equity curve, win rate, drawdown, and benchmark comparison |
| Diagnostics | Active-model health, drift, confidence PSI, current-model evaluation, training history, and validation metrics |
| Data Health | Dataset size, label balance, outcome coverage, and training-data signals |
| Guide | Compact walkthrough for using each workspace surface and interpreting the system |

The application is built to make the decision lifecycle visible: source event, structured interpretation, asset exposure, signal, actionability gate, holding impact, realized result, and model feedback all remain connected.

## Technical Breakdown

```text
backend/
  app/
    api/                     FastAPI routers
    services/                Ingestion, prediction, labeling, scoring, evaluation
    repositories/            Persistence helpers
    db/                      SQLAlchemy models and database initialization
    main.py                  FastAPI app, scheduler, background jobs

  ml/
    train_model.py           XGBoost training, calibration, validation
    model_store.py           Runtime model cache and hot reload

  scripts/
    generate_macro_seed_events.py
    seed_historical_training_data.py

frontend/
  src/
    App.tsx                  Routed decision workspace shell
    components/              Panels, charts, tables, detail views
    lib/                     API client and frontend data helpers
```

The backend is a FastAPI service with SQLAlchemy persistence, APScheduler background jobs, yfinance market data, OpenAI-based event classification, OpenAI embeddings for semantic deduplication, multi-provider financial news ingestion, actionability gates, and an XGBoost model pipeline. The frontend is a React 19, TypeScript, Vite, and TailwindCSS application with a light glassmorphism investor decision workspace.

### Ingestion

`ingestion_service.py` fetches market-relevant news, filters articles by financial keywords, classifies each article into a structured macro event, maps the event to affected assets, captures current market prices, persists the event record, and triggers signal generation.

News can be blended from Finnhub, Alpha Vantage, Marketaux, and existing feed sources. Provider metadata, ticker relevance, recency, source count, cross-provider consensus, and sentiment alignment are preserved as features so higher-quality article coverage can influence the prediction layer.

Incoming articles are deduplicated with exact content hashing and semantic similarity using OpenAI embeddings. Repeated coverage can reuse existing event records so the dataset stays centered on distinct market events.

### Event Classification

`event_classifier.py` converts raw article text into a structured schema:

| Field | Role |
|---|---|
| `event_type` | Normalized macro category |
| `affected_sectors` | Sectors connected to the event |
| `impact_direction` | Expected directional pressure |
| `confidence` | Classifier confidence |
| `severity` | Estimated market relevance |
| `reasoning` | Short explanation of the classification |

The classifier uses known event, sector, direction, and severity enums so downstream features remain consistent across ingestion, prediction, training, and evaluation.

### Asset Mapping

`mapping_service.py` links classified events to relevant ETFs and assets. One macro event can fan out into multiple asset-level predictions while preserving a shared event source.

### Feature Engineering

Each prediction stores a `FeatureSnapshot` with event, market, and derived features so each signal can be reviewed as a decision record.

| Group | Examples |
|---|---|
| Event features | Event type encoding, sentiment, severity, timestamp, embedding similarity, provider consensus |
| Market features | Price, recent returns, relative strength, volatility, asset class encoding |
| Derived features | Baseline score, event-market interactions, historical accuracy, regime features, news relevance |

Feature snapshots make signals auditable after generation. A model output can be tied back to the event, market state, and engineered feature set used at decision time.

### Prediction And Decision Gates

`prediction_pipeline.py` builds event, market, news-quality, and derived features for each mapped asset. `ml_scorer.py` supports trained XGBoost inference alongside a deterministic baseline scoring path for comparison and interpretability.

The signal layer applies investor-facing decision gates so the UI can separate actionable forecasts from blocked watch items. Blocked signals still show what the model sees, but include reasons such as gated asset, gated horizon, gated event type, model uncertainty, or live-health warmup.

Runtime model artifacts are loaded from `backend/ml/models/` and refreshed by `model_store.py` when updated artifacts are written.

### Training And Labeling

`ml/train_model.py` trains supervised classifiers from labeled signal outcomes. The training flow includes walk-forward cross-validation, horizon-aware model segments, XGBoost versus logistic regression comparison, ROC-AUC, accuracy, class balance tracking, Platt calibration, Brier score reporting, confidence bucket analysis, feature importance, and SHAP summary outputs.

Training data is exported directly from database records, so model training uses the same event, prediction, feature snapshot, gate, and outcome tables that power the application.

Historical bootstrapping creates seed macro events from BLS and FRED series, fetches market windows around each event timestamp, stores event-time feature snapshots, and labels 1-day, 3-day, and 5-day outcomes.

Live and historical labeling are handled through:

| Service | Role |
|---|---|
| `outcome_service.py` | Labels live predictions using current prices against event-time entry prices |
| `multi_horizon_service.py` | Labels 1-day, 3-day, and 5-day exits using historical closes |
| `model_evaluation_service.py` | Aggregates labeled outcomes into performance and readiness metrics |

### Model Evaluation

Basis evaluates labeled outcomes across the decision lifecycle:

| Metric Area | Examples |
|---|---|
| Accuracy | Overall accuracy, high-confidence accuracy, benchmark-relative accuracy |
| Returns | Average realized return, return buckets, portfolio simulation |
| Segments | Performance by asset, event type, horizon, and model version |
| Model comparison | XGBoost performance against deterministic decision baselines |
| Training history | Dataset size, train/test counts, ROC-AUC, calibration, label balance |
| Drift monitoring | Rolling accuracy, confidence distribution PSI, active-model warmup state |

Diagnostics default to the active model so live readiness is not confused with older baseline, historical seed, or retired model rows. Full-history evaluation remains available for audit context, while the main decision surfaces prioritize current actionable signals.

Every training run is stored in `training_runs` with validation metrics, model comparison results, top features, and label balance.

### Background Jobs

Basis runs continuous jobs through APScheduler:

| Job | Interval | Responsibility |
|---|---:|---|
| Ingestion | 15 minutes | Fetch news, classify events, map assets, generate predictions |
| Outcome computation | 1 hour | Label eligible predictions and check retraining thresholds |
| Signal evaluation | 1 hour | Evaluate signal and backtest records |

Automatic retraining is triggered when enough newly labeled samples have accumulated since the latest recorded training run.

### Market Intelligence

Basis includes live market regime features from VIX, SPY trend, and 10-year yield data. These regime encodings are injected into prediction features and surfaced in the UI so forecasts can be read alongside broader market conditions.

The holdings workspace accepts user-entered symbols and summarizes each holding with price context, related catalysts, source coverage, macro view, expected pressure, and portfolio-level risk. It is designed to show the decision context an investor would normally piece together manually from headlines, price action, and model signals.

The API exposes typed FastAPI endpoints across the intelligence workflow:

| Area | Examples |
|---|---|
| News and events | `/news`, `/combined`, `/events` |
| Market data | `/price/{symbol}`, `/market/history/{symbol}`, `/market-regime` |
| Predictions | `/predictions` |
| Outcomes | `/compute-outcomes`, `/compute-multi-horizon-outcomes` |
| Training data | `/export-training-data`, `/training-data/stats`, `/training-data/validation` |
| Model lifecycle | `/train-model`, `/training-history`, `/model-health`, `/model-evaluation` |
| Portfolio and signals | `/backtest/run`, `/backtest/summary`, `/backtest/portfolio`, `/signals/accuracy` |
| Watchlist intelligence | `/watchlist/impact` |

## Technical Highlights

- FastAPI service architecture with modular routers and dependency-injected database sessions
- SQLAlchemy data model for events, predictions, features, outcomes, training runs, and backtests
- APScheduler background jobs for continuous ingestion, labeling, and retraining checks
- OpenAI structured outputs for event classification
- OpenAI embeddings for semantic event deduplication
- Finnhub, Alpha Vantage, and Marketaux financial news integrations
- yfinance market data integration for live prices, historical windows, and regime context
- XGBoost classifier with calibration, walk-forward validation, and feature attribution
- Horizon-segmented model routing for 1-day, 3-day, and 5-day signals
- Deterministic baseline scoring for model comparison and interpretability
- Actionability gates for investor-facing decision quality control
- Multi-horizon outcome labeling at 1-day, 3-day, and 5-day windows
- Model-versus-baseline evaluation, active-model diagnostics, and drift detection
- React 19, TypeScript, Vite, and TailwindCSS frontend
- Routed light glassmorphism decision UI with signal drilldowns, holdings intelligence, model health panels, data-health checks, and an in-app guide
