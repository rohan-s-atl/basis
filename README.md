# Basis

Basis is a full-stack AI macro intelligence system that turns real-world financial news and economic releases into structured market events, asset-level forecasts, labeled outcomes, model health signals, and an interactive research interface.

## Links

Basis: https://basis-intel.vercel.app/

The system is built around a continuous intelligence loop:

```text
News -> event classification -> asset mapping -> feature engineering
     -> prediction -> outcome labeling -> model evaluation -> retraining
```

Basis treats macro information as data. Inflation releases, rate decisions, geopolitical events, supply shocks, energy disruptions, earnings surprises, and broad risk-on or risk-off shifts are converted into a normalized event schema. Those events are mapped to affected assets, scored by a prediction pipeline, tracked against future market movement, and folded back into the model lifecycle.

## Why Basis Exists

Markets absorb information quickly, but the reasoning chain behind a market move is often scattered across articles, data releases, asset reactions, and model outputs. Basis brings that chain into one place.

The goal is to make macro intelligence inspectable. A user can move from a source event to its classification, mapped assets, prediction confidence, feature snapshot, model drivers, realized outcome, historical performance, market regime context, and retraining history without losing the thread.

Basis is designed for studying how macro events become machine-learning examples. Each event is not only a news item; it becomes a structured record with context, features, predictions, outcomes, and evaluation metadata. Over time, the system builds a supervised learning dataset from the same workflow that powers the live application.

## Product Description

Basis is organized as a routed research workspace rather than a single dashboard. The interface connects macro events, affected assets, predictions, portfolio simulations, model health, market regime data, and training data quality into one navigable application.

Core surfaces include:

| Surface | Purpose |
|---|---|
| Overview | Macro pulse, latest risks, model readiness, and current market state |
| Events | Searchable feed of classified macro events |
| Event Detail | Article context, classification reasoning, and mapped assets |
| Assets | Asset impact table with linked event and prediction context |
| Asset Detail | Related events, prices, signal history, and accuracy |
| Predictions | Ranked forecasts with confidence, horizon, source event, and model version |
| Prediction Detail | Forecast case file with feature snapshot and model contributors |
| Portfolio | Signal-following equity curve, win rate, drawdown, and benchmark comparison |
| ML Lab | Model health, drift, confidence PSI, training history, and validation metrics |
| Data Health | Dataset size, label balance, outcome coverage, and training-data signals |

The application is built to make the lifecycle visible: source event, structured interpretation, asset exposure, prediction, realized result, and model feedback all remain connected.

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
    App.tsx                  Routed intelligence app shell
    components/              Panels, charts, tables, detail views
    lib/                     API client and frontend data helpers
```

The backend is a FastAPI service with SQLAlchemy persistence, APScheduler background jobs, yfinance market data, OpenAI-based event classification, OpenAI embeddings for semantic deduplication, and an XGBoost model pipeline. The frontend is a React 19, TypeScript, Vite, and TailwindCSS application with a dark quant-terminal interface.

### Ingestion

`ingestion_service.py` fetches market-relevant news, filters articles by financial keywords, classifies each article into a structured macro event, maps the event to affected assets, captures current market prices, persists the event record, and triggers prediction generation.

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

Each prediction stores a `FeatureSnapshot` with event, market, and derived features.

| Group | Examples |
|---|---|
| Event features | Event type encoding, sentiment, severity, timestamp, embedding similarity |
| Market features | Price, recent returns, relative strength, volatility, asset class encoding |
| Derived features | Baseline score, event-market interactions, historical accuracy, regime features |

Feature snapshots make predictions auditable after generation. A model output can be tied back to the event, market state, and engineered feature set used at prediction time.

### Prediction

`prediction_pipeline.py` builds event, market, and derived features for each mapped asset. `ml_scorer.py` supports trained XGBoost inference alongside a deterministic baseline scoring path for comparison and interpretability.

Runtime model artifacts are loaded from `backend/ml/models/` and refreshed by `model_store.py` when updated artifacts are written.

### Training And Labeling

`ml/train_model.py` trains a supervised classifier from labeled prediction outcomes. The training flow includes walk-forward cross-validation, XGBoost versus logistic regression comparison, ROC-AUC, accuracy, class balance tracking, Platt calibration, Brier score reporting, confidence bucket analysis, feature importance, and SHAP summary outputs.

Training data is exported directly from database records, so model training uses the same event, prediction, feature snapshot, and outcome tables that power the application.

Historical bootstrapping creates seed macro events from BLS and FRED series, fetches market windows around each event timestamp, stores event-time feature snapshots, and labels 1-day, 3-day, and 5-day outcomes.

Live and historical labeling are handled through:

| Service | Role |
|---|---|
| `outcome_service.py` | Labels live predictions using current prices against event-time entry prices |
| `multi_horizon_service.py` | Labels 1-day, 3-day, and 5-day exits using historical closes |
| `model_evaluation_service.py` | Aggregates labeled outcomes into performance and readiness metrics |

### Model Evaluation

Basis evaluates labeled outcomes across the prediction lifecycle:

| Metric Area | Examples |
|---|---|
| Accuracy | Overall accuracy, high-confidence accuracy, benchmark-relative accuracy |
| Returns | Average realized return, return buckets, portfolio simulation |
| Segments | Performance by asset, event type, horizon, and model version |
| Model comparison | XGBoost performance against deterministic baselines |
| Training history | Dataset size, train/test counts, ROC-AUC, calibration, label balance |
| Drift monitoring | Rolling accuracy and confidence distribution PSI |

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
- yfinance market data integration for live prices, historical windows, and regime context
- XGBoost classifier with calibration, walk-forward validation, and feature attribution
- Deterministic baseline scoring for model comparison and interpretability
- Multi-horizon outcome labeling at 1-day, 3-day, and 5-day windows
- Model-versus-baseline evaluation and drift detection
- React 19, TypeScript, Vite, and TailwindCSS frontend
- Routed intelligence UI with prediction drilldowns, model health panels, and portfolio analytics
