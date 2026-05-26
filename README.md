# Basis

Basis is a full-stack AI financial intelligence system for turning real-world macro news into structured market events, asset-level predictions, labeled outcomes, model health signals, and an interactive research interface.

It is built around a continuous intelligence loop:

```text
News -> LLM classification -> asset mapping -> feature engineering
     -> prediction -> outcome labeling -> model evaluation -> retraining
```

Basis is not a trading bot. It is a research-grade system for studying how macro events, market regimes, and asset reactions can be converted into a supervised learning pipeline.

---

## Product Concept

Financial markets react to events: inflation releases, rate decisions, geopolitical conflict, supply shocks, earnings surprises, energy disruptions, and broad risk-on/risk-off shifts. Basis treats each article or macro release as a candidate event, classifies it into a normalized schema, maps it to affected assets, creates model features, and tracks whether the resulting forecast was directionally correct over time.

The goal is to make the model lifecycle visible. Basis shows the source event, classification reasoning, mapped assets, prediction confidence, feature snapshot, SHAP-style model drivers, realized outcome, historical accuracy, model drift, and retraining history in one connected application.

---

## System Architecture

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

The backend is a FastAPI service with SQLAlchemy persistence, APScheduler background jobs, yfinance market data, OpenAI-based event classification, and an XGBoost model pipeline. The frontend is a React 19 + TypeScript + Vite app with a dark quant-terminal interface.

---

## Backend Pipeline

### Ingestion

`ingestion_service.py` fetches market-relevant news, filters articles by financial keywords, classifies each article into a structured macro event, maps it to affected assets, captures current market prices, persists an event record, and triggers the prediction pipeline.

Incoming articles are deduplicated through exact content hashing and semantic similarity using OpenAI embeddings. Duplicate articles reuse existing event records instead of creating noisy repeated samples.

### Event Classification

`event_classifier.py` converts raw article text into a strict JSON schema:

- `event_type`
- `affected_sectors`
- `impact_direction`
- `confidence`
- `severity`
- `reasoning`

The classifier is intentionally constrained to known event, sector, direction, and severity enums so downstream features stay stable.

### Asset Mapping

`mapping_service.py` links classified events to relevant ETFs and assets. This lets one macro event fan out into multiple asset-level predictions while preserving a shared event source.

### Feature Engineering

Each prediction stores a `FeatureSnapshot` with three feature groups:

| Group | Examples |
|---|---|
| Event features | event type encoding, sentiment, severity, timestamp, embedding similarity |
| Market features | price, recent returns, relative strength, volatility, asset class encoding |
| Derived features | baseline score, event-market interactions, historical accuracy, regime features |

This makes predictions auditable after the fact. A model output can be tied back to the exact event, market state, and engineered feature set used at prediction time.

### Prediction

`prediction_pipeline.py` builds event, market, and derived features for each mapped asset. `ml_scorer.py` attempts model inference with the trained XGBoost artifacts and falls back to a rule-based baseline scorer when a trained model is unavailable.

Runtime model artifacts are loaded from `backend/ml/models/` and refreshed by `model_store.py` when new artifacts are written.

---

## ML System

### Training

`ml/train_model.py` trains a supervised classifier from labeled prediction outcomes. The training flow includes:

- Walk-forward cross-validation using expanding time windows
- XGBoost versus logistic regression comparison
- ROC-AUC, accuracy, and class balance tracking
- Platt calibration for probability quality
- Brier score and calibration improvement reporting
- Confidence bucket analysis
- Feature importance and SHAP summary outputs

Training data is exported directly from database records, so model training uses the same event, prediction, feature snapshot, and outcome tables that power the application.

### Historical Bootstrapping

Basis can seed its ML tables from historical macro events generated from BLS and FRED series. The seeding flow creates historical events, fetches market windows around each event timestamp, stores feature snapshots using only event-time information, and labels 1-day, 3-day, and 5-day future outcomes.

Future prices are used only as labels, not as model inputs, to avoid time leakage.

### Outcome Labeling

Basis tracks both single-horizon and multi-horizon outcomes:

- `outcome_service.py` labels live predictions using current prices against event-time entry prices.
- `multi_horizon_service.py` labels 1d, 3d, and 5d exits using historical closes.

Noise filtering prevents tiny price moves from becoming misleading labels. Low-magnitude moves remain pending or skipped depending on the labeling path.

### Evaluation

`model_evaluation_service.py` combines labeled outcomes without double-counting and reports:

- Overall accuracy
- Average realized return
- High-confidence-only performance
- Benchmark-relative accuracy
- Model performance versus simple baselines
- Performance by asset, event type, horizon, return bucket, and model version
- Data quality checks
- Recommendations for whether signals should remain gated

### Experiment Tracking

Every training run is persisted to `training_runs`, including dataset size, train/test counts, accuracy, ROC-AUC, calibration metrics, walk-forward CV metrics, model comparison results, top features, and label balance.

### Drift Detection

`training_run_service.py` computes rolling model health by comparing recent labeled accuracy and prediction confidence distribution against the training history. Confidence drift uses Population Stability Index, and accuracy drift compares recent performance against the latest training benchmark.

---

## Background Jobs

Basis runs continuous jobs through APScheduler:

| Job | Interval | Responsibility |
|---|---:|---|
| Ingestion | 15 minutes | Fetch news, classify events, map assets, generate predictions |
| Outcome computation | 1 hour | Label eligible predictions and check retraining threshold |
| Signal evaluation | 1 hour | Evaluate legacy signal backtest records |

Automatic retraining is triggered when the labeled dataset grows by at least 25 samples since the last recorded training run.

---

## Market Intelligence Features

Basis includes live market regime features from VIX, SPY trend, and 10-year yield data. These regime encodings are injected into prediction features and surfaced in the UI.

The app includes:

| Surface | Purpose |
|---|---|
| Overview | Macro pulse, latest risks, model readiness, market state |
| Events | Searchable feed of classified macro events |
| Event Detail | Article context, classification reasoning, mapped assets |
| Assets | Asset impact table with event and prediction context |
| Asset Detail | Linked events, prices, signal history, accuracy |
| Predictions | Ranked forecasts with confidence, horizon, source event, model version |
| Prediction Detail | Forecast case file with feature snapshot and model contributors |
| Portfolio | Signal-following equity curve, win rate, drawdown, benchmark comparison |
| ML Lab | Model health, drift, confidence PSI, training history, validation |
| Data Health | Dataset size, label balance, pending outcomes, validation issues |

The interface is designed as a routed research application rather than a single dashboard, so users can move from a macro event to the affected asset, prediction, outcome, and model explanation.

---

## API Surface

Basis exposes typed FastAPI endpoints across the main intelligence workflow:

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

---

## Technical Highlights

- FastAPI service architecture with modular routers and dependency-injected database sessions
- SQLAlchemy data model for events, predictions, features, outcomes, training runs, and backtests
- APScheduler background jobs for continuous ingestion, labeling, and retraining checks
- OpenAI structured outputs for event classification
- OpenAI embeddings for semantic event deduplication
- yfinance market data integration for live prices, historical windows, and regime context
- XGBoost classifier with calibration, walk-forward validation, and feature attribution
- Rule-based baseline scoring fallback when trained artifacts are unavailable
- Multi-horizon outcome labeling at 1d, 3d, and 5d
- Model-vs-baseline evaluation and drift detection
- React 19, TypeScript, Vite, and TailwindCSS frontend
- Routed intelligence UI with prediction drilldowns, model health panels, and portfolio analytics

---

## Current Boundaries

- Model quality depends on the amount and quality of labeled outcome data.
- Multi-horizon labels require predictions old enough for exit prices to exist.
- The model is a directional research model, not an execution engine.
- Market data availability and API reliability can affect ingestion, labeling, and backtests.
- The application uses polling and refresh patterns rather than WebSockets.
- Runtime model artifacts are generated outputs and are intentionally not versioned with source code.

---

## Disclaimer

Basis is for software engineering, ML systems, and financial research demonstration purposes only. It is not financial advice, investment advice, or a trading system.
