# Basis

Basis is a full-stack AI financial intelligence system that converts real-world news into structured macro events, maps them to affected assets, and runs a self-improving ML prediction pipeline with live market regime awareness.

## What It Does

Financial markets react to macro events: rate decisions, inflation data, geopolitical conflict, supply shocks, earnings, energy disruptions. Basis ingests news, classifies each article into a structured financial event using an LLM, links it to relevant ETFs and assets, generates directional predictions using a trained XGBoost model, tracks outcomes across multiple time horizons, and visualizes everything in a routed market-intelligence app.

```
News → LLM Classification → Asset Mapping → Feature Engineering → XGBoost Prediction
     → Outcome Tracking → Model Retraining → Drift Detection → Basis App
```

---

## Architecture

```
backend/
  app/
    api/                     FastAPI routers
      training.py            ML training endpoints and experiment tracking
      predictions.py         Forward prediction pipeline
      market.py              Prices, history, market regime
      events.py / combined.py / signals.py / backtest.py / watchlist.py
    services/
      prediction_pipeline.py  End-to-end prediction flow
      feature_service.py      Feature engineering and snapshot flattening
      outcome_service.py      Single-horizon outcome computation
      multi_horizon_service.py 1d / 3d / 5d outcome labeling via yfinance
      training_data_service.py Training dataset export and validation
      training_run_service.py  Experiment tracking, accuracy drift, confidence drift
      regime_service.py       Live market regime (VIX, SPY, 10Y yield)
      historical_market_service.py  Historical price windows for seed training
      historical_seed_service.py     CSV-to-ML-table historical seeding
      macro_seed_generator_service.py BLS/FRED macro series event generation
      embedding_service.py    OpenAI text embeddings and semantic deduplication
      ml_scorer.py            XGBoost inference with SHAP and baseline fallback
      baseline_scoring.py     Rule-based fallback scorer
      event_classifier.py     LLM-based event classification
      ingestion_service.py    News ingestion and deduplication
      mapping_service.py      Event-to-asset mapping
      market_service.py       yfinance price and OHLCV fetching
    db/
      models.py               SQLAlchemy models (Event, Prediction, Outcome, etc.)
      init_db.py              Schema migrations and table creation
    main.py                   FastAPI app, scheduler, background jobs
  ml/
    train_model.py            XGBoost training pipeline
    model_store.py            Hot-reload model cache
    models/                   Local generated model artifacts (git-ignored)
  scripts/
    generate_macro_seed_events.py    Build historical event CSVs from BLS/FRED
    seed_historical_training_data.py Seed historical events, prices, labels

frontend/
  src/
    App.tsx                   Routed app shell and page/detail views
    components/
      MLIntelligencePanel.tsx  ML health, regime, feature importance, training history
      PredictionsPanel.tsx     Forward predictions
      BacktestPanel.tsx        Signal backtest analytics and portfolio curve
      SectorHeatmap.tsx        Sector pressure heatmap
      MarketBreadth.tsx        Breadth metrics and watchlist impact
      SignalAccuracy.tsx       Historical signal accuracy
      WatchlistPanel.tsx       Watchlist impact controls
      AlertQueue.tsx           Alert rule queue
      ExpandModal.tsx          Expandable detail surfaces
    lib/
      api.ts                  Typed API client with fallback and timeout handling
      intelligence.ts         Frontend data helpers
    types.ts                  Shared TypeScript types
```

---

## ML Pipeline

### Feature Engineering

Every prediction persists a `FeatureSnapshot` with three namespaced feature groups:

**Event features** — derived from LLM classification output:
- `event_type_encoded`, `sentiment`, `severity`, `event_timestamp_unix`
- `text_embedding_norm`, `text_embedding_pos_sim`, `text_embedding_neg_sim`, `text_embedding_vol_sim` (from OpenAI `text-embedding-3-small`)

**Market features** — derived from yfinance price data at prediction time:
- `price`, `return_1d`, `return_5d`, `return_10d`, `sector_return_5d`, `sector_return_10d`
- `relative_strength_5d`, `rolling_volatility`, `asset_encoded`, `asset_class_encoded`

**Derived features** — cross-feature interactions and live regime data:
- `baseline_score`, `sentiment_x_severity`, `sentiment_x_sector_sensitivity`, `severity_x_volatility`
- `event_type_asset_class_interaction`
- `historical_accuracy_of_event_type`, `rolling_accuracy_of_asset_predictions`
- `event_asset_avg_return`, `event_asset_accuracy`
- `vix_level`, `vix_regime_encoded`, `spy_trend`, `rate_level`, `market_regime_encoded`
- `benchmark_price` for benchmark-relative outcome evaluation when available

### Model Training

The training pipeline (`ml/train_model.py`) runs on demand or triggers automatically:

1. **Walk-forward cross-validation** — expanding window splits that respect temporal ordering, no future data leakage
2. **Model comparison** — XGBoost vs logistic regression (StandardScaler + LogisticRegression pipeline) on test accuracy and ROC-AUC
3. **SHAP explainability** — TreeExplainer mean absolute SHAP values per feature on the test set, plus per-prediction contributors for stored ML forecasts
4. **Platt calibration** — `CalibratedClassifierCV(cv=3)` wraps the raw XGBoost to produce better-calibrated probabilities; Brier score improvement tracked before and after
5. **Confidence bucket analysis** — accuracy broken down by prediction confidence band (0.5–0.6, 0.6–0.7, 0.7–0.8, 0.8+)

Manual and automatic retraining use the database export payload directly, so retraining does not depend on the server making HTTP calls back into itself. The CLI training path can still load a dataset from `/export-training-data`.

Artifacts are written at runtime to `backend/ml/models/` as `xgboost_model.json`, `calibrated_model.pkl`, and `feature_names.json`. That directory is git-ignored because these are generated local artifacts. The model store uses mtime-based hot-reload so new models are picked up without restarting the server.

API and background training export the dataset in-process instead of calling the backend through a hardcoded localhost URL. CLI training also records a `training_runs` entry when the configured database is writable.

### Historical Training Bootstrap

The project can bootstrap the ML dataset with historical macro events before waiting for live predictions to mature. This is handled in two stages:

1. **Macro event generation** (`scripts/generate_macro_seed_events.py`) converts official macro time series into seedable event rows.
   - BLS source, no API key required: CPI (`CUSR0000SA0`) and unemployment (`LNS14000000`)
   - FRED source, requires `FRED_API_KEY`: CPI (`CPIAUCSL`), unemployment (`UNRATE`), federal funds rate (`FEDFUNDS`)
   - CPI and employment rows use actual FRED/BLS release-calendar dates when available instead of rough month-based estimates.
2. **Historical seeding** (`scripts/seed_historical_training_data.py`) reads the generated CSV, fetches yfinance historical ETF prices around each event timestamp, creates `Event`, `Prediction`, and `FeatureSnapshot` records, and labels 1d/3d/5d outcomes in `MultiHorizonOutcome`.

Historical feature snapshots are built only from event-time information and pre-event/as-of market data. Future closes are used only as labels, which keeps the training flow leakage-safe. The seeder also backfills historical market regime features as of the event date: VIX level/regime, SPY 20-day trend, 10Y yield level, and encoded risk regime.

Current seeded baseline after BLS + FRED generation:

| Metric | Value |
|---|---:|
| Events | 112 |
| Predictions | 489 |
| Multi-horizon labeled outcomes | 862 |
| Training samples | 920 |
| Label balance | 51.85% positive / 48.15% negative |
| Features | 35 |
| Latest trained accuracy | 48.37% |
| Latest ROC-AUC | 0.5039 |
| Walk-forward CV mean | 53.33% |

### Outcome Labeling

**Single-horizon** (`outcome_service.py`): Fetches current price vs entry price, applies direction label, filters noise below a configurable return threshold, assigns a return bucket (`flat`, `small`, `medium`, `large`), and stores benchmark-relative fields when an event-time SPY benchmark price exists.

Noise-filtered predictions stay pending instead of becoming misleading flat labels, so later runs can retry them when price movement is large enough.

**Multi-horizon** (`multi_horizon_service.py`): Labels each prediction at 1-day, 3-day, and 5-day horizons using yfinance historical closes. Predictions are grouped by symbol to minimize API calls. A minimum-age filter (8 days) ensures exit prices exist before attempting labeling. Noise-filtered multi-horizon rows are skipped rather than persisted with null labels.

Training export combines single-horizon and multi-horizon labeled rows in chronological order. Multi-horizon labels are preferred over single-horizon for the same prediction, while single-horizon live rows are still included when no multi-horizon label exists.

### Model Evaluation

`GET /model-evaluation` is the main quality gate for the project. It combines single- and multi-horizon labels without double-counting, then reports:

- overall accuracy and average realized return
- high-confidence-only performance
- benchmark-relative accuracy when SPY benchmark returns are available
- model vs simple baselines: always-up, always-down, event-sentiment, SPY-trend
- performance by asset, event type, horizon, return bucket, and model version
- data quality checks: validation issues, duplicate event groups, missing snapshots, unlabeled predictions
- recommendations that explain whether signals should remain gated

### Experiment Tracking

Every training run is persisted to the `training_runs` table:

| Field | What it captures |
|---|---|
| `trained_at` | Timestamp of the run |
| `triggered_by` | `manual` or `auto` |
| `dataset_size` | Labeled samples at training time |
| `accuracy`, `roc_auc` | Held-out test set metrics |
| `brier_score_calibrated`, `brier_improvement` | Calibration quality |
| `walk_forward_mean`, `walk_forward_std` | CV stability |
| `xgboost_roc_auc`, `logistic_roc_auc` | Model comparison |
| `top_features` | SHAP/gain feature importance snapshot |
| `label_balance` | Class distribution at training time |

### Auto-Retraining

The hourly outcome computation job checks whether the labeled dataset has grown by 25+ samples since the last training run. If so, retraining fires automatically in the background, the new model artifacts replace the old ones, and the run is logged. No manual intervention needed.

### Drift Detection

`GET /model-health` computes rolling accuracy over the last 30 labeled predictions and compares it to the peak accuracy from the training history. It also compares recent prediction confidence against the training-time confidence distribution with Population Stability Index (PSI). If either accuracy drift or confidence-distribution drift crosses threshold, `drift_detected: true` is returned and Basis surfaces an alert.

### Semantic Deduplication

Incoming news articles are deduplicated in two stages:

1. **Exact match** — timestamp + raw text + source hash (zero API cost)
2. **Semantic similarity** — OpenAI `text-embedding-3-small` embedding compared against all events from the last 48 hours using cosine similarity (threshold 0.92). Duplicate articles reuse the existing event rather than creating a new one.

### Market Regime Detection

`regime_service.py` fetches live VIX, SPY 20-day trend, and 10Y yield (^TNX) via yfinance and encodes them into regime features injected into every prediction's derived feature set:

- `vix_regime_encoded`: 0=calm (<15), 1=normal (15–25), 2=stressed (25–35), 3=crisis (>35)
- `market_regime_encoded`: 0=risk-on, 1=neutral, 2=risk-off

Regime data is cached for 1 hour.

---

## Background Jobs

Three APScheduler jobs run continuously:

| Job | Interval | What it does |
|---|---|---|
| `_run_ingestion` | Every 15 min | Fetches news, classifies events, runs prediction pipeline |
| `_run_compute_outcomes` | Every 1 hour | Labels single- and multi-horizon outcomes, checks auto-retrain threshold |
| `_run_signal_evaluation` | Every 1 hour | Evaluates legacy signal backtest records |

---

## Basis App

Built with React 19, TypeScript, Vite, and TailwindCSS. No UI component library — entirely custom dark quant-terminal aesthetic.

Basis is organized as a routed intelligence app rather than one dense cockpit. Pages are shareable through hash routes and most rows/cards drill into deeper views. The left rail carries the Basis brand, live system state, a compact pulse readout, model-readiness progress, market regime, and a dynamic next-action prompt.

| Page | Purpose |
|---|---|
| Overview | Executive summary, macro risks, latest high-impact events, compact portfolio/model/data status |
| Events | Searchable event feed with sector, asset, severity, direction, and recency context |
| Event Detail | Article/source detail, LLM classification, reasoning, affected assets, related predictions |
| Assets | Asset impact table with linked event count, price context, and prediction direction |
| Asset Detail | Price chart, linked events, signal history, accuracy, exposure, latest model drivers |
| Predictions | Ranked forecasts with edge score, weak-signal filtering, confidence, expected move, horizon, event source, and model version |
| Prediction Detail | Forecast case file with SHAP contributors, feature snapshot, regime, outcome status, similar signals |
| Portfolio | Signal-following equity curve vs SPY, win rate, drawdown, ranked signals, event-type breakdowns |
| ML Lab | Model health, drift, confidence PSI, model-vs-baseline evaluation, high-confidence performance, feature importance, training history, validation, retrain controls |
| Data Health | Pipeline readiness, sample counts, class balance, labeling state, validation issues, API status |

Clickable surfaces include event cards, article titles, asset symbols, prediction rows, SHAP bars, training runs, market regime status, portfolio chart points, heatmap sectors, and backtest/signal rows.

---

## API Reference

```
GET  /                                Health check
GET  /news                            Latest news articles
GET  /combined                        News + LLM classification + mapped assets
GET  /events                          Persisted event records
GET  /price/{symbol}                  Current price and recent close history
GET  /market/history/{symbol}         Historical OHLCV data
GET  /market-regime                   Live VIX, SPY trend, 10Y yield, regime encoding
POST /watchlist/impact                Portfolio/watchlist impact analysis
GET  /signals/accuracy                Historical signal accuracy by event type
POST /backtest/run                    Run signal backtest
GET  /backtest/summary                Backtest analytics and top signals
GET  /backtest/portfolio              Simulated signal portfolio curve vs SPY
GET  /predictions                     Forward-looking asset predictions with weak-signal filtering
POST /compute-outcomes                Label outcomes for unlabeled predictions
POST /compute-multi-horizon-outcomes  Label 1d/3d/5d outcomes
GET  /export-training-data            Full labeled feature matrix for ML training
GET  /training-data/split             Chronological train/test split
GET  /training-data/stats             Dataset sample count and class balance
GET  /training-data/validation        Dataset quality checks and issue list
GET  /training-data/confidence-buckets Accuracy by model confidence band
POST /train-model                     Train XGBoost, save artifacts, log run
GET  /training-history                All training runs with full metrics
GET  /model-health                    Rolling accuracy, peak accuracy, drift flag
GET  /model-evaluation                Baselines, grouped performance, data quality
```

---

## Local Setup

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
cp .env.example .env         # macOS/Linux
# copy .env.example .env     # Windows
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API docs: `http://127.0.0.1:8000/docs`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Basis app: `http://127.0.0.1:5173`

### Train the Model

After the backend is running and events have been ingested:

```bash
# Label outcomes (run once to seed labels)
curl -X POST http://127.0.0.1:8000/compute-outcomes

# Train XGBoost on labeled data
curl -X POST http://127.0.0.1:8000/train-model
```

Or use the **Retrain** button in the ML Lab page of Basis.

### Seed Historical Training Data

Use the project virtualenv for the historical scripts so the same dependency set and SQLite database path are used as the backend.

Generate a no-key BLS historical macro seed file:

```bash
cd backend
..\.venv\Scripts\python.exe scripts\generate_macro_seed_events.py --source bls --start 2020-01-01 --end 2024-12-31 --output app\data\historical_events.bls.generated.csv
```

Generate a FRED historical macro seed file after setting `FRED_API_KEY`:

```bash
..\.venv\Scripts\python.exe scripts\generate_macro_seed_events.py --source fred --start 2020-01-01 --end 2024-12-31 --output app\data\historical_events.fred.generated.csv
```

Seed generated events into the ML tables:

```bash
..\.venv\Scripts\python.exe scripts\seed_historical_training_data.py --file app\data\historical_events.bls.generated.csv
..\.venv\Scripts\python.exe scripts\seed_historical_training_data.py --file app\data\historical_events.fred.generated.csv
```

Use `--refresh-existing` when regenerated calendars or historical regime features should rebuild existing seeded snapshots and labels:

```bash
..\.venv\Scripts\python.exe scripts\seed_historical_training_data.py --file app\data\historical_events.bls.generated.csv --refresh-existing
..\.venv\Scripts\python.exe scripts\seed_historical_training_data.py --file app\data\historical_events.fred.generated.csv --refresh-existing
```

Then start the backend and retrain:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
curl -X POST "http://127.0.0.1:8000/train-model?limit=50000"
```

---

## Environment Variables

**Backend** (`.env`):

```env
NEWS_API_KEY=
NEWS_API_URL=https://newsapi.org/v2/everything
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
FRED_API_KEY=                         # optional; enables FRED historical seed generation
DATABASE_URL=                          # leave blank for local SQLite
OUTCOME_NOISE_THRESHOLD=0.0001         # minimum return to count as a label
NEWS_CACHE_TTL_SECONDS=900
PRICE_CACHE_TTL_SECONDS=60
CLASSIFICATION_CACHE_TTL_SECONDS=86400
```

**Frontend** (`.env`):

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## What This Demonstrates

- **Full-stack ML product design** — end-to-end pipeline from raw news to trained model to live UI
- **Feature engineering** — event, market, derived, interaction, regime, and embedding features
- **Historical data bootstrapping** — official BLS/FRED macro series converted into event-conditioned training rows
- **XGBoost with SHAP** — tree-based classification with feature attribution and Platt calibration
- **Time-series ML discipline** — walk-forward CV, no data leakage, chronological splits
- **Multi-horizon labeling** — outcome labels at 1d, 3d, 5d horizons from historical market data
- **Autonomous ML lifecycle** — experiment tracking, auto-retraining triggers, drift detection
- **Portfolio simulation** — signal-following equity curve compared with a normalized SPY benchmark
- **LLM integration** — structured extraction via OpenAI + semantic deduplication via embeddings
- **Market regime awareness** — live VIX/SPY/yield encoding injected as ML features
- **FastAPI service architecture** — modular routers, dependency injection, typed schemas
- **React app** — professional dark terminal UI with routed pages, live data, sparklines, and interactivity
- **Background job scheduling** — APScheduler for continuous ingestion, labeling, and model health checks

## Current Boundaries

- Model metrics are only as credible as the labeled dataset. The recommended target is 300+ non-noise labeled samples with `OUTCOME_NOISE_THRESHOLD=0.0001`.
- Multi-horizon labels require predictions old enough for the 1d/3d/5d exits to exist.
- Runtime model artifacts are generated locally under `backend/ml/models/` and are intentionally ignored by git.
- Basis uses request/refresh polling, not WebSockets.
- The heatmap is a macro/sector pressure view, not a signal-correlation matrix.

---

## Disclaimer

This project is for software engineering and research demonstration purposes only. It is not financial advice, investment advice, or a trading system.
