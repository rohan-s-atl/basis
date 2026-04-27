# Macro Event Intelligence Engine

Macro Event Intelligence Engine is a full-stack AI financial intelligence system that converts real-world news into structured macro events, maps those events to affected sectors and financial assets, and evaluates whether the resulting signals are useful over time.

The project is designed as a production-style resume project: it combines news ingestion, LLM event classification, market data enrichment, explainable asset mapping, portfolio/watchlist analysis, historical price charts, signal backtesting, and an interpretable prediction layer.

## Core Idea

Financial markets react to macro events: inflation data, rate decisions, supply shocks, geopolitical conflicts, earnings, energy disruptions, and broad risk sentiment. This system ingests news, classifies each article into a macro-financial event, links the event to relevant ETFs/assets, and presents the result in a decision-focused market cockpit.

The goal is not to make trading recommendations. The goal is to build a measurable intelligence pipeline:

```text
News -> Event Classification -> Asset Mapping -> Market Data -> Prediction -> Backtest -> Model-Ready Features
```

## Major Features

- FastAPI backend with modular services, typed schemas, and test coverage.
- React + Vite + Tailwind frontend with a dark quant-style cockpit.
- News ingestion from NewsAPI with local fallback behavior.
- OpenAI-powered event classification.
- Rule/config-based macro asset mapping.
- yfinance market data for prices and historical OHLCV charts.
- SQLite persistence for classified events, market snapshots, and backtest records.
- Watchlist impact analysis for user-defined symbols.
- Event clustering and macro situation grouping.
- Forward-looking prediction engine with probability, expected move, horizon, drivers, and scenario cases.
- Backtesting layer that evaluates event-to-asset signals against market movement.
- Interpretable feature-based ML scoring layer for ranking signal quality.
- Signal accuracy and outcome tracking.

## System Architecture

```text
backend/
  app/
    main.py
    schemas.py
    api/
      backtest.py
      combined.py
      events.py
      market.py
      news.py
      predictions.py
      signals.py
      watchlist.py
    services/
      event_classifier.py
      ingestion_service.py
      mapping_service.py
      market_service.py
      news_service.py
      prediction_service.py
      backtest_service.py
      feature_service.py
      signal_evaluator.py
      watchlist_service.py
    repositories/
      event_repository.py
      backtest_repository.py
    db/
      models.py
      session.py
      init_db.py
    data/
      asset_mapping.json
      watchlist_symbol_profiles.json
  tests/
  requirements.txt

frontend/
  src/
    App.tsx
    components/
    lib/
    types.ts
    styles.css
  package.json
```

## Backend Capabilities

The backend exposes the intelligence pipeline through FastAPI.

### News and Event Intelligence

- Fetches recent macro/market news.
- Uses OpenAI to classify articles into structured event intelligence:
  - event type
  - affected sectors
  - impact direction
  - confidence
  - severity
  - reasoning
- Stores classified events for later analysis.

### Market Data

- Fetches current prices with yfinance.
- Fetches historical OHLCV market data for charting.
- Caches repeated price/history requests.
- Persists market snapshots linked to event records.

### Asset Mapping

Asset mappings are kept in JSON configuration rather than hardcoded throughout the codebase.

Examples:

- Energy/geopolitical shocks can map to `XLE`, `USO`, `DBC`, `GLD`.
- Inflation and rates can map to `TIP`, `TLT`, `SPY`, `QQQ`.
- Technology events can map to `QQQ`, `XLK`.
- Financial events can map to `XLF`.

This makes the system easier to expand as market assumptions evolve.

## AI and ML Layer

The system currently has two model-related layers.

### 1. LLM Event Classification

OpenAI is used to convert unstructured news text into structured macro-financial intelligence.

Example output:

```json
{
  "event_type": "geopolitical_conflict",
  "affected_sectors": ["energy", "commodities"],
  "impact_direction": "negative",
  "confidence": 0.86,
  "severity": "high",
  "reasoning": "A disruption near a major energy shipping route can pressure oil supply and raise volatility."
}
```

### 2. Interpretable Signal Scoring

The project includes a lightweight, explainable scoring model that behaves like a logistic model. It ranks event-to-asset signals using features such as:

- event confidence
- severity score
- direction score
- sector count
- asset specificity
- volatility proxy
- macro ETF / sector ETF flags

This is not pretending to be a fully trained model yet. It is a model-ready baseline that can later be replaced by a trained classifier once enough labeled backtest outcomes exist.

## Backtesting

The backtesting layer evaluates whether event-to-asset signals moved in the expected direction.

It stores:

- symbol
- event type
- expected direction
- entry price
- exit price
- return percentage
- correctness
- confidence
- severity
- ML score
- feature vector

Flat moves are treated as `flat`, not failed predictions. Accuracy is calculated only on actionable moves above a small return threshold.

## Prediction Engine

The prediction service generates forward-looking asset forecasts from current events.

Each prediction includes:

- symbol
- event title
- event type
- impacted sectors
- predicted direction
- probability score
- expected move range
- forecast horizon
- bull/base/bear scenarios
- model drivers
- model version

This creates a clean interface that can later be powered by a trained ML model.

## Frontend Cockpit

The frontend is a professional dark quant dashboard built with React, Vite, and Tailwind CSS.

Main sections:

- **Event Feed**: searchable macro signal stream.
- **Decision Table**: mapped assets, prices, impact direction, confidence, and mini charts.
- **Asset Detail Modal**: Yahoo-style historical chart with OHLCV data and macro readthrough.
- **Sector Heatmap**: sector pressure counts and filters.
- **Intelligence Detail**: selected event reasoning, linked assets, and suggested actions.
- **Workbench**:
  - Breadth
  - Alerts
  - Predict
  - Backtest
  - Accuracy

The UI is designed to feel like a hedge fund / Bloomberg / Palantir-style internal tool rather than a marketing dashboard.

## API Overview

```text
GET  /                         Health check
GET  /news                     Latest news articles
GET  /combined                 News + classification + mapped assets
GET  /events                   Persisted event records
GET  /price/{symbol}           Current price and recent close history
GET  /market/history/{symbol}  Historical OHLCV data
POST /watchlist/impact         Portfolio/watchlist impact analysis
GET  /signals/accuracy         Historical signal accuracy
POST /backtest/run             Run signal backtest
GET  /backtest/summary         Backtest analytics
GET  /predictions              Forward-looking asset predictions
```

## Local Setup

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

Backend docs:

```text
http://127.0.0.1:8001/docs
```

### Frontend

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Frontend:

```text
http://127.0.0.1:5173
```

## Environment Variables

Backend:

```env
NEWS_API_KEY=
NEWS_API_URL=https://newsapi.org/v2/everything
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
DATABASE_URL=
NEWS_CACHE_TTL_SECONDS=900
PRICE_CACHE_TTL_SECONDS=60
CLASSIFICATION_CACHE_TTL_SECONDS=86400
```

Frontend:

```env
VITE_API_BASE_URL=http://127.0.0.1:8001
```

If `DATABASE_URL` is left blank, the backend defaults to local SQLite storage.

## Testing

Backend:

```bash
cd backend
python -m pytest
```

Frontend:

```bash
cd frontend
npm run build
```

## Current Test Coverage

The backend test suite covers:

- combined endpoint behavior
- event classifier fallback/validation
- asset mapping logic
- market history endpoint
- watchlist impact service
- backtest service
- prediction service

## Resume Highlights

This project demonstrates:

- Full-stack product architecture.
- FastAPI service design.
- LLM-based structured extraction.
- Financial data enrichment.
- Event-driven signal generation.
- Feature engineering for ML readiness.
- Interpretable model scoring.
- Backtesting and outcome evaluation.
- Frontend data visualization.
- Clean separation between services, repositories, schemas, and UI components.

## Roadmap

Potential next steps:

- Train a real supervised model on accumulated backtest records.
- Add feature importance and model evaluation metrics.
- Add Postgres support for production deployment.
- Add scheduled background jobs with a real task queue.
- Add authentication and saved user watchlists.
- Add alert delivery through email, Slack, or web push.
- Add source reliability scoring for news providers.
- Add event deduplication using embeddings.
- Add model versioning and experiment tracking.
- Deploy backend and frontend to cloud infrastructure.

## Disclaimer

This project is for software engineering and research demonstration purposes. It is not financial advice, investment advice, or a trading system.
