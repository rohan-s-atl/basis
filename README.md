# Macro Event Intelligence Engine

An AI-powered financial intelligence backend that converts real-world news into structured macro events, impacted sectors, market direction, and related financial assets.

## What It Does

- Ingests macro and market news from NewsAPI, with mock fallback data for local development.
- Classifies raw articles into structured event intelligence using OpenAI.
- Maps events and sectors to relevant ETFs and market instruments using a configurable taxonomy.
- Fetches recent market prices with yfinance.
- Persists classified events and market snapshots for downstream analysis.
- Caches news, market prices, and classifications to reduce repeated external calls.
- Exposes clean FastAPI endpoints with typed response schemas.

## Backend Structure

```text
backend/
  app/
    main.py
    schemas.py
    api/
      news.py
      market.py
      combined.py
      events.py
    services/
      news_service.py
      market_service.py
      mapping_service.py
      event_classifier.py
      cache_service.py
      article_hash.py
    core/
      config.py
    data/
      asset_mapping.json
    db/
      init_db.py
      models.py
      session.py
    repositories/
      event_repository.py
  tests/
    test_combined_api.py
    test_event_classifier.py
    test_mapping_service.py
  requirements.txt
  .env.example
```

## Frontend Structure

```text
frontend/
  src/
    App.tsx
    main.tsx
    styles.css
    assets/
      market-intelligence-bg.png
  package.json
  vite.config.ts
  .env.example
```

## API Endpoints

```text
GET /              Health check
GET /news          Fetch latest macro news
GET /price/{sym}   Fetch market price and recent history
GET /combined      Return article + classification + mapped assets
GET /events        Return recently persisted event intelligence records
```

## Local Setup

Backend:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Frontend:

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Environment Variables

```env
NEWS_API_KEY=
NEWS_API_URL=https://newsapi.org/v2/everything
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
DATABASE_URL=sqlite:///./macro_event_intelligence.db
NEWS_CACHE_TTL_SECONDS=900
PRICE_CACHE_TTL_SECONDS=60
CLASSIFICATION_CACHE_TTL_SECONDS=86400
```

## Testing

```bash
cd backend
pytest
```

## Current Intelligence Flow

```text
News article
  -> event classification
  -> affected sector detection
  -> asset mapping
  -> market price enrichment
  -> persistence and snapshots
  -> structured API response
```

## Mapping Philosophy

Market assumptions are kept in `backend/app/data/asset_mapping.json` instead of being buried in Python code. The default taxonomy uses widely recognized ETF proxies:

- SPY for broad U.S. equity exposure, based on State Street's S&P 500 ETF documentation.
- XLE, XLK, and XLF for energy, technology, and financials sector exposure, based on the Select Sector SPDR ETF family.
- QQQ for Nasdaq-100 and large-cap growth sensitivity, based on Invesco QQQ documentation.
- TIP for inflation-linked bond exposure, based on iShares TIPS Bond ETF documentation.

These defaults are starting assumptions, not trading recommendations. The configuration should be reviewed as the product matures, especially before using it for portfolio decisions.

## Senior-Level Roadmap Ideas

- Add confidence scores and provenance for each classification.
- Store normalized events and market snapshots in Postgres.
- Add Redis caching for repeated price and news requests.
- Introduce background ingestion workers for scheduled news processing.
- Build an event-to-asset knowledge graph for explainable asset exposure.
- Add alerting for high-impact events that affect watchlist symbols.
- Track historical post-event price movement to evaluate signal quality.
- Add portfolio-aware impact analysis for user holdings.
- Add observability with structured logs, tracing, latency metrics, and model-call costs.
- Add evaluation tests for classification consistency across macro event categories.
