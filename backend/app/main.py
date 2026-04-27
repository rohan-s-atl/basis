import logging
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import backtest, combined, events, market, news, predictions, watchlist
from app.api import signals
from app.db.init_db import init_db
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def _run_ingestion() -> None:
    from app.services.ingestion_service import ingest_events
    db = SessionLocal()
    try:
        results = ingest_events(db)
        logger.info("Background ingestion complete: %d events", len(results))
    except Exception as exc:
        logger.error("Background ingestion failed: %s", exc)
    finally:
        db.close()


def _run_signal_evaluation() -> None:
    from app.services.signal_evaluator import evaluate_signals
    db = SessionLocal()
    try:
        count = evaluate_signals(db)
        logger.info("Signal evaluation complete: %d signals evaluated", count)
    except Exception as exc:
        logger.error("Signal evaluation failed: %s", exc)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    init_db()

    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(_run_ingestion, "interval", minutes=15, id="ingestion")
    scheduler.add_job(_run_signal_evaluation, "interval", hours=1, id="signal_eval")
    scheduler.start()

    # Warm caches on startup without blocking the server
    threading.Thread(target=_run_ingestion, daemon=True).start()

    yield

    scheduler.shutdown(wait=False)


app = FastAPI(
    title="Macro Event Intelligence Engine",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_origin_regex=r"http://(127\.0\.0\.1|localhost):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(news.router)
app.include_router(market.router)
app.include_router(combined.router)
app.include_router(events.router)
app.include_router(watchlist.router)
app.include_router(signals.router)
app.include_router(backtest.router)
app.include_router(predictions.router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Macro Event Intelligence Engine running"}
