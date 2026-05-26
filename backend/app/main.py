import logging
import threading
from typing import Any
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except ModuleNotFoundError:
    BackgroundScheduler = None

from app.api import backtest, combined, events, market, news, predictions, training, watchlist
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


def _run_compute_outcomes() -> None:
    from app.services.multi_horizon_service import compute_multi_horizon_outcomes
    from app.services.outcome_service import compute_outcomes
    db = SessionLocal()
    try:
        single_result = compute_outcomes(db=db)
        multi_result = compute_multi_horizon_outcomes(db=db)
        logger.info(
            (
                "Outcome computation complete: single=%d skipped=%d filtered=%d "
                "multi=%d multi_skipped=%d"
            ),
            single_result["computed"],
            single_result["skipped"],
            single_result["filtered"],
            multi_result["computed"],
            multi_result["skipped"],
        )
        if single_result["computed"] > 0 or multi_result["computed"] > 0:
            _run_auto_retrain_if_needed()
    except Exception as exc:
        logger.error("Outcome computation failed: %s", exc)
    finally:
        db.close()


def _run_auto_retrain_if_needed() -> None:
    from app.services.training_data_service import get_dataset_stats
    from app.services.training_run_service import should_auto_retrain
    db = SessionLocal()
    try:
        stats = get_dataset_stats(db)
        if should_auto_retrain(db, stats["num_samples"]):
            _run_train_model(triggered_by="auto")
    except Exception as exc:
        logger.error("Auto-retrain check failed: %s", exc)
    finally:
        db.close()


def _run_train_model(triggered_by: str = "auto") -> None:
    from ml.model_store import invalidate_cache
    from ml.train_model import result_to_dict, train_xgboost_from_payload
    from app.services.training_data_service import export_training_dataset
    from app.services.training_run_service import save_training_run
    db = SessionLocal()
    try:
        result = train_xgboost_from_payload(export_training_dataset(db, limit=50_000))
        invalidate_cache()
        payload = result_to_dict(result)
        save_training_run(db, payload, triggered_by=triggered_by)
        logger.info(
            "Auto-retrain complete: dataset_size=%d accuracy=%.4f roc_auc=%s",
            payload["dataset_size"],
            payload["accuracy"],
            payload.get("roc_auc"),
        )
    except Exception as exc:
        logger.error("Auto-retrain failed: %s", exc)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    init_db()

    scheduler: Any = None
    if BackgroundScheduler is not None:
        scheduler = BackgroundScheduler(daemon=True)
        scheduler.add_job(_run_ingestion, "interval", minutes=15, id="ingestion")
        scheduler.add_job(_run_signal_evaluation, "interval", hours=1, id="signal_eval")
        scheduler.add_job(_run_compute_outcomes, "interval", hours=1, id="compute_outcomes")
        scheduler.start()
    else:
        logger.warning("apscheduler is not installed; background jobs are disabled")

    # Warm caches on startup without blocking the server
    threading.Thread(target=_run_ingestion, daemon=True).start()

    yield

    if scheduler is not None:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="Basis",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        *settings.frontend_origins,
    ],
    allow_origin_regex=r"https://.*\.up\.railway\.app|http://(127\.0\.0\.1|localhost):\d+",
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
app.include_router(training.router)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Basis running"}
