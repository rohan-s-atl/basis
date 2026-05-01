import json
import logging

from sqlalchemy.orm import Session

from app.repositories.event_repository import create_market_snapshot, upsert_event_record
from app.services.article_hash import build_article_hash
from app.services.event_classifier import classify_event
from app.services.mapping_service import map_article_to_assets
from app.services.market_service import fetch_price
from app.services.news_service import fetch_news
from app.services.prediction_pipeline import run_prediction_pipeline

logger = logging.getLogger(__name__)

_FINANCIAL_KEYWORDS = {
    "market", "stock", "equity", "bond", "rate", "inflation", "economy", "gdp",
    "federal", "reserve", "fed", "oil", "energy", "finance", "bank", "trade",
    "currency", "dollar", "interest", "yield", "recession", "growth", "crypto",
    "debt", "deficit", "tariff", "commodity", "index", "earnings", "revenue",
    "geopolitical", "conflict", "supply", "demand", "sector", "invest", "fund",
    "nasdaq", "dow", "s&p", "portfolio", "asset", "hedge", "exchange", "risk",
}


def _is_financially_relevant(article: dict) -> bool:
    text = f"{article.get('title', '')} {article.get('description', '')}".lower()
    return any(kw in text for kw in _FINANCIAL_KEYWORDS)


def ingest_events(db: Session) -> list[dict]:
    """
    Fetch latest news, classify, map assets, persist, return combined articles.
    Uses cached news/classifications — fast on repeat calls within TTL windows.
    """
    articles = fetch_news()
    response: list[dict] = []

    for article in articles:
        if not _is_financially_relevant(article):
            logger.info("Skipping non-financial article: %s", article.get("title", ""))
            continue

        article_hash = build_article_hash(article)
        classification = classify_event(article)

        if classification.get("confidence", 0) < 0.2:
            logger.info("Skipping low-confidence article: %s", article.get("title", ""))
            continue

        mapped_symbols = map_article_to_assets(article, classification)
        assets: list[dict] = []
        prices: list[float] = []

        for symbol in mapped_symbols:
            try:
                market_data = fetch_price(symbol)
                price = float(market_data["price"])
                assets.append({"symbol": str(market_data["symbol"]), "price": price})
                prices.append(price)
                create_market_snapshot(
                    db,
                    article_hash=article_hash,
                    symbol=str(market_data["symbol"]),
                    price=price,
                )
            except Exception as exc:
                logger.warning("Failed to fetch price for %s during ingestion: %s", symbol, exc)
                continue

        avg_price = sum(prices) / len(prices) if prices else None

        upsert_event_record(
            db,
            article_hash=article_hash,
            article=article,
            classification=classification,
            mapped_assets=mapped_symbols,
            assets_json=json.dumps(assets),
            price_at_event=avg_price,
        )
        try:
            run_prediction_pipeline(
                article,
                db=db,
                classification=classification,
                mapped_assets=mapped_symbols,
            )
        except Exception as exc:
            logger.warning("Failed to persist prediction pipeline for %s: %s", article_hash, exc)

        response.append({
            "title": article.get("title", ""),
            "description": article.get("description", ""),
            "event_type": classification["event_type"],
            "affected_sectors": classification["affected_sectors"],
            "impact_direction": classification["impact_direction"],
            "confidence": classification["confidence"],
            "severity": classification["severity"],
            "reasoning": classification["reasoning"],
            "assets": assets,
        })

    return response
