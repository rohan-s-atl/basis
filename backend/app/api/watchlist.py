from fastapi import APIRouter, HTTPException

from app.schemas import WatchlistImpact, WatchlistRequest
from app.services.event_classifier import classify_event
from app.services.mapping_service import map_article_to_assets
from app.services.news_service import fetch_news
from app.services.watchlist_service import analyze_watchlist

router = APIRouter(tags=["watchlist"])


@router.post("/watchlist/impact", response_model=WatchlistImpact)
def get_watchlist_impact(request: WatchlistRequest) -> dict:
    try:
        events = []
        for article in fetch_news():
            classification = classify_event(article)
            events.append(
                {
                    **article,
                    **classification,
                    "mapped_assets": map_article_to_assets(article, classification),
                }
            )

        return analyze_watchlist(request.symbols, events)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to analyze watchlist") from exc
