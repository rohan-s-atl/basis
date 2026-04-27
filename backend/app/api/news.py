from fastapi import APIRouter, HTTPException

from app.schemas import Article
from app.services.news_service import fetch_news

router = APIRouter(tags=["news"])


@router.get("/news", response_model=list[Article])
def get_news() -> list[dict[str, str]]:
    try:
        return fetch_news()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch news") from exc
