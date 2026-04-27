import requests

from app.core.config import settings
from app.services.cache_service import cache


MOCK_ARTICLES = [
    {
        "title": "Oil prices rise amid conflict concerns",
        "description": "Energy markets react as geopolitical conflict raises supply fears.",
        "publishedAt": "2026-04-26T00:00:00Z",
    },
    {
        "title": "Inflation data renews interest rate debate",
        "description": "Investors weigh whether inflation will keep rates elevated.",
        "publishedAt": "2026-04-26T00:00:00Z",
    },
    {
        "title": "Technology shares lead market rebound",
        "description": "Large-cap technology companies helped lift major indexes.",
        "publishedAt": "2026-04-26T00:00:00Z",
    },
]


def fetch_news() -> list[dict[str, str]]:
    cached_news = cache.get("news:latest")
    if cached_news is not None:
        return cached_news

    if not settings.news_api_key:
        cache.set("news:latest", MOCK_ARTICLES, settings.news_cache_ttl_seconds)
        return MOCK_ARTICLES

    params = {
        "apiKey": settings.news_api_key,
        "q": "economy OR markets OR oil OR inflation OR technology",
        "language": "en",
        "pageSize": 10,
        "sortBy": "publishedAt",
    }

    response = requests.get(settings.news_api_url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    articles = data.get("articles", [])

    normalized_articles = [
        {
            "title": article.get("title") or "",
            "description": article.get("description") or "",
            "publishedAt": article.get("publishedAt") or "",
        }
        for article in articles
    ]

    cache.set("news:latest", normalized_articles, settings.news_cache_ttl_seconds)
    return normalized_articles
