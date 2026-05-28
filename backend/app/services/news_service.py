from datetime import UTC, datetime, timedelta

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


def fetch_news(symbols: list[str] | None = None) -> list[dict]:
    normalized_symbols = _normalize_symbols(symbols)
    cache_key = "news:latest" if not normalized_symbols else f"news:symbols:{','.join(normalized_symbols)}"
    cached_news = cache.get(cache_key)
    if cached_news is not None:
        return cached_news

    provider_articles = _provider_news(normalized_symbols)
    if provider_articles:
        cache.set(cache_key, provider_articles, settings.news_cache_ttl_seconds)
        return provider_articles

    if not settings.news_api_key:
        cache.set(cache_key, MOCK_ARTICLES, settings.news_cache_ttl_seconds)
        return MOCK_ARTICLES

    query = _newsapi_query(normalized_symbols)
    params = {
        "apiKey": settings.news_api_key,
        "q": query,
        "searchIn": "title,description",
        "language": "en",
        "pageSize": 30 if normalized_symbols else 15,
        "sortBy": "relevancy" if normalized_symbols else "publishedAt",
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
            "source": _source_name(article.get("source")) or "NewsAPI",
            "provider": "newsapi",
            "providers": "newsapi",
            "source_count": 1,
            "url": article.get("url") or "",
            "related": ",".join(normalized_symbols),
        }
        for article in articles
    ]

    cache.set(cache_key, normalized_articles, settings.news_cache_ttl_seconds)
    return normalized_articles


def _provider_news(symbols: list[str]) -> list[dict]:
    provider_batches = [
        _safe_provider_fetch(lambda: _fetch_symbol_news(symbols)),
        _safe_provider_fetch(lambda: _fetch_alpha_vantage_news(symbols)),
        _safe_provider_fetch(lambda: _fetch_marketaux_news(symbols)),
    ]
    return _dedupe_articles([article for batch in provider_batches for article in batch])[:40]


def _safe_provider_fetch(fetcher) -> list[dict]:
    try:
        return fetcher()
    except Exception:
        return []


def _fetch_symbol_news(symbols: list[str]) -> list[dict]:
    if not settings.finnhub_api_key:
        return []

    today = datetime.now(UTC).date()
    start = today - timedelta(days=14)
    articles: list[dict] = []
    seen_urls: set[str] = set()
    for symbol in symbols[:12]:
        response = requests.get(
            f"{settings.finnhub_api_url.rstrip('/')}/company-news",
            params={
                "symbol": symbol,
                "from": start.isoformat(),
                "to": today.isoformat(),
                "token": settings.finnhub_api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        for article in response.json()[:8]:
            url = str(article.get("url") or "")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            articles.append(
                {
                    "title": article.get("headline") or "",
                    "description": article.get("summary") or "",
                    "publishedAt": _finnhub_datetime(article.get("datetime")),
                    "source": article.get("source") or "Finnhub",
                    "provider": "finnhub",
                    "providers": "finnhub",
                    "source_count": 1,
                    "url": url,
                    "related": article.get("related") or symbol,
                }
            )
    articles.sort(key=lambda item: item.get("publishedAt", ""), reverse=True)
    return articles[:30]


def _fetch_alpha_vantage_news(symbols: list[str]) -> list[dict]:
    if not settings.alpha_vantage_api_key:
        return []

    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": settings.alpha_vantage_api_key,
        "sort": "LATEST",
        "limit": "50",
    }
    if symbols:
        params["tickers"] = ",".join(symbols[:12])
    else:
        params["topics"] = "financial_markets,economy_macro,economy_monetary,earnings,technology"

    response = requests.get(settings.alpha_vantage_api_url, params=params, timeout=10)
    response.raise_for_status()
    feed = response.json().get("feed", [])

    articles: list[dict] = []
    for article in feed:
        related = _alpha_related_symbols(article)
        articles.append(
            {
                "title": article.get("title") or "",
                "description": article.get("summary") or "",
                "publishedAt": _alpha_datetime(article.get("time_published")),
                "source": article.get("source") or "Alpha Vantage",
                "provider": "alpha_vantage",
                "providers": "alpha_vantage",
                "source_count": 1,
                "url": article.get("url") or "",
                "related": ",".join(related or symbols),
                "provider_sentiment": str(article.get("overall_sentiment_label") or ""),
            }
        )
    return articles


def _fetch_marketaux_news(symbols: list[str]) -> list[dict]:
    if not settings.marketaux_api_key:
        return []

    params = {
        "api_token": settings.marketaux_api_key,
        "language": "en",
        "limit": "25" if symbols else "15",
    }
    if symbols:
        params["symbols"] = ",".join(symbols[:12])
    else:
        params["filter_entities"] = "true"

    response = requests.get(settings.marketaux_api_url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json().get("data", [])

    articles: list[dict] = []
    for article in data:
        entities = article.get("entities") or []
        related = [
            str(entity.get("symbol", "")).upper()
            for entity in entities
            if str(entity.get("symbol", "")).strip()
        ]
        sentiment = next(
            (
                str(entity.get("sentiment_score"))
                for entity in entities
                if entity.get("sentiment_score") is not None
            ),
            "",
        )
        articles.append(
            {
                "title": article.get("title") or "",
                "description": article.get("description") or article.get("snippet") or "",
                "publishedAt": article.get("published_at") or "",
                "source": _source_name(article.get("source")) or "Marketaux",
                "provider": "marketaux",
                "providers": "marketaux",
                "source_count": 1,
                "url": article.get("url") or "",
                "related": ",".join(related or symbols),
                "provider_sentiment": sentiment,
            }
        )
    return articles


def _newsapi_query(symbols: list[str]) -> str:
    if not symbols:
        return "(economy OR markets OR oil OR inflation OR technology OR earnings OR rates) AND (stocks OR investors OR market OR shares)"
    company_terms = [_company_query_term(symbol) for symbol in symbols[:12]]
    return f"({' OR '.join(company_terms)}) AND (stock OR shares OR earnings OR revenue OR investors OR market)"


def _company_query_term(symbol: str) -> str:
    names = {
        "AAPL": '"Apple"',
        "AMZN": '"Amazon"',
        "TSM": '"TSMC" OR "Taiwan Semiconductor"',
    }
    return f"{symbol} OR {names.get(symbol, symbol)}"


def _normalize_symbols(symbols: list[str] | None) -> list[str]:
    return sorted({str(symbol).strip().upper() for symbol in symbols or [] if str(symbol).strip()})


def _finnhub_datetime(value: object) -> str:
    try:
        return datetime.fromtimestamp(int(value), tz=UTC).isoformat()
    except Exception:
        return ""


def _alpha_datetime(value: object) -> str:
    text = str(value or "")
    if len(text) < 8:
        return ""
    try:
        return datetime.strptime(text[:15], "%Y%m%dT%H%M%S").replace(tzinfo=UTC).isoformat()
    except Exception:
        return text


def _alpha_related_symbols(article: dict) -> list[str]:
    return [
        str(item.get("ticker", "")).upper()
        for item in article.get("ticker_sentiment", []) or []
        if str(item.get("ticker", "")).strip()
    ]


def _source_name(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("id") or "")
    return str(value or "")


def _dedupe_articles(articles: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    deduped: list[dict] = []
    for article in sorted(articles, key=lambda item: item.get("publishedAt", ""), reverse=True):
        key = (article.get("url") or article.get("title") or "").strip().lower()
        if not key:
            continue
        if key in seen:
            _merge_duplicate_article(seen[key], article)
            continue
        copy = dict(article)
        _normalize_article_provider_fields(copy)
        seen[key] = copy
        deduped.append(copy)
    return deduped


def _normalize_article_provider_fields(article: dict) -> None:
    providers = _split_values(article.get("providers") or article.get("provider"))
    if not providers and article.get("source"):
        providers = _split_values(article.get("source"))
    article["providers"] = ",".join(sorted(providers))
    article["source_count"] = max(_safe_int(article.get("source_count"), 1), len(providers) or 1)


def _merge_duplicate_article(target: dict, duplicate: dict) -> None:
    target_providers = _split_values(target.get("providers") or target.get("provider"))
    duplicate_providers = _split_values(duplicate.get("providers") or duplicate.get("provider"))
    target["providers"] = ",".join(sorted(target_providers | duplicate_providers))
    target["source_count"] = max(
        _safe_int(target.get("source_count"), 1),
        len(target_providers | duplicate_providers),
    )
    related = _split_values(target.get("related")) | _split_values(duplicate.get("related"))
    target["related"] = ",".join(sorted(related))
    if not target.get("provider_sentiment") and duplicate.get("provider_sentiment"):
        target["provider_sentiment"] = duplicate.get("provider_sentiment")


def _split_values(value: object) -> set[str]:
    if isinstance(value, list):
        parts = value
    else:
        parts = str(value or "").replace("|", ",").replace(";", ",").split(",")
    return {str(part).strip().lower() for part in parts if str(part).strip()}


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
