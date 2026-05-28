from app.services import news_service


def test_fetch_news_blends_alpha_and_marketaux(monkeypatch) -> None:
    monkeypatch.setattr(news_service.cache, "get", lambda key: None)
    saved = {}
    monkeypatch.setattr(news_service.cache, "set", lambda key, value, ttl: saved.setdefault("value", value))
    monkeypatch.setattr(news_service.settings, "finnhub_api_key", None)
    monkeypatch.setattr(news_service.settings, "news_api_key", None)
    monkeypatch.setattr(news_service.settings, "alpha_vantage_api_key", "alpha")
    monkeypatch.setattr(news_service.settings, "marketaux_api_key", "marketaux")

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, params, timeout):
        if "alphavantage" in url:
            return FakeResponse(
                {
                    "feed": [
                        {
                            "title": "Apple earnings lift shares",
                            "summary": "Investors reacted to Apple earnings.",
                            "time_published": "20260528T120000",
                            "source": "Example Alpha",
                            "url": "https://example.com/aapl",
                            "overall_sentiment_label": "Bullish",
                            "ticker_sentiment": [{"ticker": "AAPL"}],
                        }
                    ]
                }
            )
        return FakeResponse(
            {
                "data": [
                    {
                        "title": "TSMC expands chip capacity",
                        "description": "TSMC capacity news matters to semiconductor investors.",
                        "published_at": "2026-05-28T13:00:00Z",
                        "source": "Example Marketaux",
                        "url": "https://example.com/tsm",
                        "entities": [{"symbol": "TSM", "sentiment_score": 0.24}],
                    }
                ]
            }
        )

    monkeypatch.setattr(news_service.requests, "get", fake_get)

    articles = news_service.fetch_news(symbols=["AAPL", "TSM"])

    assert [article["provider"] for article in articles] == ["marketaux", "alpha_vantage"]
    assert articles[0]["related"] == "TSM"
    assert articles[1]["related"] == "AAPL"
    assert saved["value"] == articles
