from app.services.event_classifier import classify_event


def test_classify_event_returns_fallback_when_title_missing() -> None:
    article = {
        "title": "",
        "description": "Inflation rose faster than expected.",
    }

    result = classify_event(article)

    assert result == {
        "event_type": "general_market",
        "affected_sectors": ["broad_market"],
        "impact_direction": "neutral",
        "confidence": 0.0,
        "severity": "low",
        "reasoning": "insufficient data",
    }


def test_classify_event_returns_fallback_when_description_missing() -> None:
    article = {
        "title": "Oil markets rally",
        "description": None,
    }

    result = classify_event(article)

    assert result["event_type"] == "general_market"
    assert result["affected_sectors"] == ["broad_market"]
    assert result["impact_direction"] == "neutral"
    assert result["confidence"] == 0.0
    assert result["severity"] == "low"
    assert result["reasoning"] == "insufficient data"
