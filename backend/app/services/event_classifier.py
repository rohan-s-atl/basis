import json
from typing import Any

from openai import OpenAI

from app.core.config import settings
from app.services.article_hash import build_article_hash
from app.services.cache_service import cache

FALLBACK_CLASSIFICATION = {
    "event_type": "general_market",
    "affected_sectors": ["broad_market"],
    "impact_direction": "neutral",
    "confidence": 0.0,
    "severity": "low",
    "reasoning": "insufficient data",
}

EVENT_TYPES = {
    "geopolitical_conflict",
    "inflation",
    "interest_rate_change",
    "economic_data",
    "corporate_earnings",
    "supply_shock",
    "general_market",
}

SECTORS = {
    "energy",
    "technology",
    "financials",
    "airlines",
    "commodities",
    "broad_market",
}

IMPACT_DIRECTIONS = {"positive", "negative", "neutral"}
SEVERITY_LEVELS = {"low", "medium", "high", "critical"}


def classify_event(article: dict) -> dict:
    title = str(article.get("title") or "").strip()
    description = str(article.get("description") or "").strip()

    if not title or not description:
        return FALLBACK_CLASSIFICATION.copy()

    article_hash = build_article_hash(article)
    cache_key = f"classification:{article_hash}"
    cached_classification = cache.get(cache_key)
    if cached_classification is not None:
        return cached_classification

    if not settings.openai_api_key:
        return {
            **FALLBACK_CLASSIFICATION,
            "confidence": 0.5,
            "reasoning": "OpenAI API key missing",
        }

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.responses.create(**_build_request_payload(title, description))
        content = _extract_response_text(response)
        parsed = json.loads(content)
        classification = _validate_classification(parsed)
        cache.set(
            cache_key,
            classification,
            settings.classification_cache_ttl_seconds,
        )
        return classification
    except Exception:
        return {
            **FALLBACK_CLASSIFICATION,
            "reasoning": "classification unavailable",
        }


def _build_request_payload(title: str, description: str) -> dict[str, Any]:
    prompt = f"""
Analyze this financial news article and classify the macro-financial event.

Article:
Title: {title}
Description: {description}

Return ONLY valid JSON. Do not include markdown, commentary, or text outside JSON.
Use only these values:
- event_type: geopolitical_conflict, inflation, interest_rate_change, economic_data, corporate_earnings, supply_shock, general_market
- affected_sectors: energy, technology, financials, airlines, commodities, broad_market
- impact_direction: positive, negative, neutral
- confidence: number from 0.0 to 1.0
- severity: low, medium, high, critical

The JSON must match:
{{
  "event_type": "...",
  "affected_sectors": ["..."],
  "impact_direction": "...",
  "confidence": 0.0,
  "severity": "...",
  "reasoning": "short explanation"
}}
""".strip()

    return {
        "model": settings.openai_model,
        "temperature": 0,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You classify financial news into strict JSON for a backend API.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "event_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "event_type": {
                            "type": "string",
                            "enum": sorted(EVENT_TYPES),
                        },
                        "affected_sectors": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": sorted(SECTORS),
                            },
                            "minItems": 1,
                        },
                        "impact_direction": {
                            "type": "string",
                            "enum": sorted(IMPACT_DIRECTIONS),
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "severity": {
                            "type": "string",
                            "enum": sorted(SEVERITY_LEVELS),
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": [
                        "event_type",
                        "affected_sectors",
                        "impact_direction",
                        "confidence",
                        "severity",
                        "reasoning",
                    ],
                },
            }
        },
    }


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    raise ValueError("No output text found in OpenAI response")


def _validate_classification(classification: dict[str, Any]) -> dict:
    event_type = classification.get("event_type")
    affected_sectors = classification.get("affected_sectors")
    impact_direction = classification.get("impact_direction")
    confidence = classification.get("confidence")
    severity = classification.get("severity")
    reasoning = classification.get("reasoning")

    if event_type not in EVENT_TYPES:
        raise ValueError("Invalid event_type")

    if not isinstance(affected_sectors, list) or not affected_sectors:
        raise ValueError("Invalid affected_sectors")

    if any(sector not in SECTORS for sector in affected_sectors):
        raise ValueError("Invalid affected sector")

    if impact_direction not in IMPACT_DIRECTIONS:
        raise ValueError("Invalid impact_direction")

    if not isinstance(confidence, (int, float)):
        raise ValueError("Invalid confidence")

    confidence = max(0.0, min(1.0, float(confidence)))

    if severity not in SEVERITY_LEVELS:
        raise ValueError("Invalid severity")

    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Invalid reasoning")

    return {
        "event_type": event_type,
        "affected_sectors": affected_sectors,
        "impact_direction": impact_direction,
        "confidence": confidence,
        "severity": severity,
        "reasoning": reasoning.strip(),
    }
