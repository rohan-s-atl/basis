import json
from functools import lru_cache
from pathlib import Path
from typing import Any

MAPPING_PATH = Path(__file__).resolve().parents[1] / "data" / "asset_mapping.json"


def map_article_to_assets(article: dict, classification: dict | None = None) -> list[str]:
    mapping = _load_mapping()
    title = str(article.get("title", ""))
    description = str(article.get("description", ""))
    content = f"{title} {description}".lower()

    assets: set[str] = set()

    for rule in mapping.get("keyword_rules", []):
        keywords = rule.get("keywords", [])
        if any(str(keyword).lower() in content for keyword in keywords):
            assets.update(str(asset).upper() for asset in rule.get("assets", []))

    if classification:
        event_type = str(classification.get("event_type", ""))
        sectors = classification.get("affected_sectors", [])

        assets.update(
            str(asset).upper()
            for asset in mapping.get("event_assets", {}).get(event_type, [])
        )

        if isinstance(sectors, list):
            for sector in sectors:
                assets.update(
                    str(asset).upper()
                    for asset in mapping.get("sector_assets", {}).get(str(sector), [])
                )

    return sorted(assets)


def asset_exposure_for(symbol: str, classification: dict | None = None) -> dict[str, float | str]:
    """Return directional macro exposure for an asset in the context of a classified event.

    sign=1 means the event's positive impact direction is expected to help the asset.
    sign=-1 means the event's positive impact direction is expected to hurt the asset.
    """
    normalized = symbol.upper()
    event_type = str((classification or {}).get("event_type") or "general_market")
    sectors = classification.get("affected_sectors", []) if classification else []
    sign = 1.0
    confidence = 0.55
    rationale = "default macro exposure"

    if event_type in {"inflation", "interest_rate_change"}:
        if normalized in {"QQQ", "SPY", "IWM", "TLT", "XLK"}:
            sign = -1.0
            confidence = 0.75
            rationale = "rates and inflation pressure duration-sensitive assets"
        elif normalized in {"XLF", "UUP"}:
            sign = 1.0
            confidence = 0.65
            rationale = "rates can support financials or dollar proxies"
        elif normalized in {"GLD", "TIP"}:
            sign = 1.0
            confidence = 0.6
            rationale = "inflation hedge exposure"
    elif event_type in {"geopolitical_conflict", "supply_shock"}:
        if normalized in {"XLE", "USO", "DBC", "GLD"}:
            sign = 1.0
            confidence = 0.75
            rationale = "supply stress can support energy, commodities, and hedges"
        elif normalized in {"JETS", "QQQ", "SPY"}:
            sign = -1.0
            confidence = 0.65
            rationale = "risk-off and input-cost pressure"
    elif event_type == "economic_data":
        sign = 1.0
        confidence = 0.6
        rationale = "growth-sensitive exposure"

    if isinstance(sectors, list) and "airlines" in sectors and normalized == "JETS":
        sign = -1.0
        confidence = max(confidence, 0.7)
        rationale = "airlines are vulnerable to energy and demand shocks"

    return {
        "sign": sign,
        "confidence": confidence,
        "rationale": rationale,
    }


@lru_cache(maxsize=1)
def _load_mapping() -> dict[str, Any]:
    with MAPPING_PATH.open("r", encoding="utf-8") as mapping_file:
        return json.load(mapping_file)
