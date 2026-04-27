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


@lru_cache(maxsize=1)
def _load_mapping() -> dict[str, Any]:
    with MAPPING_PATH.open("r", encoding="utf-8") as mapping_file:
        return json.load(mapping_file)
