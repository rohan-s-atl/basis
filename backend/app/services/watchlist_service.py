import json
from functools import lru_cache
from pathlib import Path

from app.services.mapping_service import map_article_to_assets

PROFILE_PATH = Path(__file__).resolve().parents[1] / "data" / "watchlist_symbol_profiles.json"


def analyze_watchlist(symbols: list[str], events: list[dict]) -> dict:
    normalized_symbols = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    impacted_assets = []

    for symbol in normalized_symbols:
        matched_events = _match_events(symbol, events)
        if not matched_events:
            continue

        directions = [event.get("impact_direction", "neutral") for event in matched_events]
        highest_severity = max(
            (str(event.get("severity", "low")) for event in matched_events),
            key=_severity_score,
        )

        impacted_assets.append(
            {
                "symbol": symbol,
                "matched_events": len(matched_events),
                "highest_severity": highest_severity,
                "net_direction": _net_direction(directions),
                "reason": _build_reason(symbol, matched_events),
            }
        )

    risk_score = _portfolio_risk_score(impacted_assets, max(len(normalized_symbols), 1))

    return {
        "watchlist": normalized_symbols,
        "impacted_assets": sorted(
            impacted_assets,
            key=lambda item: (_severity_score(item["highest_severity"]), item["matched_events"]),
            reverse=True,
        ),
        "portfolio_risk_score": risk_score,
    }


def _match_events(symbol: str, events: list[dict]) -> list[dict]:
    profiles = _load_profiles()
    symbol_sectors = set(profiles.get(symbol, []))
    matched = []

    for event in events:
        event_assets = set(map_article_to_assets(event, event))
        event_assets.update(str(asset).upper() for asset in event.get("mapped_assets", []) if str(asset).strip())
        event_assets.update(str(asset).upper() for asset in str(event.get("related", "")).replace(",", " ").split() if str(asset).strip())
        event_sectors = set(event.get("affected_sectors", []))

        if symbol in event_assets or symbol_sectors.intersection(event_sectors):
            matched.append(event)

    return matched


@lru_cache(maxsize=1)
def _load_profiles() -> dict[str, list[str]]:
    with PROFILE_PATH.open("r", encoding="utf-8") as profile_file:
        return json.load(profile_file)


def _severity_score(severity: str) -> int:
    return {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(severity, 1)


def _net_direction(directions: list[str]) -> str:
    score = directions.count("positive") - directions.count("negative")
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def _portfolio_risk_score(impacted_assets: list[dict], watchlist_size: int) -> int:
    if not impacted_assets:
        return 0

    raw_score = sum(
        _severity_score(item["highest_severity"]) * max(item["matched_events"], 1)
        for item in impacted_assets
    )
    return min(100, round((raw_score / (watchlist_size * 4)) * 100))


def _build_reason(symbol: str, matched_events: list[dict]) -> str:
    top_event = max(matched_events, key=lambda event: _severity_score(str(event.get("severity", "low"))))
    event_type = str(top_event.get("event_type", "general_market")).replace("_", " ")
    direction = str(top_event.get("impact_direction", "neutral"))
    providers = sorted({
        str(event.get("provider") or event.get("source") or "").replace("_", " ").title()
        for event in matched_events
        if str(event.get("provider") or event.get("source") or "").strip()
    })
    provider_text = f" Sources: {', '.join(providers[:3])}." if providers else ""
    return f"{symbol} is exposed to {event_type} events with {direction} market pressure.{provider_text}"
