from app.services.watchlist_service import analyze_watchlist


def test_analyze_watchlist_matches_symbol_sector_exposure() -> None:
    events = [
        {
            "title": "Oil supply disruption",
            "description": "Energy markets react to supply risk.",
            "event_type": "supply_shock",
            "affected_sectors": ["energy", "commodities"],
            "impact_direction": "negative",
            "severity": "high",
            "mapped_assets": ["XLE", "USO"],
        }
    ]

    result = analyze_watchlist(["XOM", "MSFT"], events)

    assert result["portfolio_risk_score"] > 0
    assert result["impacted_assets"][0]["symbol"] == "XOM"
    assert result["impacted_assets"][0]["net_direction"] == "negative"
