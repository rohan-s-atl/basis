from app.services.mapping_service import map_article_to_assets


def test_maps_oil_conflict_article_to_energy_assets() -> None:
    article = {
        "title": "Oil rises as conflict disrupts supply routes",
        "description": "Energy traders react to war-related production risks.",
    }

    assets = map_article_to_assets(article)

    assert "USO" in assets
    assert "XLE" in assets


def test_maps_classification_to_sector_and_event_assets() -> None:
    article = {
        "title": "Central bank signals rate decision",
        "description": "Investors assess the impact across banks and growth stocks.",
    }
    classification = {
        "event_type": "interest_rate_change",
        "affected_sectors": ["financials", "technology"],
    }

    assets = map_article_to_assets(article, classification)

    assert "TLT" in assets
    assert "XLF" in assets
    assert "XLK" in assets
