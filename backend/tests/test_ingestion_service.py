from app.services.ingestion_service import _is_financially_relevant


def test_financial_relevance_accepts_market_news() -> None:
    assert _is_financially_relevant(
        {
            "title": "Inflation data lifts Treasury yields",
            "description": "Investors price in a slower path for rate cuts.",
        }
    )


def test_financial_relevance_rejects_no_reserve_auction_noise() -> None:
    assert not _is_financially_relevant(
        {
            "title": "1975 Yamaha DT400 Enduro Project at No Reserve",
            "description": "A motorcycle auction listing attracts bidders.",
        }
    )


def test_financial_relevance_requires_more_than_one_weak_macro_word() -> None:
    assert not _is_financially_relevant(
        {
            "title": "The legal framework for ocean carbon capture and storage",
            "description": "Researchers discuss environmental permitting.",
        }
    )
