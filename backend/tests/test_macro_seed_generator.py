from datetime import date

from app.services.macro_seed_generator_service import (
    BLS_SERIES_SPECS,
    FRED_SERIES_SPECS,
    fetch_fred_release_calendar,
    generate_macro_event_rows,
    normalize_bls_observations,
    parse_bls_release_calendar,
    serialize_macro_events_csv,
)


def test_generate_macro_event_rows_flags_cpi_surprises() -> None:
    observations = [
        {"date": "2023-01-01", "value": "100.0"},
        {"date": "2023-02-01", "value": "100.2"},
        {"date": "2023-03-01", "value": "100.4"},
        {"date": "2023-04-01", "value": "100.6"},
        {"date": "2023-05-01", "value": "100.8"},
        {"date": "2023-06-01", "value": "101.0"},
        {"date": "2023-07-01", "value": "101.2"},
        {"date": "2023-08-01", "value": "101.4"},
        {"date": "2023-09-01", "value": "102.6"},
    ]

    rows = generate_macro_event_rows(FRED_SERIES_SPECS["CPIAUCSL"], observations)

    assert len(rows) == 1
    row = rows[0]
    assert row["event_type"] == "inflation"
    assert row["impact_direction"] == "negative"
    assert row["timestamp"].startswith("2023-10-10T08:30:00")
    assert row["affected_sectors"] == "broad_market;technology;rates"
    assert row["mapped_assets"] == "SPY;QQQ;TLT;GLD"


def test_generate_macro_event_rows_flags_cooling_cpi_as_positive() -> None:
    observations = [
        {"date": "2023-01-01", "value": "100.0"},
        {"date": "2023-02-01", "value": "100.7"},
        {"date": "2023-03-01", "value": "101.4"},
        {"date": "2023-04-01", "value": "102.1"},
        {"date": "2023-05-01", "value": "102.8"},
        {"date": "2023-06-01", "value": "103.5"},
        {"date": "2023-07-01", "value": "104.2"},
        {"date": "2023-08-01", "value": "104.9"},
        {"date": "2023-09-01", "value": "105.0"},
    ]

    rows = generate_macro_event_rows(FRED_SERIES_SPECS["CPIAUCSL"], observations)

    assert len(rows) == 1
    assert rows[0]["impact_direction"] == "positive"


def test_serialize_macro_events_csv() -> None:
    rows = [
        {
            "timestamp": "2023-09-11T08:30:00-04:00",
            "title": "CPI runs above trend",
            "description": "Generated row.",
            "event_type": "inflation",
            "impact_direction": "negative",
            "severity": "high",
            "confidence": "0.85",
            "affected_sectors": "broad_market;technology;rates",
            "mapped_assets": "SPY;QQQ;TLT;GLD",
        }
    ]

    text = serialize_macro_events_csv(rows)

    assert text.startswith("timestamp,title,description,event_type")
    assert "CPI runs above trend" in text


def test_normalize_bls_observations_sorts_monthly_rows() -> None:
    observations = [
        {"year": "2023", "period": "M03", "value": "101.0"},
        {"year": "2023", "period": "M13", "value": "annual"},
        {"year": "2023", "period": "M01", "value": "100.0"},
        {"year": "2023", "period": "M02", "value": "100.5"},
    ]

    normalized = normalize_bls_observations(observations)

    assert normalized == [
        {"date": "2023-01-01", "value": "100.0"},
        {"date": "2023-02-01", "value": "100.5"},
        {"date": "2023-03-01", "value": "101.0"},
    ]


def test_generate_macro_event_rows_accepts_bls_specs() -> None:
    observations = normalize_bls_observations([
        {"year": "2023", "period": "M01", "value": "100.0"},
        {"year": "2023", "period": "M02", "value": "100.2"},
        {"year": "2023", "period": "M03", "value": "100.4"},
        {"year": "2023", "period": "M04", "value": "100.6"},
        {"year": "2023", "period": "M05", "value": "100.8"},
        {"year": "2023", "period": "M06", "value": "101.0"},
        {"year": "2023", "period": "M07", "value": "101.2"},
        {"year": "2023", "period": "M08", "value": "101.4"},
        {"year": "2023", "period": "M09", "value": "102.6"},
    ])

    rows = generate_macro_event_rows(BLS_SERIES_SPECS["CUSR0000SA0"], observations)

    assert len(rows) == 1
    assert rows[0]["event_type"] == "inflation"
    assert rows[0]["impact_direction"] == "negative"


def test_generate_macro_event_rows_uses_release_calendar() -> None:
    observations = [
        {"date": "2023-01-01", "value": "100.0"},
        {"date": "2023-02-01", "value": "100.2"},
        {"date": "2023-03-01", "value": "100.4"},
        {"date": "2023-04-01", "value": "100.6"},
        {"date": "2023-05-01", "value": "100.8"},
        {"date": "2023-06-01", "value": "101.0"},
        {"date": "2023-07-01", "value": "101.2"},
        {"date": "2023-08-01", "value": "101.4"},
        {"date": "2023-09-01", "value": "102.6"},
    ]
    calendar = parse_bls_release_calendar(
        """
        <p>Thursday, October 12, 2023</p>
        <p>08:30 AM</p>
        <p>Consumer Price Index for September 2023</p>
        """
    )

    rows = generate_macro_event_rows(
        FRED_SERIES_SPECS["CPIAUCSL"],
        observations,
        release_calendar=calendar,
    )

    assert rows[0]["timestamp"] == "2023-10-12T08:30:00-04:00"


def test_parse_bls_release_calendar_extracts_cpi_and_jobs_dates() -> None:
    calendar = parse_bls_release_calendar(
        """
        <p>Friday, March 08, 2024</p>
        <p>08:30 AM</p>
        <p>Employment Situation for February 2024</p>
        <p>Tuesday, March 12, 2024</p>
        <p>08:30 AM</p>
        <p>Consumer Price Index for February 2024</p>
        """
    )

    assert calendar[("Employment Situation", date(2024, 2, 1))].isoformat() == (
        "2024-03-08T08:30:00-05:00"
    )
    assert calendar[("Consumer Price Index", date(2024, 2, 1))].isoformat() == (
        "2024-03-12T08:30:00-04:00"
    )


def test_fetch_fred_release_calendar_maps_release_to_prior_observation_month(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"release_dates": [{"date": "2024-03-12"}]}

    monkeypatch.setattr(
        "app.services.macro_seed_generator_service.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )

    calendar = fetch_fred_release_calendar(
        api_key="test",
        observation_start="2024-01-01",
        observation_end="2024-12-31",
    )

    assert calendar[("Consumer Price Index", date(2024, 2, 1))].isoformat() == (
        "2024-03-12T08:30:00-04:00"
    )
