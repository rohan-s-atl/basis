from __future__ import annotations

import csv
import html
import io
import re
import time as time_module
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from app.core.config import settings

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
BLS_SERIES_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data"
BLS_HEADERS = {"User-Agent": "macro-event-intelligence/1.0 research contact: local"}
FRED_RELEASE_DATES_URL = "https://api.stlouisfed.org/fred/release/dates"
FRED_RELEASE_IDS = {
    "Consumer Price Index": 10,
    "Employment Situation": 50,
}
CSV_FIELDS = [
    "timestamp",
    "title",
    "description",
    "event_type",
    "impact_direction",
    "severity",
    "confidence",
    "affected_sectors",
    "mapped_assets",
]
NY_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MacroSeriesSpec:
    series_id: str
    name: str
    event_type: str
    affected_sectors: str
    mapped_assets: str
    release_day: int
    release_hour: int
    release_minute: int
    threshold: float
    rolling_window: int
    transform: str
    high_is_positive: bool
    release_name: str | None = None


FRED_SERIES_SPECS: dict[str, MacroSeriesSpec] = {
    "CPIAUCSL": MacroSeriesSpec(
        series_id="CPIAUCSL",
        name="CPI",
        event_type="inflation",
        affected_sectors="broad_market;technology;rates",
        mapped_assets="SPY;QQQ;TLT;GLD",
        release_day=10,
        release_hour=8,
        release_minute=30,
        threshold=0.0015,
        rolling_window=6,
        transform="pct_change",
        high_is_positive=False,
        release_name="Consumer Price Index",
    ),
    "UNRATE": MacroSeriesSpec(
        series_id="UNRATE",
        name="Unemployment rate",
        event_type="economic_data",
        affected_sectors="broad_market;technology;rates",
        mapped_assets="SPY;QQQ;IWM;TLT",
        release_day=7,
        release_hour=8,
        release_minute=30,
        threshold=0.1,
        rolling_window=6,
        transform="diff",
        high_is_positive=False,
        release_name="Employment Situation",
    ),
    "FEDFUNDS": MacroSeriesSpec(
        series_id="FEDFUNDS",
        name="Federal funds rate",
        event_type="interest_rate_change",
        affected_sectors="broad_market;technology;rates;financials",
        mapped_assets="SPY;QQQ;TLT;XLF",
        release_day=1,
        release_hour=14,
        release_minute=0,
        threshold=0.1,
        rolling_window=3,
        transform="diff",
        high_is_positive=False,
        release_name=None,
    ),
}

BLS_SERIES_SPECS: dict[str, MacroSeriesSpec] = {
    "CUSR0000SA0": MacroSeriesSpec(
        series_id="CUSR0000SA0",
        name="CPI",
        event_type="inflation",
        affected_sectors="broad_market;technology;rates",
        mapped_assets="SPY;QQQ;TLT;GLD",
        release_day=10,
        release_hour=8,
        release_minute=30,
        threshold=0.0015,
        rolling_window=6,
        transform="pct_change",
        high_is_positive=False,
        release_name="Consumer Price Index",
    ),
    "LNS14000000": MacroSeriesSpec(
        series_id="LNS14000000",
        name="Unemployment rate",
        event_type="economic_data",
        affected_sectors="broad_market;technology;rates",
        mapped_assets="SPY;QQQ;IWM;TLT",
        release_day=7,
        release_hour=8,
        release_minute=30,
        threshold=0.1,
        rolling_window=6,
        transform="diff",
        high_is_positive=False,
        release_name="Employment Situation",
    ),
}


def fetch_fred_observations(
    series_id: str,
    *,
    api_key: str | None = None,
    observation_start: str | None = None,
    observation_end: str | None = None,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    effective_api_key = api_key or settings.fred_api_key
    if not effective_api_key:
        raise ValueError("FRED_API_KEY is required to fetch FRED observations.")

    params: dict[str, str] = {
        "series_id": series_id,
        "api_key": effective_api_key,
        "file_type": "json",
        "sort_order": "asc",
    }
    if observation_start:
        params["observation_start"] = observation_start
    if observation_end:
        params["observation_end"] = observation_end

    return list(_get_json_with_retries(FRED_OBSERVATIONS_URL, params=params, timeout=timeout).get("observations", []))


def generate_fred_macro_event_rows(
    *,
    series_ids: list[str],
    api_key: str | None = None,
    observation_start: str | None = None,
    observation_end: str | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    release_calendar = _fetch_release_calendar_for_observation_range(
        observation_start,
        observation_end,
        api_key=api_key,
    )
    for series_id in series_ids:
        spec = _spec_for_series(series_id)
        observations = fetch_fred_observations(
            spec.series_id,
            api_key=api_key,
            observation_start=observation_start,
            observation_end=observation_end,
        )
        rows.extend(generate_macro_event_rows(spec, observations, release_calendar=release_calendar))
    return sorted(rows, key=lambda row: row["timestamp"])


def fetch_bls_observations(
    series_id: str,
    *,
    start_year: int,
    end_year: int,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    response = requests.post(
        BLS_SERIES_URL,
        json={
            "seriesid": [series_id],
            "startyear": str(start_year),
            "endyear": str(end_year),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    series = payload.get("Results", {}).get("series", [])
    if not series:
        return []
    return list(series[0].get("data", []))


def generate_bls_macro_event_rows(
    *,
    series_ids: list[str],
    start_year: int,
    end_year: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    release_calendar = _fetch_release_calendar_for_years(start_year, end_year)
    for series_id in series_ids:
        spec = _bls_spec_for_series(series_id)
        observations = normalize_bls_observations(
            fetch_bls_observations(
                spec.series_id,
                start_year=start_year,
                end_year=end_year,
            )
        )
        rows.extend(generate_macro_event_rows(spec, observations, release_calendar=release_calendar))
    return sorted(rows, key=lambda row: row["timestamp"])


def normalize_bls_observations(observations: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for observation in observations:
        period = str(observation.get("period") or "")
        if not period.startswith("M") or period == "M13":
            continue
        month = int(period[1:])
        year = int(str(observation["year"]))
        normalized.append(
            {
                "date": date(year, month, 1).isoformat(),
                "value": str(observation["value"]),
            }
        )
    return sorted(normalized, key=lambda item: item["date"])


def generate_macro_event_rows(
    spec: MacroSeriesSpec,
    observations: list[dict[str, Any]],
    *,
    release_calendar: dict[tuple[str, date], datetime] | None = None,
) -> list[dict[str, str]]:
    clean = _clean_observations(observations)
    changes = _series_changes(clean, transform=spec.transform)
    rows: list[dict[str, str]] = []

    for index in range(spec.rolling_window + 1, len(changes)):
        current_date, current_change = changes[index]
        baseline_values = [value for _, value in changes[index - spec.rolling_window:index]]
        baseline = sum(baseline_values) / len(baseline_values)
        surprise = current_change - baseline
        if abs(surprise) < spec.threshold:
            continue

        high_reading = surprise > 0
        impact_direction = "positive" if high_reading == spec.high_is_positive else "negative"
        severity = _severity(abs(surprise), spec.threshold)
        confidence = _confidence(abs(surprise), spec.threshold)
        release_timestamp = _release_timestamp_for_observation(
            spec,
            current_date,
            release_calendar=release_calendar,
        )
        rows.append(
            {
                "timestamp": release_timestamp.isoformat(),
                "title": _title(spec, high_reading),
                "description": _description(
                    spec,
                    current_change=current_change,
                    baseline=baseline,
                    surprise=surprise,
                ),
                "event_type": spec.event_type,
                "impact_direction": impact_direction,
                "severity": severity,
                "confidence": f"{confidence:.2f}",
                "affected_sectors": spec.affected_sectors,
                "mapped_assets": spec.mapped_assets,
            }
        )
    return rows


def fetch_bls_release_calendar(
    *,
    start_year: int,
    end_year: int,
    timeout: float = 30.0,
) -> dict[tuple[str, date], datetime]:
    calendar: dict[tuple[str, date], datetime] = {}
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            url = f"https://www.bls.gov/schedule/{year}/{month:02d}_sched_list.htm"
            response = requests.get(url, headers=BLS_HEADERS, timeout=timeout)
            if response.status_code in {403, 404}:
                continue
            response.raise_for_status()
            calendar.update(parse_bls_release_calendar(response.text))
    return calendar


def fetch_fred_release_calendar(
    *,
    api_key: str | None = None,
    observation_start: str | None = None,
    observation_end: str | None = None,
    timeout: float = 30.0,
) -> dict[tuple[str, date], datetime]:
    effective_api_key = api_key or settings.fred_api_key
    if not effective_api_key:
        raise ValueError("FRED_API_KEY is required to fetch FRED release dates.")

    calendar: dict[tuple[str, date], datetime] = {}
    for release_name, release_id in FRED_RELEASE_IDS.items():
        params: dict[str, str] = {
            "release_id": str(release_id),
            "api_key": effective_api_key,
            "file_type": "json",
            "sort_order": "asc",
            "limit": "1000",
        }
        if observation_start:
            params["realtime_start"] = observation_start
        if observation_end:
            end_year = int(observation_end[:4]) + 1
            params["realtime_end"] = f"{end_year}-12-31"

        payload = _get_json_with_retries(FRED_RELEASE_DATES_URL, params=params, timeout=timeout)
        for item in payload.get("release_dates", []):
            release_date = date.fromisoformat(str(item["date"]))
            reference_date = _previous_month_start(release_date)
            calendar[(release_name, reference_date)] = datetime.combine(
                release_date,
                time(8, 30),
                tzinfo=NY_TZ,
            )
    return calendar


def parse_bls_release_calendar(raw_html: str) -> dict[tuple[str, date], datetime]:
    text = html.unescape(re.sub(r"<[^>]+>", "\n", raw_html))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    calendar: dict[tuple[str, date], datetime] = {}

    index = 0
    while index + 2 < len(lines):
        line = lines[index]
        if not _looks_like_calendar_date(line):
            index += 1
            continue
        release_date = _parse_calendar_date(line)
        release_time = lines[index + 1]
        release_text = lines[index + 2]
        release_name, reference_date = _parse_bls_release_text(release_text)
        if release_name and reference_date:
            calendar[(release_name, reference_date)] = datetime.combine(
                release_date,
                _parse_release_time(release_time),
                tzinfo=NY_TZ,
            )
        index += 3
    return calendar


def write_macro_events_csv(rows: list[dict[str, str]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_macro_events_csv(rows), encoding="utf-8")


def serialize_macro_events_csv(rows: list[dict[str, str]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _spec_for_series(series_id: str) -> MacroSeriesSpec:
    normalized = series_id.upper()
    if normalized not in FRED_SERIES_SPECS:
        supported = ", ".join(sorted(FRED_SERIES_SPECS))
        raise ValueError(f"Unsupported FRED series {series_id}. Supported: {supported}")
    return FRED_SERIES_SPECS[normalized]


def _bls_spec_for_series(series_id: str) -> MacroSeriesSpec:
    normalized = series_id.upper()
    if normalized not in BLS_SERIES_SPECS:
        supported = ", ".join(sorted(BLS_SERIES_SPECS))
        raise ValueError(f"Unsupported BLS series {series_id}. Supported: {supported}")
    return BLS_SERIES_SPECS[normalized]


def _fetch_release_calendar_for_observation_range(
    observation_start: str | None,
    observation_end: str | None,
    *,
    api_key: str | None = None,
) -> dict[tuple[str, date], datetime]:
    try:
        return fetch_fred_release_calendar(
            api_key=api_key,
            observation_start=observation_start,
            observation_end=observation_end,
        )
    except Exception:
        start_year = int((observation_start or "2020-01-01")[:4])
        end_year = int((observation_end or f"{start_year + 5}-12-31")[:4])
        return fetch_bls_release_calendar(start_year=start_year, end_year=end_year + 1)


def _fetch_release_calendar_for_years(
    start_year: int,
    end_year: int,
) -> dict[tuple[str, date], datetime]:
    return _fetch_release_calendar_for_observation_range(
        f"{start_year}-01-01",
        f"{end_year}-12-31",
    )


def _clean_observations(observations: list[dict[str, Any]]) -> list[tuple[date, float]]:
    clean: list[tuple[date, float]] = []
    for observation in observations:
        value = str(observation.get("value") or "").strip()
        if not value or value == ".":
            continue
        clean.append((date.fromisoformat(str(observation["date"])), float(value)))
    return sorted(clean, key=lambda item: item[0])


def _series_changes(
    observations: list[tuple[date, float]],
    *,
    transform: str,
) -> list[tuple[date, float]]:
    changes: list[tuple[date, float]] = []
    for previous, current in zip(observations, observations[1:]):
        current_date, current_value = current
        _, previous_value = previous
        if transform == "pct_change":
            if previous_value == 0:
                continue
            changes.append((current_date, (current_value - previous_value) / previous_value))
        elif transform == "diff":
            changes.append((current_date, current_value - previous_value))
        else:
            raise ValueError(f"Unsupported transform: {transform}")
    return changes


def _estimated_release_timestamp(
    observation_date: date,
    *,
    release_day: int,
    release_hour: int,
    release_minute: int,
) -> datetime:
    release_month = _add_months(observation_date, 1).replace(day=1)
    last_day = _last_day_of_month(release_month)
    day = min(release_day, last_day)
    release_date = release_month.replace(day=day)
    while release_date.weekday() >= 5:
        release_date += timedelta(days=1)
    return datetime.combine(
        release_date,
        time(release_hour, release_minute),
        tzinfo=NY_TZ,
    )


def _release_timestamp_for_observation(
    spec: MacroSeriesSpec,
    observation_date: date,
    *,
    release_calendar: dict[tuple[str, date], datetime] | None,
) -> datetime:
    if spec.release_name and release_calendar:
        release = release_calendar.get((spec.release_name, observation_date))
        if release is not None:
            return release
    return _estimated_release_timestamp(
        observation_date,
        release_day=spec.release_day,
        release_hour=spec.release_hour,
        release_minute=spec.release_minute,
    )


def _looks_like_calendar_date(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]+,\s+[A-Za-z]+\s+\d{1,2},\s+\d{4}$", value))


def _parse_calendar_date(value: str) -> date:
    return datetime.strptime(value.replace("Sept.", "Sep."), "%A, %B %d, %Y").date()


def _parse_release_time(value: str) -> time:
    return datetime.strptime(value.strip(), "%I:%M %p").time()


def _parse_bls_release_text(value: str) -> tuple[str | None, date | None]:
    for release_name in ("Consumer Price Index", "Employment Situation"):
        pattern = rf"^{re.escape(release_name)}(?:\s+\([^)]+\))?\s+for\s+(.+)$"
        match = re.match(pattern, value)
        if not match:
            continue
        reference = _parse_reference_period(match.group(1))
        if reference is not None:
            return release_name, reference
    return None, None


def _parse_reference_period(value: str) -> date | None:
    match = re.match(r"^([A-Za-z]+)\s+(\d{4})$", value.strip())
    if not match:
        return None
    month = datetime.strptime(match.group(1), "%B").month
    year = int(match.group(2))
    return date(year, month, 1)


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    return value.replace(year=year, month=month)


def _last_day_of_month(value: date) -> int:
    next_month = _add_months(value.replace(day=1), 1)
    return (next_month - timedelta(days=1)).day


def _previous_month_start(value: date) -> date:
    first_of_month = value.replace(day=1)
    previous_month = _add_months(first_of_month, -1)
    return previous_month.replace(day=1)


def _title(spec: MacroSeriesSpec, high_reading: bool) -> str:
    direction = "runs above trend" if high_reading else "runs below trend"
    return f"{spec.name} {direction}"


def _description(
    spec: MacroSeriesSpec,
    *,
    current_change: float,
    baseline: float,
    surprise: float,
) -> str:
    return (
        f"{spec.name} moved by {current_change:.4f} versus a rolling baseline of "
        f"{baseline:.4f}, creating a macro surprise of {surprise:.4f}."
    )


def _severity(surprise_magnitude: float, threshold: float) -> str:
    multiple = surprise_magnitude / threshold
    if multiple >= 3:
        return "critical"
    if multiple >= 2:
        return "high"
    if multiple >= 1.25:
        return "medium"
    return "low"


def _confidence(surprise_magnitude: float, threshold: float) -> float:
    return min(0.95, max(0.55, 0.55 + 0.12 * (surprise_magnitude / threshold)))


def _get_json_with_retries(
    url: str,
    *,
    params: dict[str, str],
    timeout: float,
    attempts: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and status < 500:
                break
        except requests.RequestException as exc:
            last_error = exc
        if attempt < attempts - 1:
            time_module.sleep(0.75 * (attempt + 1))
    raise RuntimeError("FRED request failed after retries") from last_error
