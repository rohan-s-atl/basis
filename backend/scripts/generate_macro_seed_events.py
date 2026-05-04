from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.macro_seed_generator_service import (
    BLS_SERIES_SPECS,
    FRED_SERIES_SPECS,
    generate_bls_macro_event_rows,
    generate_fred_macro_event_rows,
    write_macro_events_csv,
)


def main() -> None:
    args = _parse_args()
    if args.source == "bls":
        rows = generate_bls_macro_event_rows(
            series_ids=args.series,
            start_year=int(args.start[:4]),
            end_year=int(args.end[:4]) if args.end else int(args.start[:4]) + 5,
        )
    else:
        rows = generate_fred_macro_event_rows(
            series_ids=args.series,
            api_key=args.fred_api_key,
            observation_start=args.start,
            observation_end=args.end,
        )
    write_macro_events_csv(rows, args.output)
    print(f"Generated {len(rows)} macro seed events")
    print(f"Output: {args.output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate historical macro-event seed rows from BLS or FRED time series."
    )
    parser.add_argument(
        "--source",
        choices=["fred", "bls"],
        default="bls",
        help="Data source. BLS requires no API key; FRED requires FRED_API_KEY.",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=None,
        help=(
            "Series IDs to convert into seed events. "
            f"FRED: {', '.join(sorted(FRED_SERIES_SPECS))}. "
            f"BLS: {', '.join(sorted(BLS_SERIES_SPECS))}."
        ),
    )
    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="Observation start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Observation end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output",
        default="app/data/historical_events.generated.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--fred-api-key",
        default=None,
        help="FRED API key. Defaults to FRED_API_KEY from .env.",
    )
    args = parser.parse_args()
    if args.series is None:
        args.series = sorted(BLS_SERIES_SPECS if args.source == "bls" else FRED_SERIES_SPECS)
    _validate_series(args.source, args.series)
    return args


def _validate_series(source: str, series: list[str]) -> None:
    allowed = BLS_SERIES_SPECS if source == "bls" else FRED_SERIES_SPECS
    unsupported = [series_id for series_id in series if series_id.upper() not in allowed]
    if unsupported:
        supported = ", ".join(sorted(allowed))
        raise SystemExit(f"Unsupported {source.upper()} series: {unsupported}. Supported: {supported}")


if __name__ == "__main__":
    main()
