from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.init_db import init_db
from app.services.historical_seed_service import seed_historical_training_data_from_csv


def main() -> None:
    args = _parse_args()
    init_db()
    result = seed_historical_training_data_from_csv(
        Path(args.file),
        noise_threshold=args.noise_threshold,
        refresh_existing=args.refresh_existing,
    )

    print("Historical seed complete")
    print(f"  events_inserted:      {result.events_inserted}")
    print(f"  events_reused:        {result.events_reused}")
    print(f"  predictions_inserted: {result.predictions_inserted}")
    print(f"  outcomes_inserted:    {result.outcomes_inserted}")
    print(f"  skipped_rows:         {result.skipped_rows}")
    if result.errors:
        print("  errors:")
        for error in result.errors:
            print(f"    - {error}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed historical macro events into the ML training tables."
    )
    parser.add_argument(
        "--file",
        default="app/data/historical_events.sample.csv",
        help="Path to a historical events CSV file.",
    )
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=None,
        help="Minimum absolute return required for a non-null label.",
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Rebuild feature snapshots and multi-horizon labels for existing seeded predictions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
