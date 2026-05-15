from __future__ import annotations

import argparse
import json
from pathlib import Path

from resonance.market_cohort import DEFAULT_MARKET_INDEX_PATH, write_market_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local market memory index.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=str(DEFAULT_MARKET_INDEX_PATH))
    args = parser.parse_args()

    index = write_market_index(Path(args.data_dir), Path(args.output))
    print(json.dumps({
        "path": args.output,
        "total_creators": index["summary"]["total_creators"],
        "total_videos": index["summary"]["total_videos"],
        "coverage_tiers": index["summary"]["creators_by_coverage_tier"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
