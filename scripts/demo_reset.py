from __future__ import annotations

import argparse
import json
from pathlib import Path

from resonance.review_decisions import (
    DEFAULT_DECISIONS_PATH,
    reset_review_decisions,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reset only the local demo review-decision artifact."
    )
    parser.add_argument(
        "--path",
        default=str(DEFAULT_DECISIONS_PATH),
        help="Review decision JSONL file to reset.",
    )
    args = parser.parse_args()

    result = reset_review_decisions(Path(args.path))
    print(json.dumps({
        **result,
        "note": (
            "Demo reset only truncates the review decision JSONL file. "
            "It does not touch training data, labels, raw visuals, captions, embeddings, or demo cache."
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
