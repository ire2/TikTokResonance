from __future__ import annotations

import argparse
import json
from pathlib import Path

from resonance.artifact_manifest import MEDIA_EXTENSIONS


KNOWN_MEDIA_DIRS = [
    Path("data/raw_videos"),
    Path("data/test/video"),
]


def find_media_files() -> list[Path]:
    files: list[Path] = []
    for folder in KNOWN_MEDIA_DIRS:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
                files.append(path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delete only raw media files from known video folders."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete media files. Omit for dry run.",
    )
    args = parser.parse_args()

    files = find_media_files()
    deleted = []
    if args.apply:
        for path in files:
            path.unlink()
            deleted.append(str(path))

    print(json.dumps({
        "mode": "delete" if args.apply else "dry_run",
        "candidate_count": len(files),
        "candidates": [str(path) for path in files],
        "deleted_count": len(deleted),
        "protected_artifacts": [
            "data/raw_captions",
            "data/labels",
            "data/raw_visual",
            "data/embeddings_store",
            "data/reviews",
            "data/demo",
            "data/artifacts",
        ],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
