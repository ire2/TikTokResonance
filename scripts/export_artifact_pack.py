from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from resonance.artifact_manifest import MEDIA_EXTENSIONS, ensure_artifact_indexes


DEFAULT_OUTPUT = Path("data/artifact_packs/tiktok_resonance_artifacts.zip")
DERIVED_DIRS = [
    "data/raw_data",
    "data/raw_captions",
    "data/raw_visual",
    "data/embeddings_store",
    "data/drafts",
    "data/labels",
    "data/artifacts",
]


def _should_skip(path: Path, include_reviews: bool) -> bool:
    if path.suffix.lower() in MEDIA_EXTENSIONS:
        return True
    if "__pycache__" in path.parts:
        return True
    if path.parts[:2] == ("data", "reviews") and not include_reviews:
        return True
    return False


def _iter_pack_files(include_reviews: bool) -> list[Path]:
    roots = [Path(root) for root in DERIVED_DIRS]
    if include_reviews:
        roots.append(Path("data/reviews"))

    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not _should_skip(path, include_reviews):
                files.append(path)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export derived TikTokResonance artifacts without raw video files."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--include-reviews",
        action="store_true",
        help="Include local review JSONL artifacts in the pack.",
    )
    args = parser.parse_args()

    ensure_artifact_indexes(Path(args.data_dir))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    files = _iter_pack_files(args.include_reviews)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.as_posix())

    print(json.dumps({
        "path": str(output),
        "files": len(files),
        "include_reviews": args.include_reviews,
        "excluded_media_extensions": sorted(MEDIA_EXTENSIONS),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
