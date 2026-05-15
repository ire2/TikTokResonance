from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from resonance.artifact_manifest import MEDIA_EXTENSIONS


ALLOWED_PREFIXES = (
    "data/raw_data/",
    "data/raw_captions/",
    "data/raw_visual/",
    "data/embeddings_store/",
    "data/drafts/",
    "data/labels/",
    "data/artifacts/",
    "data/reviews/",
)


def _safe_member(name: str) -> bool:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    if path.suffix.lower() in MEDIA_EXTENSIONS:
        return False
    return any(name.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import a derived TikTokResonance artifact pack into local data/."
    )
    parser.add_argument("pack", help="Path to artifact zip pack.")
    parser.add_argument(
        "--target-root",
        default=".",
        help="Repository root to merge artifacts into.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be imported without writing them.",
    )
    args = parser.parse_args()

    pack = Path(args.pack)
    target_root = Path(args.target_root)
    imported = []
    skipped = []

    with zipfile.ZipFile(pack) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            if not _safe_member(member.filename):
                skipped.append(member.filename)
                continue
            imported.append(member.filename)
            if args.dry_run:
                continue
            target = target_root / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())

    print(json.dumps({
        "pack": str(pack),
        "target_root": str(target_root),
        "dry_run": args.dry_run,
        "imported": len(imported),
        "skipped": len(skipped),
        "skipped_examples": skipped[:10],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
