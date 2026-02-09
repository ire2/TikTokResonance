from pathlib import Path
import json
import csv

from profiling.utils.creator_config import get_active_creator, get_training_creators


RAW_VISUAL_DIR = Path("data/raw_visual")
RAW_DATA_PATH = Path("data/raw_data/creator_metadata.json")
OUT_PATH = Path("data/labels/format_labels.csv")

DEFAULT_FORMAT_LABELS = [
    "talking_head",
    "voiceover",
    "text_heavy",
    "broll",
    "dance",
    "duo_or_group",
    "tutorial_or_demo",
    "educational",
    "skit_or_comedy",
    "product_or_ad",
    "food_or_cooking",
    "other",
]
DEFAULT_PERFORMANCE_LABELS = ["hit", "ok", "miss"]


def load_raw_data():
    if not RAW_DATA_PATH.exists():
        return {}
    try:
        return json.loads(RAW_DATA_PATH.read_text())
    except Exception:
        return {}


def load_visual_cache(creator_id: str):
    path = RAW_VISUAL_DIR / f"{creator_id}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_existing_labels():
    if not OUT_PATH.exists():
        return {}
    with open(OUT_PATH, "r") as f:
        reader = csv.DictReader(f)
        existing = {}
        for r in reader:
            key = (r.get("creator_id"), r.get("video_id"))
            existing[key] = r
        return existing


def _prompt_label(label_type: str, options: list) -> str:
    print(f"Select {label_type} label:")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}. {opt}")
    print("  s. skip")
    print("  q. quit")

    while True:
        val = input("> ").strip().lower()
        if val in ("s", "skip"):
            return ""
        if val in ("q", "quit"):
            raise KeyboardInterrupt
        if val.isdigit():
            idx = int(val) - 1
            if 0 <= idx < len(options):
                return options[idx]
        if val in options:
            return val
        print("Invalid selection. Try again.")


def _write_rows(rows):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "creator_id",
                "video_id",
                "format_label",
                "performance_label",
                "tiktok_url",
                "views",
                "likes",
                "comments",
                "duration_sec",
                "posted_at",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    raw_data = load_raw_data()
    creators = get_training_creators()
    if not creators:
        creators = [get_active_creator()]

    existing = _load_existing_labels()
    rows = []

    for creator_id in creators:
        visual_cache = load_visual_cache(creator_id)
        if not visual_cache:
            print(
                f"[WARN] No raw_visual cache for {creator_id}. Skipping."
            )
            continue

        videos = raw_data.get(creator_id, [])
        video_meta = {v.get("video_id"): v for v in videos}

        for video_id, signals in visual_cache.items():
            meta = video_meta.get(video_id, {})
            prev = existing.get((creator_id, video_id), {})
            rows.append({
                "creator_id": creator_id,
                "video_id": video_id,
                "format_label": prev.get("format_label", ""),
                "performance_label": prev.get("performance_label", ""),
                "tiktok_url": f"https://www.tiktok.com/@{creator_id}/video/{video_id}",
                "views": meta.get("views"),
                "likes": meta.get("likes"),
                "comments": meta.get("comments"),
                "duration_sec": meta.get("duration_sec"),
                "posted_at": meta.get("posted_at"),
            })

    merged = list(existing.values())
    existing_keys = {(r.get("creator_id"), r.get("video_id")) for r in merged}
    for r in rows:
        key = (r.get("creator_id"), r.get("video_id"))
        if key not in existing_keys:
            merged.append(r)
            existing_keys.add(key)

    _write_rows(merged)
    print(f"[OK] Wrote {len(merged)} rows to {OUT_PATH}")
    print(
        "[LABELS] Open the labeling UI with:\n"
        "  uvicorn profiling.label_ui.app:app --reload"
    )


if __name__ == "__main__":
    main()
