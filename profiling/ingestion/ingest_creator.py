from utils.trace import trace
import json
from pathlib import Path
import os

from .fetch_raw import fetch_raw_videos
from profiling.utils.creator_config import (
    get_default_scan_limit,
    get_selection_mode,
    get_selection_percentile,
    get_selection_metric,
)
from .normalize import normalize_videos


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = Path("data/raw_data")
RAW_DATA_PATH = RAW_DATA_DIR / "creator_metadata.json"
LEGACY_RAW_DATA_PATH = BASE_DIR / "metadata" / \
    "raw_data" / "creator_metadata.json"


@trace
def ingest_creator(creator_handle: str, video_limit: int = 30):
    """
    End-to-end ingestion for a creator:
    - fetch raw metadata
    - download mp4s (if missing)
    - normalize metadata
    - persist to raw_data
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    scan_limit = get_default_scan_limit()
    selection_mode = get_selection_mode()
    selection_percentile = get_selection_percentile()
    selection_metric = get_selection_metric()

    # Load existing metadata
    if RAW_DATA_PATH.exists():
        existing = json.loads(RAW_DATA_PATH.read_text())
    elif LEGACY_RAW_DATA_PATH.exists():
        existing = json.loads(LEGACY_RAW_DATA_PATH.read_text())
    else:
        existing = {}

    raw_videos = fetch_raw_videos(
        creator_handle=creator_handle,
        video_limit=video_limit,
        scan_limit=scan_limit if scan_limit else video_limit,
        selection_mode=selection_mode,
        selection_percentile=selection_percentile,
        selection_metric=selection_metric,
    )

    existing_list = existing.get(creator_handle, [])
    existing_ids = {v.get("video_id")
                    for v in existing_list if v.get("video_id")}
    print(
        f"[INGEST][{creator_handle}] existing videos: {len(existing_ids)}; fetched: {len(raw_videos)}"
    )
    new_raw = [v for v in raw_videos if v.get("id") not in existing_ids]

    if not new_raw:
        print(
            f"[INGEST][{creator_handle}] No new videos found. Skipping normalization.")
        return existing_list

    normalized = normalize_videos(new_raw, creator_id=creator_handle)

    merged = list(existing_list)
    merged_ids = {v.get("video_id") for v in merged if v.get("video_id")}
    for v in normalized:
        vid = v.get("video_id")
        if vid and vid not in merged_ids:
            merged.append(v)
            merged_ids.add(vid)

    existing[creator_handle] = merged

    RAW_DATA_PATH.write_text(json.dumps(existing, indent=2))

    return normalized
