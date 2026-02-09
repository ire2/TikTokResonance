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
from .normalize import normalize_videos, normalize_video


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

    existing_list = existing.get(creator_handle, [])
    existing_ids = {
        v.get("video_id") for v in existing_list if v.get("video_id")
    }
    if len(existing_ids) >= video_limit:
        print(
            f"[INGEST][{creator_handle}] existing videos: {len(existing_ids)}; "
            f"target={video_limit}. Skipping metadata scan."
        )
        return existing_list

    raw_videos = fetch_raw_videos(
        creator_handle=creator_handle,
        video_limit=video_limit,
        scan_limit=scan_limit if scan_limit else video_limit,
        selection_mode=selection_mode,
        selection_percentile=selection_percentile,
        selection_metric=selection_metric,
    )

    existing_by_id = {
        v.get("video_id"): v for v in existing_list if v.get("video_id")
    }
    raw_ids = [v.get("id") for v in raw_videos if v.get("id")]

    print(
        f"[INGEST][{creator_handle}] existing videos: {len(existing_ids)}; fetched: {len(raw_videos)}"
    )

    to_normalize = [v for v in raw_videos if v.get("id") not in existing_ids]
    normalized_by_id = {}
    if to_normalize:
        total = len(to_normalize)
        for idx, raw in enumerate(to_normalize, start=1):
            norm = normalize_video(
                raw, creator_id=creator_handle, idx=idx, total=total)
            vid = norm.get("video_id")
            if not vid:
                continue
            normalized_by_id[vid] = norm
            existing_by_id[vid] = norm
            existing[creator_handle] = list(existing_by_id.values())
            RAW_DATA_PATH.write_text(json.dumps(existing, indent=2))

    rebuilt = []
    for v in raw_videos:
        vid = v.get("id")
        if not vid:
            continue
        if vid in normalized_by_id:
            rebuilt.append(normalized_by_id[vid])
        elif vid in existing_by_id:
            rebuilt.append(existing_by_id[vid])

    if not to_normalize and set(raw_ids) == existing_ids and len(rebuilt) == len(existing_list):
        print(
            f"[INGEST][{creator_handle}] No new videos found. Skipping normalization."
        )
        return existing_list

    existing[creator_handle] = rebuilt

    RAW_DATA_PATH.write_text(json.dumps(existing, indent=2))

    return list(normalized_by_id.values())
