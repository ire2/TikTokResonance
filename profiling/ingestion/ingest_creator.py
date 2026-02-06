from utils.trace import trace
import json
from pathlib import Path

from .fetch_raw import fetch_raw_videos
from .normalize import normalize_videos


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = BASE_DIR / "metadata" / "raw_data"
RAW_DATA_PATH = RAW_DATA_DIR / "creator_metadata.json"


@trace
def ingest_creator(creator_handle: str, video_limit: int = 30):
    """
    End-to-end ingestion for a creator:
    - fetch raw metadata
    - download mp4s (if missing)
    - normalize metadata
    - persist to raw_data
    """

    RAW_DATA_DIR.mkdir(exist_ok=True)

    raw_videos = fetch_raw_videos(
        creator_handle=creator_handle,
        video_limit=video_limit,
    )

    normalized = normalize_videos(raw_videos)

    # Load existing metadata
    if RAW_DATA_PATH.exists():
        existing = json.loads(RAW_DATA_PATH.read_text())
    else:
        existing = {}

    existing[creator_handle] = normalized

    RAW_DATA_PATH.write_text(json.dumps(existing, indent=2))

    return normalized


if __name__ == "__main__":
    ingest_creator("expoparker", video_limit=5)
    print("Ingestion complete.")
