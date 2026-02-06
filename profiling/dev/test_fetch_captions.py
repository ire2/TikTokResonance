from profiling.ingestion.fetch_raw import fetch_captions
from pathlib import Path


def test_fetch_captions_smoke():
    creator = "expoparker"
    limit = 2

    fetch_captions(creator, limit)

    raw_dir = Path("profiling/raw_videos")
    assert raw_dir.exists()

    print("fetch_captions completed without error")
