import random
from pathlib import Path

from profiling.ingestion.fetch_raw import fetch_metadata, download_selected_videos
from profiling.utils.creator_config import get_active_creator, get_test_creator


OUTPUT_DIR = Path("data/test/video")


def main():
    creator_id = get_test_creator()
    print(f"[RANDOM] Fetching metadata for {creator_id}...")
    videos = fetch_metadata(creator_id, video_limit=200)
    if not videos:
        raise RuntimeError("No videos returned from metadata fetch")

    existing_ids = set()
    for p in Path("data/raw_videos").glob(f"{creator_id}_*.mp4"):
        vid = p.stem.replace(f"{creator_id}_", "")
        existing_ids.add(vid)

    candidates = [v for v in videos if v.get("id") not in existing_ids]
    if not candidates:
        raise RuntimeError("No new videos available (all already downloaded)")

    choice = random.choice(candidates)
    video_id = choice.get("id")
    if not video_id:
        raise RuntimeError("Random video missing id")

    print(f"[RANDOM] Selected video_id={video_id}")
    download_selected_videos(creator_id, [choice])

    # Copy/move into profiling/test/video for resonance test
    src = Path(choice["local_path"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = OUTPUT_DIR / src.name
    if src.exists() and src.resolve() != dst.resolve():
        dst.write_bytes(src.read_bytes())

    print(f"[RANDOM] Saved test video → {dst}")


if __name__ == "__main__":
    main()
