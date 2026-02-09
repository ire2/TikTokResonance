import random
from pathlib import Path

from profiling.ingestion.fetch_raw import fetch_metadata, download_selected_videos
from profiling.utils.creator_config import get_active_creator, get_test_creator


OUTPUT_DIR = Path("data/test/video")


def main():
    creator_id = get_test_creator()
    print(f"[RANDOM] Fetching metadata for {creator_id}...")
    videos = fetch_metadata(creator_id, video_limit=50)
    if not videos:
        raise RuntimeError("No videos returned from metadata fetch")

    existing_ids = set()
    for p in Path("data/raw_videos").glob(f"{creator_id}_*.mp4"):
        vid = p.stem.replace(f"{creator_id}_", "")
        existing_ids.add(vid)

    candidates = [v for v in videos if v.get("id") not in existing_ids]
    if not candidates:
        raise RuntimeError("No new videos available (all already downloaded)")

    random.shuffle(candidates)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for choice in candidates:
        video_id = choice.get("id")
        if not video_id:
            continue

        print(f"[RANDOM] Selected video_id={video_id}")
        downloaded = download_selected_videos(creator_id, [choice])
        if not downloaded:
            print(f"[RANDOM] Skipping {video_id} (no mp4 downloaded)")
            continue

        src = Path(choice["local_path"])
        if not src.exists():
            print(f"[RANDOM] Missing file after download: {src}")
            continue

        dst = OUTPUT_DIR / src.name
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())

        print(f"[RANDOM] Saved test video → {dst}")
        return

    raise RuntimeError("No mp4-compatible videos could be downloaded")


if __name__ == "__main__":
    main()
