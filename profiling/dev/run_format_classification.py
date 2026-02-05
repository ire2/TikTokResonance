import argparse

from profiling.ingestion.fetch_raw import fetch_raw_videos
from profiling.ingestion.normalize import normalize_videos
from profiling.cv.simple_classifier import SimpleFormatClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    raw_videos = fetch_raw_videos(
        creator_handle=args.creator,
        video_limit=args.limit,
    )

    normalized = normalize_videos(raw_videos)

    for v in normalized:
        # v["format"] = classifier.classify(v)
        print(
            f"{v['video_id']} | "
            f"{v['duration_sec']}s | "
            f"voice={v['has_voice']} | "
            f"text={v['has_text']} | "
            f"→ {v['format']}"
        )


if __name__ == "__main__":
    main()
