from pathlib import Path
from profiling.cv.visual_signals import extract_visual_signals

RAW_VIDEO_DIR = Path("profiling/raw_videos")


def main():
    video = list(RAW_VIDEO_DIR.glob("*.mp4"))[0]
    signals = extract_visual_signals(str(video))

    print(f"Video: {video.name}")
    for k, v in signals.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
