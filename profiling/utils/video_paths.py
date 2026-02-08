from pathlib import Path

# Canonical directory for local mp4s
RAW_VIDEO_DIR = Path("data/raw_videos")


def video_path(creator_id: str, video_id: str) -> Path:
    """
    Deterministically resolve local video path from creator + video id.
    """
    return RAW_VIDEO_DIR / f"{creator_id}_{video_id}.mp4"
