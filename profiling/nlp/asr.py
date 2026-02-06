from pathlib import Path
from typing import Optional
import subprocess
import json

from utils.trace import trace


# Canonical location for all caption artifacts
CAPTION_DIR = Path("profiling/metadata/raw_captions")
CAPTION_DIR.mkdir(parents=True, exist_ok=True)


@trace
def caption_path(creator_id: str, video_id: str) -> Path:
    """
    Canonical caption path for a video.
    Stores Whisper JSON (with timestamps).
    """
    return CAPTION_DIR / f"{creator_id}_{video_id}.json"


@trace
def captions_exist(creator_id: str, video_id: str) -> bool:
    """
    Check if captions already exist on disk.
    """
    return caption_path(creator_id, video_id).exists()


@trace
def run_asr(video_path: str, out_path: Path):
    """
    Run ASR on a video using Whisper (JSON output).

    This function is intentionally isolated so Whisper
    can be swapped for another engine later.
    """

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "whisper",
        video_path,
        "--language", "en",
        "--model", "base",
        "--fp16", "False",
        "--output_format", "json",
        "--output_dir", str(out_path.parent),
    ]

    subprocess.run(cmd, check=True)

    # Whisper outputs <video_stem>.json — rename to canonical name
    generated = out_path.parent / (Path(video_path).stem + ".json")

    if not generated.exists():
        raise RuntimeError("ASR failed: no output JSON generated")

    generated.rename(out_path)


@trace
def ensure_captions(
    creator_id: str,
    video_id: str,
    video_path: str,
) -> Optional[Path]:
    """
    Ensure captions exist for a video.

    Resolution order:
    1. Existing captions on disk
    2. ASR fallback (Whisper)

    Returns:
        Path to caption JSON, or None if unavailable.
    """

    out_path = caption_path(creator_id, video_id)

    # 1. Already exists
    if out_path.exists():
        return out_path

    # 2. ASR fallback
    try:
        run_asr(video_path, out_path)
        return out_path

    except Exception as e:
        print(f"[ASR WARN] Failed for {video_id}: {e}")
        return None
