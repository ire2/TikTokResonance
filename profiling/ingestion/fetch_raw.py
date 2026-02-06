import subprocess
import json
from pathlib import Path
from typing import List, Dict

from utils.trace import trace


# Canonical location for raw videos
RAW_VIDEO_DIR = Path("profiling/raw_videos")
RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)


@trace
def fetch_metadata(
    creator_handle: str,
    video_limit: int,
):
    """
    Fetch metadata only (no downloads).
    """

    url = f"https://www.tiktok.com/@{creator_handle}"

    cmd = [
        "yt-dlp",
        "--dump-json",
        "--skip-download",
        "--playlist-end", str(video_limit),
        url,
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    videos = []

    for line in process.stdout:
        try:
            videos.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return videos


@trace
def fetch_captions(
    creator_handle: str,
    video_limit: int,
):
    """
    Download auto-generated captions (.en.vtt) if available.
    """

    url = f"https://www.tiktok.com/@{creator_handle}"

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-subs",
        "--sub-lang", "en",
        "--sub-format", "vtt",
        "-o", str(RAW_VIDEO_DIR / f"{creator_handle}_%(id)s.%(ext)s"),
        "--playlist-end", str(video_limit),
        url,
    ]

    subprocess.run(cmd, check=True)


@trace
def download_missing_videos(
    creator_handle: str,
    videos: List[Dict],
    video_limit: int,
):
    """
    Download mp4 files only if missing locally.
    """

    missing = []

    for v in videos:
        video_id = v["id"]
        path = RAW_VIDEO_DIR / f"{creator_handle}_{video_id}.mp4"

        v["local_path"] = str(path)
        v["downloaded"] = path.exists()

        if not v["downloaded"]:
            missing.append(v)

    if not missing:
        print("[INGEST] All videos already present. Skipping download.")
        return

    print(f"[INGEST] Downloading {len(missing)} videos...")

    url = f"https://www.tiktok.com/@{creator_handle}"

    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "--merge-output-format", "mp4",
        "-o", str(RAW_VIDEO_DIR / f"{creator_handle}_%(id)s.%(ext)s"),
        "--playlist-end", str(video_limit),
        url,
    ]

    subprocess.run(cmd, check=True)


@trace
def fetch_raw_videos(
    creator_handle: str,
    video_limit: int = 30,
):
    """
    Fetch raw video metadata AND ensure mp4 files exist locally.

    Returns:
        List[dict]: yt-dlp metadata with added `local_path`
    """
    videos = fetch_metadata(creator_handle, video_limit)

    fetch_captions(creator_handle, video_limit)

    download_missing_videos(
        creator_handle=creator_handle,
        videos=videos,
        video_limit=video_limit,
    )

    return videos
