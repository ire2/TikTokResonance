import subprocess
import json
from pathlib import Path

from utils.trace import trace


# Canonical location for raw videos
RAW_VIDEO_DIR = Path("profiling/raw_videos")
RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)


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

    url = f"https://www.tiktok.com/@{creator_handle}"

    # -----------------------------
    # 1. Metadata pass (cheap)
    # -----------------------------
    meta_cmd = [
        "yt-dlp",
        "--dump-json",
        "--playlist-end", str(video_limit),
        url,
    ]

    process = subprocess.Popen(
        meta_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    videos = []

    for line in process.stdout:
        try:
            data = json.loads(line)
            video_id = data["id"]

            local_path = RAW_VIDEO_DIR / f"{creator_handle}_{video_id}.mp4"

            data["local_path"] = str(local_path)
            data["downloaded"] = local_path.exists()

            videos.append(data)

        except json.JSONDecodeError:
            continue

    # -----------------------------
    # 2. Download missing videos
    # -----------------------------
    missing = [v for v in videos if not v["downloaded"]]

    if missing:
        print(f"[INGEST] Downloading {len(missing)} videos...")

        download_cmd = [
            "yt-dlp",
            "-f", "mp4",
            "--merge-output-format", "mp4",
            "-o", str(RAW_VIDEO_DIR / f"{creator_handle}_%(id)s.%(ext)s"),
            "--playlist-end", str(video_limit),
            url,
        ]

        subprocess.run(download_cmd, check=True)

    else:
        print("[INGEST] All videos already present. Skipping download.")

    return videos
