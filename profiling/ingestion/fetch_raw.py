import subprocess
import json


def fetch_raw_videos(creator_handle: str, video_limit: int = 30):
    """
    Fetch raw video metadata for a TikTok creator using yt-dlp.

    Returns:
        List[dict]: one dict per video (raw yt-dlp output)
    """

    url = f"https://www.tiktok.com/@{creator_handle}"

    cmd = [
        "yt-dlp",
        "--dump-json",
        "--playlist-end",
        str(video_limit),
        url,
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    raw_videos = []

    for line in process.stdout:
        try:
            raw_videos.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return raw_videos
