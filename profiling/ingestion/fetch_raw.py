import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional

from utils.trace import trace


# Canonical location for raw videos
RAW_VIDEO_DIR = Path("profiling/metadata/raw_videos")
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
def fetch_captions_for_videos(
    creator_handle: str,
    videos: List[Dict],
):
    """
    Download captions for specific video IDs.
    """
    if not videos:
        return

    urls = [
        f"https://www.tiktok.com/@{creator_handle}/video/{v['id']}"
        for v in videos
    ]

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-subs",
        "--sub-lang", "en",
        "--sub-format", "vtt",
        "-o", str(RAW_VIDEO_DIR / f"{creator_handle}_%(id)s.%(ext)s"),
    ] + urls

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
def download_selected_videos(
    creator_handle: str,
    videos: List[Dict],
):
    """
    Download mp4 files for specific video IDs only.
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
        print("[INGEST] All selected videos already present. Skipping download.")
        return

    print(f"[INGEST] Downloading {len(missing)} selected videos...")

    urls = [
        f"https://www.tiktok.com/@{creator_handle}/video/{v['id']}"
        for v in missing
    ]

    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "--merge-output-format", "mp4",
        "-o", str(RAW_VIDEO_DIR / f"{creator_handle}_%(id)s.%(ext)s"),
    ] + urls

    subprocess.run(cmd, check=True)


@trace
def _metric_value(v: Dict, metric: str) -> int:
    if metric == "likes":
        return int(v.get("like_count") or 0)
    if metric == "comments":
        return int(v.get("comment_count") or 0)
    return int(v.get("view_count") or 0)


@trace
def select_videos(
    videos: List[Dict],
    metric: str = "views",
    percentile: Optional[float] = None,
    mode: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    if not videos or not mode:
        return videos

    if percentile is None:
        percentile = 0.2

    if mode not in ("bottom", "top", "both"):
        return videos

    scored = [(v, _metric_value(v, metric)) for v in videos]
    scored.sort(key=lambda x: x[1], reverse=True)

    k = max(1, int(len(scored) * float(percentile)))

    if mode == "top":
        selected = [v for v, _ in scored[:k]]
    elif mode == "bottom":
        selected = [v for v, _ in scored[-k:]]
    else:
        top = [v for v, _ in scored[:k]]
        bottom = [v for v, _ in scored[-k:]]
        selected = top + bottom

    if limit and len(selected) > limit:
        if mode == "both":
            half = max(1, limit // 2)
            selected = top[:half] + bottom[: max(1, limit - half)]
        else:
            selected = selected[:limit]

    return selected


@trace
def fetch_raw_videos(
    creator_handle: str,
    video_limit: int = 30,
    scan_limit: int = 30,
    selection_mode: Optional[str] = None,
    selection_percentile: Optional[float] = None,
    selection_metric: Optional[str] = None,
):
    """
    Fetch raw video metadata AND ensure mp4 files exist locally.

    Returns:
        List[dict]: yt-dlp metadata with added `local_path`
    """
    videos = fetch_metadata(creator_handle, scan_limit)

    selected = select_videos(
        videos=videos,
        metric=selection_metric or "views",
        percentile=selection_percentile,
        mode=selection_mode,
        limit=video_limit,
    )

    if selection_mode:
        fetch_captions_for_videos(creator_handle, selected)
        download_selected_videos(creator_handle, selected)
        return selected

    fetch_captions(creator_handle, video_limit)

    download_missing_videos(
        creator_handle=creator_handle,
        videos=selected,
        video_limit=video_limit,
    )

    return selected
