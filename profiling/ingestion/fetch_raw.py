import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional

import os
from utils.trace import trace


# Canonical location for raw videos
RAW_VIDEO_DIR = Path("data/raw_videos")
RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
MP4_FORMAT_SELECTOR = "bv*[ext=mp4]+ba[ext=m4a]/bv*[ext=mp4]/b[ext=mp4]"


def _yt_dlp_common_args() -> List[str]:
    args: List[str] = []
    cookies = os.getenv("YT_DLP_COOKIES")
    if cookies:
        args += ["--cookies", cookies]
    impersonate = os.getenv("YT_DLP_IMPERSONATE")
    if impersonate:
        args += ["--impersonate", impersonate]
    extra = os.getenv("YT_DLP_ARGS")
    if extra:
        args += extra.split()
    return args


def _video_mp4_path(creator_handle: str, video_id: str) -> Path:
    return RAW_VIDEO_DIR / f"{creator_handle}_{video_id}.mp4"


def _download_video_mp4(url: str, out_template: str) -> bool:
    cmd = [
        "yt-dlp",
        "-f", MP4_FORMAT_SELECTOR,
        "--merge-output-format", "mp4",
        "-o", out_template,
        url,
    ] + _yt_dlp_common_args()

    result = subprocess.run(cmd)
    return result.returncode == 0


@trace
def fetch_metadata(
    creator_handle: str,
    video_limit: int,
):
    """
    Fetch metadata only (no downloads).
    """

    url = f"https://www.tiktok.com/@{creator_handle}"
    print(
        f"[INGEST][{creator_handle}] Scanning metadata (limit={video_limit})")

    cmd = [
        "yt-dlp",
        "--dump-json",
        "--skip-download",
        "--playlist-end", str(video_limit),
        url,
    ] + _yt_dlp_common_args()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    videos = []

    count = 0
    for line in process.stdout:
        try:
            videos.append(json.loads(line))
            count += 1
            if count % 10 == 0:
                print(f"[INGEST][{creator_handle}] metadata scanned: {count}")
        except json.JSONDecodeError:
            continue
    stderr = process.stderr.read() if process.stderr else ""
    returncode = process.wait()

    print(f"[INGEST][{creator_handle}] metadata total: {count}")
    if returncode != 0:
        print(
            f"[INGEST][{creator_handle}] yt-dlp exited with code {returncode}")
    if count == 0 and stderr:
        snippet = stderr.strip().splitlines()[-5:]
        print(f"[INGEST][{creator_handle}] yt-dlp stderr (last lines):")
        for line in snippet:
            print(f"[INGEST][{creator_handle}] {line}")

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
    ] + _yt_dlp_common_args()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(
            f"[INGEST][{creator_handle}] captions download failed (exit {result.returncode})")


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
    ] + _yt_dlp_common_args() + urls

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
    downloaded = []

    for v in videos:
        video_id = v["id"]
        path = _video_mp4_path(creator_handle, video_id)

        v["local_path"] = str(path)
        v["downloaded"] = path.exists()

        if not v["downloaded"]:
            missing.append(v)
        else:
            downloaded.append(v)

    if not missing:
        print("[INGEST] All videos already present. Skipping download.")
        return downloaded

    print(f"[INGEST] Downloading {len(missing)} videos...")

    for v in missing:
        video_id = v["id"]
        url = f"https://www.tiktok.com/@{creator_handle}/video/{video_id}"
        out_template = str(RAW_VIDEO_DIR / f"{creator_handle}_%(id)s.%(ext)s")
        ok = _download_video_mp4(url, out_template)
        v["downloaded"] = ok and _video_mp4_path(
            creator_handle, video_id).exists()
        if v["downloaded"]:
            downloaded.append(v)
        else:
            print(
                f"[INGEST] Skipping {video_id}: no mp4 format available.")

    return downloaded


@trace
def download_selected_videos(
    creator_handle: str,
    videos: List[Dict],
    limit: Optional[int] = None,
):
    """
    Download mp4 files for specific video IDs only.
    """
    if not videos:
        print("[INGEST] No videos selected. Skipping download.")
        return

    missing = []
    downloaded = []

    for v in videos:
        video_id = v["id"]
        path = _video_mp4_path(creator_handle, video_id)
        v["local_path"] = str(path)
        v["downloaded"] = path.exists()
        if not v["downloaded"]:
            missing.append(v)
        else:
            downloaded.append(v)

    if not missing:
        print("[INGEST] All selected videos already present. Skipping download.")
        return downloaded

    print(f"[INGEST] Downloading {len(missing)} selected videos...")

    for v in missing:
        if limit and len(downloaded) >= limit:
            break
        video_id = v["id"]
        url = f"https://www.tiktok.com/@{creator_handle}/video/{video_id}"
        out_template = str(RAW_VIDEO_DIR / f"{creator_handle}_%(id)s.%(ext)s")
        ok = _download_video_mp4(url, out_template)
        v["downloaded"] = ok and _video_mp4_path(
            creator_handle, video_id).exists()
        if v["downloaded"]:
            downloaded.append(v)
        else:
            print(
                f"[INGEST] Skipping {video_id}: no mp4 format available.")

    return downloaded


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

    if not selected:
        print(
            f"[INGEST][{creator_handle}] No videos found in metadata. Skipping captions/download.")
        return []

    if selection_mode:
        selected_ids = {v.get("id") for v in selected if v.get("id")}
        pool = list(selected)
        # Always allow backfill from the full scan pool
        for v in videos:
            vid = v.get("id")
            if vid and vid not in selected_ids:
                pool.append(v)

        downloaded = download_selected_videos(
            creator_handle, pool, limit=video_limit)
        fetch_captions_for_videos(creator_handle, downloaded)
        return downloaded

    downloaded = download_missing_videos(
        creator_handle=creator_handle,
        videos=selected,
        video_limit=video_limit,
    )

    fetch_captions_for_videos(creator_handle, downloaded)
    return downloaded
