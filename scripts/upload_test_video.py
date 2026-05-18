from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen


TEST_VIDEO_DIR = Path("data/test/video")
MP4_FORMAT_SELECTOR = "bv*[ext=mp4]+ba[ext=m4a]/bv*[ext=mp4]/b[ext=mp4]"
DEFAULT_DASHBOARD_URLS = (
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003",
)
CREATOR_ALIASES = {
    "cleoabram": "cleoabram",
    "cleoabrams": "cleoabram",
    "graydavis": "gray.davis",
}


def _clean_creator_id(creator_id: str) -> str:
    creator_id = (creator_id or "").strip().lstrip("@")
    alias_key = re.sub(r"[^A-Za-z0-9]+", "", creator_id).lower()
    creator_id = CREATOR_ALIASES.get(alias_key, creator_id)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", creator_id):
        raise ValueError("creator must look like a TikTok handle, e.g. gray.davis")
    return creator_id


def _video_ref_to_url(creator_id: str, video_ref: str) -> str:
    video_ref = (video_ref or "").strip()
    if re.fullmatch(r"\d{8,}", video_ref):
        return f"https://www.tiktok.com/@{creator_id}/video/{video_ref}"
    if re.search(r"https?://", video_ref):
        return video_ref
    raise ValueError("second argument must be a TikTok URL or numeric video id")


def _creator_from_tiktok_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path or ""
    match = re.search(r"/@([^/]+)/video/\d+", path)
    if not match:
        raise ValueError("could not read creator from TikTok URL")
    return _clean_creator_id(match.group(1))


def _parse_upload_request(creator_or_url: str, video_ref: str | None) -> tuple[str, str]:
    if video_ref is None:
        if not re.search(r"https?://", creator_or_url or ""):
            raise ValueError("usage: upload <creator> <video id> or upload <TikTok URL>")
        creator_id = _creator_from_tiktok_url(creator_or_url)
        return creator_id, creator_or_url

    creator_id = _clean_creator_id(creator_or_url)
    return creator_id, _video_ref_to_url(creator_id, video_ref)


def _yt_dlp_common_args() -> list[str]:
    args: list[str] = []
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


def _newest_downloaded_path(creator_id: str, before: set[Path]) -> Path | None:
    candidates = [
        p for p in TEST_VIDEO_DIR.glob(f"{creator_id}_*.*")
        if p not in before and p.is_file()
    ]
    if not candidates:
        candidates = [
            p for p in TEST_VIDEO_DIR.glob(f"{creator_id}_*.*")
            if p.is_file()
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def download_video(creator_id: str, url: str) -> Path:
    creator_id = _clean_creator_id(creator_id)
    if not url.strip():
        raise ValueError("url is required")
    TEST_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    before = set(TEST_VIDEO_DIR.glob(f"{creator_id}_*.*"))
    output_template = str(TEST_VIDEO_DIR / f"{creator_id}_%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f",
        MP4_FORMAT_SELECTOR,
        "--merge-output-format",
        "mp4",
        "--print",
        "after_move:filepath",
        "-o",
        output_template,
        url,
    ] + _yt_dlp_common_args()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or f"yt-dlp failed with exit {result.returncode}")

    printed_paths = [
        Path(line.strip())
        for line in (result.stdout or "").splitlines()
        if line.strip()
    ]
    for path in reversed(printed_paths):
        if path.exists():
            return path

    path = _newest_downloaded_path(creator_id, before)
    if path is None:
        raise RuntimeError("yt-dlp finished, but no downloaded video file was found")
    return path


def _candidate_dashboard_urls(explicit_url: str | None) -> list[str]:
    if explicit_url:
        return [explicit_url.rstrip("/")]
    env_url = os.getenv("RESONANCE_DASHBOARD_URL")
    if env_url:
        return [env_url.rstrip("/")]
    return [url.rstrip("/") for url in DEFAULT_DASHBOARD_URLS]


def _looks_like_dashboard(base_url: str, timeout: float = 1.5) -> bool:
    try:
        with urlopen(f"{base_url}/", timeout=timeout) as response:
            html = response.read(250_000).decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError, OSError):
        return False
    return "Creator Strategy Workspace" in html and "/api/video-review" in html


def find_dashboard_url(explicit_url: str | None = None) -> str | None:
    for url in _candidate_dashboard_urls(explicit_url):
        if _looks_like_dashboard(url):
            return url
    return None


def upload_to_dashboard(base_url: str, creator_id: str, video_path: Path) -> dict[str, Any]:
    if shutil.which("curl") is None:
        raise RuntimeError("curl is required to upload to the running dashboard")

    endpoint = f"{base_url.rstrip('/')}/api/video-review"
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "-X",
            "POST",
            "-F",
            f"creator_id={creator_id}",
            "-F",
            f"video=@{video_path}",
            endpoint,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "").strip() or "dashboard upload failed")
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"dashboard returned non-JSON response: {result.stdout[:200]}") from exc
    if payload.get("error") or payload.get("detail"):
        raise RuntimeError(str(payload.get("detail") or payload.get("error")))
    return payload


def _print_payload_summary(payload: dict[str, Any]) -> None:
    resonance = payload.get("resonance") or {}
    readout = payload.get("transcript_readout") or {}
    print(f"[upload] score: {resonance.get('resonance_score', 'n/a')}")
    if readout.get("summary"):
        print(f"[upload] summary: {readout['summary']}")
    if readout.get("word_count"):
        print(f"[upload] transcript: {readout['word_count']} words")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="upload",
        description="Download a TikTok test video and upload it to the local demo dashboard if it is running.",
    )
    parser.add_argument(
        "creator_or_url",
        help="Creator handle, or a full TikTok URL to auto-detect the creator.",
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="TikTok URL or numeric video id when the first argument is a creator.",
    )
    parser.add_argument(
        "--dashboard-url",
        help="Override dashboard URL. Defaults to detecting localhost ports 8000, 8002, 8003.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the video into data/test/video.",
    )
    args = parser.parse_args(argv)

    try:
        creator_id, url = _parse_upload_request(args.creator_or_url, args.video)
        print(f"[upload] downloading {creator_id} video...")
        video_path = download_video(creator_id, url)
        print(f"[upload] downloaded: {video_path}")

        if args.download_only:
            return 0

        dashboard_url = find_dashboard_url(args.dashboard_url)
        if not dashboard_url:
            print("[upload] no running dashboard found; downloaded only")
            print("[upload] start one with: make dashboard-demo PORT=8000")
            return 0

        print(f"[upload] uploading to dashboard: {dashboard_url}")
        payload = upload_to_dashboard(dashboard_url, creator_id, video_path)
        print("[upload] dashboard analysis complete")
        _print_payload_summary(payload)
        return 0
    except Exception as exc:
        print(f"[upload] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
