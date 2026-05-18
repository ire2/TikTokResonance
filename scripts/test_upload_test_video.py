from pathlib import Path
import subprocess

import pytest

from scripts import upload_test_video


def test_clean_creator_id_accepts_plain_or_at_handle():
    assert upload_test_video._clean_creator_id("gray.davis") == "gray.davis"
    assert upload_test_video._clean_creator_id("@gray.davis") == "gray.davis"
    assert upload_test_video._clean_creator_id("cleoabrams") == "cleoabram"


def test_clean_creator_id_rejects_shell_like_input():
    with pytest.raises(ValueError):
        upload_test_video._clean_creator_id("gray.davis;rm")


def test_video_ref_to_url_accepts_id_or_full_url():
    assert upload_test_video._video_ref_to_url(
        "cleoabram",
        "7180009782248754478",
    ) == "https://www.tiktok.com/@cleoabram/video/7180009782248754478"
    assert upload_test_video._video_ref_to_url(
        "gray.davis",
        "https://www.tiktok.com/@gray.davis/video/123",
    ) == "https://www.tiktok.com/@gray.davis/video/123"


def test_parse_upload_request_accepts_url_only():
    creator_id, url = upload_test_video._parse_upload_request(
        "https://www.tiktok.com/@cleoabram/video/7180009782248754478?is_from_webapp=1",
        None,
    )

    assert creator_id == "cleoabram"
    assert url.startswith("https://www.tiktok.com/@cleoabram/video/7180009782248754478")


def test_parse_upload_request_rejects_id_only_without_creator():
    with pytest.raises(ValueError):
        upload_test_video._parse_upload_request("7180009782248754478", None)


def test_download_video_uses_demo_folder_and_printed_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    downloaded = Path("data/test/video/gray.davis_123.mp4")

    def fake_run(cmd, capture_output, text):
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        downloaded.write_bytes(b"video")
        assert "yt-dlp" in cmd
        assert "data/test/video/gray.davis_%(id)s.%(ext)s" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{downloaded}\n", stderr="")

    monkeypatch.setattr(upload_test_video.subprocess, "run", fake_run)

    assert upload_test_video.download_video(
        "gray.davis",
        "https://www.tiktok.com/@gray.davis/video/123",
    ) == downloaded


def test_find_dashboard_url_requires_dashboard_html(monkeypatch):
    def fake_looks_like_dashboard(url):
        return url == "http://127.0.0.1:8002"

    monkeypatch.setattr(upload_test_video, "_looks_like_dashboard", fake_looks_like_dashboard)

    assert upload_test_video.find_dashboard_url() == "http://127.0.0.1:8002"
