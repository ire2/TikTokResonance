import csv

from fastapi.testclient import TestClient

from profiling.label_ui import app as label_app


def _write_label_queue(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "creator_id",
                "video_id",
                "format_label",
                "performance_label",
                "tiktok_url",
                "views",
                "likes",
                "comments",
                "duration_sec",
                "posted_at",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "creator_id": "expoparker",
            "video_id": "v1",
            "format_label": "",
            "performance_label": "",
            "tiktok_url": "https://www.tiktok.com/@expoparker/video/v1",
            "views": "100",
            "likes": "10",
            "comments": "1",
            "duration_sec": "15",
            "posted_at": "20260515",
        })


def test_label_endpoint_validates_allowed_labels(tmp_path, monkeypatch):
    queue_path = tmp_path / "format_labels.csv"
    _write_label_queue(queue_path)
    monkeypatch.setattr(label_app, "CSV_PATH", queue_path)

    client = TestClient(label_app.app)
    response = client.post(
        "/label",
        data={
            "creator_id": "expoparker",
            "video_id": "v1",
            "format_label": "not_a_real_format",
            "performance_label": "hit",
        },
        follow_redirects=False,
    )

    assert response.status_code == 400


def test_label_endpoint_persists_valid_review(tmp_path, monkeypatch):
    queue_path = tmp_path / "format_labels.csv"
    _write_label_queue(queue_path)
    monkeypatch.setattr(label_app, "CSV_PATH", queue_path)

    client = TestClient(label_app.app)
    response = client.post(
        "/label",
        data={
            "creator_id": "expoparker",
            "video_id": "v1",
            "format_label": "skit_or_comedy",
            "performance_label": "hit",
        },
        follow_redirects=False,
    )

    rows = list(csv.DictReader(queue_path.open()))
    assert response.status_code == 303
    assert rows[0]["format_label"] == "skit_or_comedy"
    assert rows[0]["performance_label"] == "hit"
