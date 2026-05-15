import json

from fastapi.testclient import TestClient

import resonance.dashboard.app as dashboard_app


def _demo_payload():
    return {
        "creator_id": "expoparker",
        "model_name": "BAAI/bge-large-en-v1.5",
        "video_path": "data/test/video/example.mp4",
        "idea_text": "A short creator idea.",
        "resonance": {
            "semantic_alignment": 0.5,
            "format_alignment": 1.0,
            "motion_alignment": 0.9,
            "text_density_alignment": 0.8,
            "resonance_score": 0.44,
        },
        "interpretation": {
            "semantic_fit": "medium",
            "format_match": "strong",
        },
        "evidence": [
            {"video_id": "v1", "text": "matching moment", "similarity": 0.7}
        ],
        "suggestions": [],
    }


def test_demo_cache_endpoint_loads_cached_resonance(tmp_path, monkeypatch):
    cache_path = tmp_path / "resonance_cache.json"
    cache_path.write_text(json.dumps(_demo_payload()))
    monkeypatch.setattr(dashboard_app, "DEMO_MODE", True)
    monkeypatch.setattr(dashboard_app, "CACHE_PATH", cache_path)

    response = TestClient(dashboard_app.app).get("/api/resonance")

    assert response.status_code == 200
    data = response.json()
    assert data["creator_id"] == "expoparker"
    assert data["resonance"]["resonance_score"] == 0.44
    assert data["evidence"][0]["tiktok_url"] == "https://www.tiktok.com/@expoparker/video/v1"


def test_dashboard_page_exposes_human_review_controls():
    response = TestClient(dashboard_app.app).get("/")

    assert response.status_code == 200
    assert "Human Review Decision" in response.text
    assert "decision-approve" in response.text
    assert "reviewNotes" in response.text


def test_review_decision_endpoint_writes_local_artifact(tmp_path, monkeypatch):
    cache_path = tmp_path / "resonance_cache.json"
    decisions_path = tmp_path / "decisions.jsonl"
    cache_path.write_text(json.dumps(_demo_payload()))
    monkeypatch.setattr(dashboard_app, "DEMO_MODE", True)
    monkeypatch.setattr(dashboard_app, "CACHE_PATH", cache_path)
    monkeypatch.setattr(dashboard_app, "REVIEW_DECISIONS_PATH", decisions_path)

    client = TestClient(dashboard_app.app)
    response = client.post(
        "/api/review-decision",
        json={"decision": "approve", "notes": "Evidence looks strong enough."},
    )

    assert response.status_code == 200
    assert response.json()["saved"] is True
    saved = [json.loads(line) for line in decisions_path.read_text().splitlines()]
    assert saved[0]["decision"] == "approve"
    assert saved[0]["source"] == "demo"
    assert saved[0]["notes"] == "Evidence looks strong enough."

    recent = client.get("/api/review-decisions").json()["decisions"]
    assert recent[0]["decision"] == "approve"
