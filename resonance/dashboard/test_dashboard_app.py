import json

from fastapi.testclient import TestClient

import resonance.dashboard.app as dashboard_app


def _write_creator_fixture(data_dir):
    creator_id = "expoparker"
    (data_dir / "drafts").mkdir(parents=True)
    (data_dir / "raw_captions").mkdir(parents=True)
    (data_dir / "raw_visual").mkdir(parents=True)
    (data_dir / "labels").mkdir(parents=True)
    (data_dir / "embeddings_store").mkdir(parents=True)
    (data_dir / "raw_data").mkdir(parents=True)

    (data_dir / "drafts" / f"{creator_id}_draft.yaml").write_text(
        "creator_id: expoparker\nobserved_patterns:\n  dominant_formats:\n    - skit_or_comedy\n",
        encoding="utf-8",
    )
    for idx in range(5):
        (data_dir / "raw_captions" / f"{creator_id}_v{idx}.json").write_text(
            json.dumps({"text": f"caption {idx}"}),
            encoding="utf-8",
        )
    (data_dir / "raw_data" / "creator_metadata.json").write_text(
        json.dumps({
            creator_id: [
                {
                    "video_id": f"v{idx}",
                    "views": (idx + 1) * 100,
                    "likes": (idx + 1) * 10,
                    "comments": idx,
                    "format_pred": "skit_or_comedy",
                }
                for idx in range(5)
            ]
        }),
        encoding="utf-8",
    )
    (data_dir / "raw_visual" / f"{creator_id}.json").write_text(
        json.dumps({f"v{idx}": {"motion_intensity": 0.1} for idx in range(5)}),
        encoding="utf-8",
    )
    (data_dir / "labels" / "format_labels.csv").write_text(
        "\n".join([
            "creator_id,video_id,format_label,performance_label,tiktok_url",
            "expoparker,v1,skit_or_comedy,hit,https://www.tiktok.com/@expoparker/video/v1",
            "expoparker,v2,skit_or_comedy,miss,https://www.tiktok.com/@expoparker/video/v2",
            "expoparker,v3,skit_or_comedy,ok,https://www.tiktok.com/@expoparker/video/v3",
        ]),
        encoding="utf-8",
    )
    (data_dir / "embeddings_store" / f"{creator_id}_BAAI_test.json").write_text(
        json.dumps({"num_segments": 2}),
        encoding="utf-8",
    )
    (data_dir / "embeddings_store" / f"{creator_id}_BAAI_test.segments.json").write_text(
        json.dumps([
            {"video_id": "v1", "text": "A sharp joke about awkward dating advice."},
            {"video_id": "v2", "text": "A joke that misses because the setup is confusing."},
        ]),
        encoding="utf-8",
    )
    return creator_id


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
    assert "Creator Strategy Workspace" in response.text
    assert "Market Memory" in response.text
    assert "Coverage tiers" in response.text or "coverage tiers" in response.text
    assert "Creator Library" in response.text
    assert "Idea Review" in response.text
    assert "Human Decision" in response.text
    assert "decision-approve" in response.text
    assert "ideaInput" in response.text
    assert "reviewNotes" in response.text


def test_creator_library_endpoint_returns_curated_cohort(tmp_path, monkeypatch):
    _write_creator_fixture(tmp_path)
    monkeypatch.setattr(dashboard_app, "DATA_DIR", tmp_path)

    response = TestClient(dashboard_app.app).get("/api/creators")

    assert response.status_code == 200
    data = response.json()
    assert data["cohort_label"] == "Curated demo cohort"
    assert data["creators"][0]["creator_id"] == "expoparker"
    assert data["creators"][0]["confidence_level"] == "medium"


def test_market_endpoint_returns_market_memory(tmp_path, monkeypatch):
    _write_creator_fixture(tmp_path)
    monkeypatch.setattr(dashboard_app, "DATA_DIR", tmp_path)

    response = TestClient(dashboard_app.app).get("/api/market")

    assert response.status_code == 200
    data = response.json()
    assert data["schema_version"] == "market_index.v1"
    assert data["summary"]["total_creators"] == 1
    assert data["summary"]["total_videos"] == 5
    assert data["summary"]["formats_by_hit_rate"][0]["format"] == "skit_or_comedy"


def test_paste_idea_endpoint_returns_local_review_payload(tmp_path, monkeypatch):
    _write_creator_fixture(tmp_path)
    monkeypatch.setattr(dashboard_app, "DATA_DIR", tmp_path)

    response = TestClient(dashboard_app.app).post(
        "/api/idea-review",
        json={
            "creator_id": "expoparker",
            "idea_text": "A sharp awkward dating joke with a clear setup.",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_mode"] == "local_segment_overlap"
    assert data["creator_id"] == "expoparker"
    assert data["evidence"][0]["performance_label"] == "hit"


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
    assert saved[0]["reviewer_id"] == "creator_strategist"

    recent = client.get("/api/review-decisions").json()["decisions"]
    assert recent[0]["decision"] == "approve"


def test_review_decision_endpoint_saves_pasted_idea_summary(tmp_path, monkeypatch):
    _write_creator_fixture(tmp_path)
    decisions_path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(dashboard_app, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dashboard_app, "DEMO_MODE", True)
    monkeypatch.setattr(dashboard_app, "REVIEW_DECISIONS_PATH", decisions_path)

    response = TestClient(dashboard_app.app).post(
        "/api/review-decision",
        json={
            "decision": "revise",
            "notes": "Make the setup clearer.",
            "creator_id": "expoparker",
            "idea_text": "A sharp awkward dating joke with a clear setup.",
        },
    )

    assert response.status_code == 200
    saved = [json.loads(line) for line in decisions_path.read_text().splitlines()]
    assert saved[0]["source"] == "demo-local"
    assert saved[0]["creator_id"] == "expoparker"
    assert saved[0]["idea_snippet"].startswith("A sharp awkward dating joke")
