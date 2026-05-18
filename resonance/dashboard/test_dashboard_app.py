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

    (data_dir / "drafts" / f"{creator_id}_draft.yaml").write_text(
        "creator_id: expoparker\nobserved_patterns:\n  dominant_formats:\n    - skit_or_comedy\n",
        encoding="utf-8",
    )
    for idx in range(5):
        (data_dir / "raw_captions" / f"{creator_id}_v{idx}.json").write_text(
            json.dumps({"text": f"caption {idx}"}),
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
    assert "Creator Library" in response.text
    assert "Idea Review" in response.text
    assert "Human Decision" in response.text
    assert "decision-approve" in response.text
    assert "ideaInput" in response.text
    assert "videoInput" in response.text
    assert "/api/video-review" in response.text
    assert "reviewNotes" in response.text
    assert "Transcript & Content Readout" in response.text
    assert "transcriptPanel" in response.text
    assert "View transcript/data" in response.text
    assert "compactEvidenceText" in response.text


def test_creator_library_endpoint_returns_curated_cohort(tmp_path, monkeypatch):
    _write_creator_fixture(tmp_path)
    monkeypatch.setattr(dashboard_app, "DATA_DIR", tmp_path)

    response = TestClient(dashboard_app.app).get("/api/creators")

    assert response.status_code == 200
    data = response.json()
    assert data["cohort_label"] == "Curated demo cohort"
    assert data["creators"][0]["creator_id"] == "expoparker"
    assert data["creators"][0]["confidence_level"] == "medium"


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


def test_demo_idea_endpoint_masks_explicit_evidence_text(tmp_path, monkeypatch):
    creator_id = _write_creator_fixture(tmp_path)
    segments_path = tmp_path / "embeddings_store" / f"{creator_id}_BAAI_test.segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    segments[0]["text"] = "A sharp fucking joke about awkward dating advice."
    segments_path.write_text(json.dumps(segments), encoding="utf-8")

    monkeypatch.setattr(dashboard_app, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dashboard_app, "DEMO_MODE", True)

    response = TestClient(dashboard_app.app).post(
        "/api/idea-review",
        json={
            "creator_id": "expoparker",
            "idea_text": "A sharp awkward dating joke with a clear setup.",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["presentation_safe_text"] is True
    assert "fucking" not in response.text.lower()
    assert "f******" in data["evidence"][0]["text"]


def test_video_review_endpoint_scores_uploaded_transcript(tmp_path, monkeypatch):
    _write_creator_fixture(tmp_path)
    draft_path = tmp_path / "drafts" / "expoparker_draft.yaml"
    draft_path.write_text(
        "\n".join([
            "creator_id: expoparker",
            "observed_patterns:",
            "  dominant_formats:",
            "    - skit_or_comedy",
            "visual_signals:",
            "  avg_motion_intensity: 0.2",
            "  avg_text_density_heuristic: 0.3",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard_app, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dashboard_app, "DRAFTS_DIR", tmp_path / "drafts")
    monkeypatch.setattr(dashboard_app, "TEST_VIDEO_DIR", tmp_path / "test" / "video")
    monkeypatch.setattr(
        dashboard_app,
        "_extract_idea_text",
        lambda video_path: "A sharp awkward dating joke with a clear setup.",
    )
    monkeypatch.setattr(
        dashboard_app,
        "extract_visual_signals",
        lambda *args, **kwargs: {
            "motion_intensity": 0.25,
            "text_density_heuristic": 0.4,
        },
    )

    response = TestClient(dashboard_app.app).post(
        "/api/video-review",
        data={"creator_id": "expoparker"},
        files={"video": ("clip.mp4", b"fake video bytes", "video/mp4")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["analysis_mode"] == "uploaded_video_transcript"
    assert data["creator_id"] == "expoparker"
    assert data["evidence"][0]["performance_label"] == "hit"
    assert data["video_path"].endswith("clip.mp4")
    assert data["transcript_readout"]["summary"].startswith(
        "The clip reads as a short spoken"
    )
    assert data["transcript_readout"]["paragraphs"][0].startswith("A sharp awkward")
    assert data["transcript_readout"]["word_count"] > 0
    assert data["resonance"]["format_alignment"] == 1.0
    assert data["resonance"]["motion_alignment"] == 0.95
    assert data["resonance"]["text_density_alignment"] == 0.9
    assert data["inferred_format"] == "skit_or_comedy"
    assert data["upload_metric_sources"]["format"] == "uploaded_transcript_keywords"


def test_video_review_warns_when_uploaded_source_creator_differs_from_target(tmp_path, monkeypatch):
    for dirname in ["drafts", "labels", "raw_captions", "raw_visual", "embeddings_store"]:
        (tmp_path / dirname).mkdir(parents=True, exist_ok=True)
    (tmp_path / "drafts" / "cleoabram_draft.yaml").write_text(
        "\n".join([
            "creator_id: cleoabram",
            "observed_patterns:",
            "  dominant_formats:",
            "    - educational",
            "visual_signals:",
            "  avg_motion_intensity: 0.2",
            "  avg_text_density_heuristic: 0.3",
        ]),
        encoding="utf-8",
    )
    (tmp_path / "drafts" / "gray.davis_draft.yaml").write_text(
        "\n".join([
            "creator_id: gray.davis",
            "observed_patterns:",
            "  dominant_formats:",
            "    - food_or_cooking",
            "visual_signals:",
            "  avg_motion_intensity: 0.2",
            "  avg_text_density_heuristic: 0.3",
        ]),
        encoding="utf-8",
    )
    (tmp_path / "labels" / "format_labels.csv").write_text(
        "\n".join([
            "creator_id,video_id,format_label,performance_label,tiktok_url",
            "cleoabram,c1,educational,hit,https://www.tiktok.com/@cleoabram/video/c1",
            "gray.davis,123,food_or_cooking,hit,https://www.tiktok.com/@gray.davis/video/123",
        ]),
        encoding="utf-8",
    )
    (tmp_path / "embeddings_store" / "cleoabram_BAAI_test.segments.json").write_text(
        json.dumps([
            {"video_id": "c1", "text": "A science explainer about space and technology."},
        ]),
        encoding="utf-8",
    )

    monkeypatch.setattr(dashboard_app, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dashboard_app, "DRAFTS_DIR", tmp_path / "drafts")
    monkeypatch.setattr(dashboard_app, "TEST_VIDEO_DIR", tmp_path / "test" / "video")
    monkeypatch.setattr(
        dashboard_app,
        "_extract_idea_text",
        lambda video_path: "Have you ever tried this fruit science technology from the garden?",
    )
    monkeypatch.setattr(
        dashboard_app,
        "extract_visual_signals",
        lambda *args, **kwargs: {
            "motion_intensity": 0.25,
            "text_density_heuristic": 0.4,
        },
    )

    response = TestClient(dashboard_app.app).post(
        "/api/video-review",
        data={"creator_id": "cleoabram"},
        files={"video": ("gray.davis_123.mp4", b"fake video bytes", "video/mp4")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["uploaded_format"]["source_creator_id"] == "gray.davis"
    assert data["inferred_format"] == "food_or_cooking"
    assert data["resonance"]["format_alignment"] == 0.35
    assert data["resonance"]["resonance_score"] < data["upload_score_adjustment"]["original_score"]
    assert data["upload_score_adjustment"]["reason"] == "source creator mismatch and low target format fit"
    assert "gray.davis" in data["upload_warning"]
    assert "cleoabram" in data["upload_warning"]


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


def test_review_decision_endpoint_can_save_current_upload_analysis(tmp_path, monkeypatch):
    decisions_path = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(dashboard_app, "DEMO_MODE", True)
    monkeypatch.setattr(dashboard_app, "REVIEW_DECISIONS_PATH", decisions_path)

    response = TestClient(dashboard_app.app).post(
        "/api/review-decision",
        json={
            "decision": "approve",
            "notes": "Uploaded video fits the creator pattern.",
            "analysis_payload": {
                "creator_id": "expoparker",
                "video_path": "data/test/video/uploads/clip.mp4",
                "idea_text": "A sharp awkward dating joke with a clear setup.",
                "analysis_mode": "uploaded_video_transcript",
                "model_name": "local_segment_overlap",
                "resonance": {"resonance_score": 0.42, "semantic_alignment": 0.5},
                "evidence": [{"video_id": "v1"}],
            },
        },
    )

    assert response.status_code == 200
    saved = [json.loads(line) for line in decisions_path.read_text().splitlines()]
    assert saved[0]["source"] == "demo-upload"
    assert saved[0]["video_path"].endswith("clip.mp4")
