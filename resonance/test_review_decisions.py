import json

import pytest

from resonance.review_decisions import (
    MAX_NOTES_CHARS,
    ReviewDecisionError,
    build_review_decision,
    load_review_decisions,
    save_review_decision,
)


def _payload():
    return {
        "creator_id": "expoparker",
        "model_name": "BAAI/bge-large-en-v1.5",
        "video_path": "data/test/video/example.mp4",
        "idea_text": "A short creator idea.",
        "resonance": {
            "resonance_score": 0.42,
            "semantic_alignment": 0.5,
            "format_alignment": 1.0,
            "motion_alignment": 0.8,
            "text_density_alignment": 0.7,
        },
        "evidence": [
            {"video_id": "v1"},
            {"video_id": "v2"},
            {"video_id": "v3"},
            {"video_id": "v4"},
        ],
    }


def test_build_review_decision_keeps_auditable_summary():
    record = build_review_decision(
        _payload(),
        decision="approve",
        notes="Evidence supports this direction.",
        source="demo",
        created_at="2026-05-15T12:00:00Z",
    )

    assert record["decision"] == "approve"
    assert record["notes"] == "Evidence supports this direction."
    assert record["source"] == "demo"
    assert record["creator_id"] == "expoparker"
    assert record["resonance_score"] == 0.42
    assert record["evidence_video_ids"] == ["v1", "v2", "v3"]
    assert len(record["idea_fingerprint"]) == 16


def test_save_review_decision_appends_jsonl_and_loads_recent_first(tmp_path):
    path = tmp_path / "decisions.jsonl"

    save_review_decision(_payload(), "revise", "Tighten the hook.", path=path)
    save_review_decision(_payload(), "reject", "Too weak.", path=path)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["decision"] == "revise"

    recent = load_review_decisions(path, limit=1)
    assert len(recent) == 1
    assert recent[0]["decision"] == "reject"


def test_review_decision_validation_rejects_bad_values():
    with pytest.raises(ReviewDecisionError):
        build_review_decision(_payload(), decision="maybe")

    with pytest.raises(ReviewDecisionError):
        build_review_decision(_payload(), decision="approve", notes="x" * (MAX_NOTES_CHARS + 1))
