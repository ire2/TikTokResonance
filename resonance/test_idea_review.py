import json

import pytest

from resonance.idea_review import IdeaReviewError, analyze_pasted_idea


def _write_review_fixture(data_dir):
    creator_id = "cleo"
    (data_dir / "drafts").mkdir(parents=True)
    (data_dir / "labels").mkdir(parents=True)
    (data_dir / "embeddings_store").mkdir(parents=True)

    (data_dir / "drafts" / f"{creator_id}_draft.yaml").write_text(
        "creator_id: cleo\nobserved_patterns:\n  dominant_formats:\n    - science_explainer\n",
        encoding="utf-8",
    )
    (data_dir / "labels" / "format_labels.csv").write_text(
        "\n".join([
            "creator_id,video_id,format_label,performance_label,tiktok_url",
            "cleo,v1,science_explainer,hit,https://www.tiktok.com/@cleo/video/v1",
            "cleo,v2,science_explainer,miss,https://www.tiktok.com/@cleo/video/v2",
        ]),
        encoding="utf-8",
    )
    (data_dir / "embeddings_store" / f"{creator_id}_BAAI_test.json").write_text(
        json.dumps({"num_segments": 2}),
        encoding="utf-8",
    )
    (data_dir / "embeddings_store" / f"{creator_id}_BAAI_test.segments.json").write_text(
        json.dumps([
            {
                "video_id": "v1",
                "text": "A science explainer about rockets, gravity, and space safety.",
            },
            {
                "video_id": "v2",
                "text": "A confusing rocket bit that never explains the takeaway.",
            },
        ]),
        encoding="utf-8",
    )
    return creator_id


def test_analyze_pasted_idea_uses_local_segments_and_label_evidence(tmp_path):
    creator_id = _write_review_fixture(tmp_path)

    result = analyze_pasted_idea(
        creator_id=creator_id,
        idea_text="Explain rocket safety and gravity with a clear takeaway.",
        data_dir=tmp_path,
    )

    assert result["creator_id"] == creator_id
    assert result["analysis_mode"] == "local_segment_overlap"
    assert result["resonance"]["resonance_score"] > 0
    assert result["evidence"][0]["video_id"] == "v1"
    assert result["hit_evidence"][0]["video_id"] == "v1"
    assert result["miss_evidence"][0]["video_id"] == "v2"
    assert result["label_summary"]["hit"] == 1
    assert result["label_summary"]["miss"] == 1
    assert "not a virality prediction" in result["analysis_note"]


def test_analyze_pasted_idea_validates_creator_and_text(tmp_path):
    with pytest.raises(IdeaReviewError):
        analyze_pasted_idea(
            creator_id="missing",
            idea_text="test",
            data_dir=tmp_path,
        )

    _write_review_fixture(tmp_path)
    with pytest.raises(IdeaReviewError):
        analyze_pasted_idea(
            creator_id="cleo",
            idea_text=" ",
            data_dir=tmp_path,
        )
