import json

from resonance.creator_library import build_creator_library, get_creator


def _write_high_coverage_creator(data_dir, creator_id="alice"):
    (data_dir / "drafts").mkdir(parents=True)
    (data_dir / "raw_captions").mkdir(parents=True)
    (data_dir / "raw_visual").mkdir(parents=True)
    (data_dir / "labels").mkdir(parents=True)
    (data_dir / "embeddings_store").mkdir(parents=True)

    (data_dir / "drafts" / f"{creator_id}_draft.yaml").write_text(
        "creator_id: alice\nobserved_patterns:\n  dominant_formats:\n    - explainer\n",
        encoding="utf-8",
    )
    for idx in range(20):
        (data_dir / "raw_captions" / f"{creator_id}_{1000 + idx}.json").write_text(
            json.dumps({"text": f"caption {idx}"}),
            encoding="utf-8",
        )

    visual = {str(1000 + idx): {"motion_intensity": 0.1} for idx in range(10)}
    (data_dir / "raw_visual" / f"{creator_id}.json").write_text(
        json.dumps(visual),
        encoding="utf-8",
    )

    rows = [
        "creator_id,video_id,format_label,performance_label,tiktok_url",
    ]
    labels = ["hit", "hit", "ok", "miss", "hit", "ok", "miss", "hit", "ok", "hit"]
    for idx, label in enumerate(labels):
        rows.append(
            f"{creator_id},{1000 + idx},explainer,{label},https://www.tiktok.com/@{creator_id}/video/{1000 + idx}"
        )
    (data_dir / "labels" / "format_labels.csv").write_text(
        "\n".join(rows),
        encoding="utf-8",
    )

    (data_dir / "embeddings_store" / f"{creator_id}_BAAI_test.json").write_text(
        json.dumps({"num_segments": 2}),
        encoding="utf-8",
    )
    (data_dir / "embeddings_store" / f"{creator_id}_BAAI_test.segments.json").write_text(
        json.dumps([
            {"video_id": "1000", "text": "science explainer"},
            {"video_id": "1001", "text": "another segment"},
        ]),
        encoding="utf-8",
    )


def test_creator_library_computes_coverage_and_confidence(tmp_path):
    _write_high_coverage_creator(tmp_path)

    creators = build_creator_library(tmp_path)
    alice = creators[0]

    assert alice["creator_id"] == "alice"
    assert alice["videos_analyzed"] == 20
    assert alice["visual_signal_count"] == 10
    assert alice["caption_count"] == 20
    assert alice["human_label_count"] == 10
    assert alice["hit_count"] == 5
    assert alice["ok_count"] == 3
    assert alice["miss_count"] == 2
    assert alice["dominant_formats"] == ["explainer"]
    assert alice["embeddings_exist"] is True
    assert alice["segment_count"] == 2
    assert alice["confidence_level"] == "high"


def test_creator_library_degrades_to_low_confidence_with_sparse_artifacts(tmp_path):
    (tmp_path / "drafts").mkdir(parents=True)
    (tmp_path / "drafts" / "sparse_draft.yaml").write_text(
        "creator_id: sparse\n",
        encoding="utf-8",
    )

    creator = get_creator("sparse", tmp_path)

    assert creator["videos_analyzed"] == 0
    assert creator["caption_count"] == 0
    assert creator["human_label_count"] == 0
    assert creator["embeddings_exist"] is False
    assert creator["confidence_level"] == "low"
