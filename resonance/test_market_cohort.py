import json

from resonance.market_cohort import (
    build_market_index,
    calculate_coverage_tier,
    infer_creator_local_performance_labels,
    recommend_next_processing_step,
)


def test_creator_local_hit_ok_miss_normalization_does_not_compare_globally():
    small_creator = [
        {"video_id": f"s{i}", "views": views}
        for i, views in enumerate([10, 20, 30, 40, 50])
    ]
    large_creator = [
        {"video_id": f"l{i}", "views": views}
        for i, views in enumerate([1000, 2000, 3000, 4000, 5000])
    ]

    small_labels = infer_creator_local_performance_labels(small_creator)
    large_labels = infer_creator_local_performance_labels(large_creator)

    assert small_labels["s4"]["label"] == "hit"
    assert small_labels["s0"]["label"] == "miss"
    assert large_labels["l4"]["label"] == "hit"
    assert large_labels["l0"]["label"] == "miss"


def test_human_labels_are_preserved_when_normalizing_missing_labels():
    videos = [
        {"video_id": f"v{i}", "views": views}
        for i, views in enumerate([10, 20, 30, 40, 50])
    ]

    labels = infer_creator_local_performance_labels(
        videos,
        {"v0": "hit"},
    )

    assert labels["v0"] == {"label": "hit", "source": "human"}
    assert labels["v4"] == {
        "label": "hit",
        "source": "inferred_creator_local_views",
    }


def test_coverage_tier_and_next_processing_step():
    assert calculate_coverage_tier(
        metadata_count=10,
        caption_count=0,
        visual_count=0,
        label_count=0,
        embedding_video_count=0,
    ) == "metadata_only"
    assert calculate_coverage_tier(
        metadata_count=10,
        caption_count=10,
        visual_count=0,
        label_count=0,
        embedding_video_count=10,
    ) == "semantic_ready"
    assert calculate_coverage_tier(
        metadata_count=10,
        caption_count=10,
        visual_count=10,
        label_count=5,
        embedding_video_count=10,
    ) == "deep_style_ready"
    assert recommend_next_processing_step(
        video_count=10,
        caption_count=0,
        visual_count=0,
        label_count=0,
        embedding_video_count=0,
    ) == "generate_captions"
    assert recommend_next_processing_step(
        video_count=10,
        caption_count=10,
        visual_count=10,
        label_count=5,
        embedding_video_count=10,
    ) == "ready_for_idea_review"


def _write_market_fixture(data_dir):
    (data_dir / "raw_data").mkdir(parents=True)
    (data_dir / "raw_captions").mkdir(parents=True)
    (data_dir / "raw_visual").mkdir(parents=True)
    (data_dir / "labels").mkdir(parents=True)
    (data_dir / "embeddings_store").mkdir(parents=True)
    (data_dir / "drafts").mkdir(parents=True)

    metadata = {
        "deep": [
            {
                "video_id": f"d{i}",
                "views": (i + 1) * 100,
                "likes": (i + 1) * 10,
                "comments": i,
                "format_pred": "explainer",
            }
            for i in range(5)
        ],
        "metadata": [
            {"video_id": f"m{i}", "views": (i + 1) * 10, "format_pred": "skit"}
            for i in range(5)
        ],
    }
    (data_dir / "raw_data" / "creator_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    for i in range(5):
        (data_dir / "raw_captions" / f"deep_d{i}.json").write_text(
            json.dumps({"text": "caption"}),
            encoding="utf-8",
        )
    (data_dir / "raw_visual" / "deep.json").write_text(
        json.dumps({f"d{i}": {"motion_intensity": 0.1} for i in range(5)}),
        encoding="utf-8",
    )
    (data_dir / "labels" / "format_labels.csv").write_text(
        "\n".join([
            "creator_id,video_id,format_label,performance_label,tiktok_url",
            "deep,d0,explainer,miss,",
            "deep,d1,explainer,ok,",
            "deep,d2,explainer,ok,",
            "deep,d3,explainer,hit,",
            "deep,d4,explainer,hit,",
        ]),
        encoding="utf-8",
    )
    (data_dir / "embeddings_store" / "deep_model.segments.json").write_text(
        json.dumps([{"video_id": f"d{i}", "text": "segment"} for i in range(5)]),
        encoding="utf-8",
    )
    (data_dir / "drafts" / "deep_draft.yaml").write_text(
        "\n".join([
            "creator_id: deep",
            "profile_nlp:",
            "  topic_profile:",
            "    topics:",
            "      - top_terms: [science, market, test]",
        ]),
        encoding="utf-8",
    )


def test_market_index_summary_and_processing_queue(tmp_path):
    _write_market_fixture(tmp_path)

    index = build_market_index(tmp_path)
    summary = index["summary"]
    creators = {c["creator_id"]: c for c in index["creators"]}

    assert summary["total_creators"] == 2
    assert summary["total_videos"] == 10
    assert summary["creators_by_coverage_tier"]["deep_style_ready"] == 1
    assert summary["formats_by_hit_rate"][0]["format"] == "explainer"
    assert summary["creators_ready_for_idea_review"] == ["deep"]
    assert summary["creators_needing_captions"] == ["metadata"]
    assert creators["deep"]["hit_ok_miss_distribution"] == {
        "hit": 2,
        "ok": 2,
        "miss": 1,
    }
    assert creators["metadata"]["recommended_next_processing_step"] == "generate_captions"
