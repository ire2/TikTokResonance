import json

from resonance.artifact_manifest import build_artifact_manifest


def test_artifact_manifest_tracks_video_artifact_status(tmp_path):
    (tmp_path / "raw_data").mkdir(parents=True)
    (tmp_path / "raw_captions").mkdir(parents=True)
    (tmp_path / "raw_visual").mkdir(parents=True)
    (tmp_path / "labels").mkdir(parents=True)
    (tmp_path / "embeddings_store").mkdir(parents=True)

    (tmp_path / "raw_data" / "creator_metadata.json").write_text(
        json.dumps({
            "creator": [
                {
                    "video_id": "v1",
                    "local_path": str(tmp_path / "raw_videos" / "creator_v1.mp4"),
                    "views": 100,
                }
            ]
        }),
        encoding="utf-8",
    )
    (tmp_path / "raw_captions" / "creator_v1.json").write_text(
        json.dumps({"text": "caption"}),
        encoding="utf-8",
    )
    (tmp_path / "raw_visual" / "creator.json").write_text(
        json.dumps({"v1": {"motion_intensity": 0.1}}),
        encoding="utf-8",
    )
    (tmp_path / "labels" / "format_labels.csv").write_text(
        "\n".join([
            "creator_id,video_id,format_label,performance_label,tiktok_url",
            "creator,v1,explainer,hit,https://www.tiktok.com/@creator/video/v1",
        ]),
        encoding="utf-8",
    )
    (tmp_path / "embeddings_store" / "creator_model.segments.json").write_text(
        json.dumps([{"video_id": "v1", "text": "segment"}]),
        encoding="utf-8",
    )

    manifest = build_artifact_manifest(tmp_path)
    row = manifest["videos"][0]

    assert manifest["schema_version"] == "media_manifest.v1"
    assert row["creator_id"] == "creator"
    assert row["video_id"] == "v1"
    assert row["metadata_present"] is True
    assert row["caption_path"].endswith("raw_captions/creator_v1.json")
    assert row["visual_artifact_path"].endswith("raw_visual/creator.json")
    assert row["embedding_status"] == "segment_memory"
    assert row["label_status"] == "human"
    assert row["coverage_tier"] == "deep_style_ready"
    assert row["source_url"] == "https://www.tiktok.com/@creator/video/v1"
