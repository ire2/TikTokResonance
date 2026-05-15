import numpy as np

from resonance.resonance_score import compute_resonance


def test_compute_resonance_combines_semantic_format_and_visual_signals():
    payload = {
        "creator_embedding": np.array([1.0, 0.0]),
        "segments": [
            {"video_id": "v1", "text": "same idea", "embedding": np.array([1.0, 0.0])},
            {"video_id": "v2", "text": "opposite idea", "embedding": np.array([0.0, 1.0])},
        ],
    }
    profile = {
        "profile_nlp": {
            "dialogue_video_ratio": 0.8,
            "solo_rant_ratio": 0.2,
        },
        "visual_signals": {
            "avg_talking_head_confidence": 0.7,
            "avg_motion_intensity": 0.25,
            "avg_text_density_heuristic": 0.3,
        },
        "observed_patterns": {
            "dominant_formats": ["talking_head"],
            "underused_formats": ["dance"],
        },
    }

    result = compute_resonance(
        idea_embedding=np.array([1.0, 0.0]),
        creator_embedding_payload=payload,
        creator_profile=profile,
        idea_motion_intensity=0.3,
        idea_text_density=0.35,
        idea_format="talking_head",
    )

    assert result["semantic_alignment"] == 0.5
    assert result["format_alignment"] == 1.0
    assert result["motion_alignment"] == 0.95
    assert result["text_density_alignment"] == 0.95
    assert result["semantic_gate"] == 1.0
    assert result["resonance_score"] > 0.4
    assert [e["video_id"] for e in result["evidence"]] == ["v1", "v2"]
