import json
import yaml
from collections import Counter
from statistics import mean
from utils.trace import trace


@trace
def generate_profile(creator_id: str, raw_data_path: str) -> dict:
    with open(raw_data_path, "r") as f:
        data = json.load(f)

    if creator_id not in data:
        raise ValueError(f"Creator '{creator_id}' not found in raw data")

    videos = data[creator_id]
    total = len(videos)

    format_counts = Counter(v["format"] for v in videos)
    voice_pct = sum(v["has_voice"] for v in videos) / total
    text_pct = sum(v["has_text"] for v in videos) / total
    avg_duration = mean(v["duration_sec"] for v in videos)

    dominant_formats = [
        f for f, c in format_counts.items() if c / total > 0.6
    ]

    underused_formats = [
        f for f, c in format_counts.items() if c / total < 0.2
    ]

    profile = {
        "creator_id": creator_id,
        "generated_by": "ConstraintSpace Profiling v0.1",
        "status": "draft",
        "analysis_window": f"last_{total}_videos",

        "observed_patterns": {
            "dominant_formats": dominant_formats,
            "underused_formats": underused_formats,
            "avg_duration_sec": round(avg_duration, 1)
        },

        "modality_bias": {
            "voice": "high" if voice_pct > 0.75 else "medium",
            "text": "high" if text_pct > 0.75 else "medium"
        },

        "creative_interpretation": {
            "summary": (
                "Creator shows strong preference for consistent explanatory formats. "
                "Format repetition suggests opportunity for low-risk experimentation."
            )
        },

        "human_review_required": True
    }

    return profile


def write_profile(profile: dict, out_path: str):
    with open(out_path, "w") as f:
        yaml.dump(profile, f, sort_keys=False)
