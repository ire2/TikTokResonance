import json
import yaml
from collections import Counter
from statistics import mean
from utils.trace import trace

from profiling.cv.visual_signals import extract_visual_signals
from profiling.utils.video_paths import video_path


@trace
def generate_profile(creator_id: str, raw_data_path: str) -> dict:
    """
    Generate a profiling report for a creator based on:
    - normalized metadata
    - CV-extracted visual signals
    """

    with open(raw_data_path, "r") as f:
        data = json.load(f)

    if creator_id not in data:
        raise ValueError(f"Creator '{creator_id}' not found in raw data")

    videos = data[creator_id]
    total = len(videos)

    if total == 0:
        raise ValueError("No videos available for profiling")

    # -----------------------------
    # CV: extract per-video signals
    # -----------------------------
    visual_signals_per_video = []

    for v in videos:
        video_id = v.get("video_id")
        if not video_id:
            continue

        path = video_path(creator_id, video_id)
        if not path.exists():
            print(f"[CV WARN] Missing video file: {path}")
            continue

        try:
            signals = extract_visual_signals(str(path))
            visual_signals_per_video.append(signals)
        except Exception as e:
            print(f"[CV WARN] CV failed on {path}: {e}")

    def avg(key):
        vals = [v[key] for v in visual_signals_per_video if key in v]
        return round(sum(vals) / len(vals), 3) if vals else None

    visual_summary = {
        "avg_face_presence_ratio": avg("face_presence_ratio"),
        "avg_talking_head_confidence": avg("talking_head_confidence"),
        "avg_motion_intensity": avg("motion_intensity"),
        "avg_face_area_stability": avg("face_area_stability"),
    }
    # -----------------------------
    # CV-based gating logic for Caption generaton
    # -----------------------------

    def compute_nlp_gate(visual_summary):
        r = visual_summary.get("avg_face_presence_ratio")
        if r is None:
            return {
                "weight": 0.5,
                "confidence": "low",
                "reason": "insufficient CV data",
            }

        if r < 0.2:
            return {
                "weight": 0.0,
                "confidence": "high",
                "reason": "faces rarely present",
            }

        if r < 0.6:
            return {
                "weight": 0.5,
                "confidence": "medium",
                "reason": "mixed visual structure",
            }

        return {
            "weight": 1.0,
            "confidence": "high",
            "reason": "consistent talking-head presence",
        }

    # -----------------------------
    # Metadata patterns (fallback)
    # -----------------------------
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

    # -----------------------------
    # CV-backed override (KEY PART)
    # -----------------------------
    th_conf = visual_summary.get("avg_talking_head_confidence")
    motion = visual_summary.get("avg_motion_intensity")

    if th_conf is not None and th_conf > 0.6:
        dominant_formats = ["talking_head"]
    elif motion is not None and motion > 0.4:
        dominant_formats = ["broll"]

    nlp_gate = compute_nlp_gate(visual_summary)
    # -----------------------------
    # Final profile
    # -----------------------------
    profile = {
        "creator_id": creator_id,
        "generated_by": "ConstraintSpace Profiling v0.1",
        "status": "draft",
        "analysis_window": f"last_{total}_videos",

        "observed_patterns": {
            "dominant_formats": dominant_formats,
            "underused_formats": underused_formats,
            "avg_duration_sec": round(avg_duration, 1),
        },

        "modality_bias": {
            "voice": "high" if voice_pct > 0.75 else "medium",
            "text": "high" if text_pct > 0.75 else "medium",
        },

        "visual_signals": visual_summary,
        "nlp_captioning_gate": nlp_gate,

        "creative_interpretation": {
            "summary": (
                "Creator behavior inferred using both metadata and "
                "computer vision signals. Visual consistency suggests "
                "dominant content structure with room for controlled experimentation."
            )
        },

        "human_review_required": True,
    }

    return profile


def write_profile(profile: dict, out_path: str):
    with open(out_path, "w") as f:
        yaml.dump(profile, f, sort_keys=False)
