from collections import Counter
from statistics import mean
from utils.trace import trace


@trace
def compute_metadata_profile(videos: list) -> dict:
    total = len(videos)

    formats = []
    for v in videos:
        fmt = v.get("format") or v.get("format_pred")
        if fmt:
            formats.append(fmt)
    format_counts = Counter(formats)
    dominant_formats = [f for f, c in format_counts.items() if c / total > 0.6]
    underused_formats = [
        f for f, c in format_counts.items() if c / total < 0.2]

    return {
        "dominant_formats": dominant_formats,
        "underused_formats": underused_formats,
        "avg_duration_sec": round(mean(v["duration_sec"] for v in videos), 1),
        "voice_pct": sum(v["has_voice"] for v in videos) / total,
        "text_pct": sum(v["has_text"] for v in videos) / total,
    }
