
from utils.trace import trace
from profiling.cv.simple_classifier import SimpleFormatClassifier

classifier = SimpleFormatClassifier()  # global instance for simplicity

# DEFUNCT
# def infer_format(v):
#     """
#     Heuristic format inference (v0).

#     TODO (v1):
#     Replace with CV-based classification:
#     - Shot detection + face presence (talking head vs b-roll)
#     - OCR density for text-heavy videos
#     - Audio energy + speaker diarization

#     Heuristics are intentionally used in v0 for speed,
#     explainability, and human override during review.
#     """

#     has_voice = v.get("acodec") not in (None, "none")
#     has_text = bool(v.get("description"))
#     duration = v.get("duration") or 0

#     if has_voice and duration >= 20:
#         return "talking_head"
#     if has_voice and duration < 20:
#         return "voiceover_short"
#     if not has_voice and has_text:
#         return "text_only"
#     return "other"


@trace
def normalize_videos(raw_videos, creator_id: str | None = None):
    """
    Convert raw yt-dlp video objects into our internal raw_data schema.

    Input:
        raw_videos: List[dict] (raw yt-dlp metadata)

    Output:
        List[dict] (normalized, stable schema)
    """

    normalized = []
    total = len(raw_videos)

    for idx, v in enumerate(raw_videos, start=1):
        vid = v.get("id")
        if vid:
            prefix = f"[NORMALIZE][{creator_id}]" if creator_id else "[NORMALIZE]"
            print(f"{prefix} {idx}/{total} video_id={vid}")
        normalized.append({
            "video_id": v.get("id"),
            "local_path": v.get("local_path"),
            "duration_sec": v.get("duration"),
            "likes": v.get("like_count"),
            "views": v.get("view_count"),
            "comments": v.get("comment_count"),
            "has_voice": v.get("acodec") not in (None, "none"),
            "has_text": bool(v.get("description")),
            "format": classifier.classify(v),
            'description': v.get("description"),
            "posted_at": v.get("upload_date"),
        })

    return normalized
