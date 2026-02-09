
from utils.trace import trace
from profiling.cv.simple_classifier import SimpleFormatClassifier

classifier = SimpleFormatClassifier()  # global instance for simplicity

# DEFUNCT
# def infer_format(v):
#     """
#     Heuristic format inference (v0).


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
        normalized.append(normalize_video(v, creator_id=creator_id, idx=idx, total=total))

    return normalized


def normalize_video(raw_video, creator_id: str | None = None, idx: int | None = None, total: int | None = None):
    vid = raw_video.get("id")
    if vid:
        prefix = f"[NORMALIZE][{creator_id}]" if creator_id else "[NORMALIZE]"
        if idx is not None and total is not None:
            print(f"{prefix} {idx}/{total} video_id={vid}")
        else:
            print(f"{prefix} video_id={vid}")
    format_pred = classifier.classify(raw_video)
    return {
        "video_id": raw_video.get("id"),
        "local_path": raw_video.get("local_path"),
        "duration_sec": raw_video.get("duration"),
        "likes": raw_video.get("like_count"),
        "views": raw_video.get("view_count"),
        "comments": raw_video.get("comment_count"),
        "has_voice": raw_video.get("acodec") not in (None, "none"),
        "has_text": bool(raw_video.get("description")),
        "format": None,
        "format_pred": format_pred,
        "description": raw_video.get("description"),
        "posted_at": raw_video.get("upload_date"),
    }
