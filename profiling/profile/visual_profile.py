from statistics import mean
from utils.trace import trace
from profiling.cv.visual_signals import extract_visual_signals
from profiling.utils.video_paths import video_path


@trace
def compute_visual_profile(creator_id: str, videos: list) -> dict:
    signals = []

    for v in videos:
        video_id = v.get("video_id")
        if not video_id:
            continue

        path = video_path(creator_id, video_id)
        if not path.exists():
            continue

        try:
            signals.append(extract_visual_signals(str(path)))
        except Exception:
            continue

    def avg(key):
        vals = [s[key] for s in signals if key in s]
        return round(mean(vals), 3) if vals else None

    return {
        "avg_face_presence_ratio": avg("face_presence_ratio"),
        "avg_talking_head_confidence": avg("talking_head_confidence"),
        "avg_motion_intensity": avg("motion_intensity"),
        "avg_face_area_stability": avg("face_area_stability"),
    }
