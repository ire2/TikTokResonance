from statistics import mean
from pathlib import Path
import json
import os
import multiprocessing as mp
from utils.trace import trace
from profiling.cv.visual_signals import extract_visual_signals
from profiling.utils.video_paths import video_path


def _extract_visual_for_video(args):
    creator_id, video_id, fast_visual = args
    path = video_path(creator_id, video_id)
    if not path.exists():
        return (video_id, None)
    try:
        sig = extract_visual_signals(
            str(path),
            use_ocr=not fast_visual,
            use_objects=not fast_visual,
            use_audio=not fast_visual,
        )
        return (video_id, sig)
    except Exception:
        return (video_id, None)


@trace
def compute_visual_profile(creator_id: str, videos: list) -> dict:
    base_dir = Path(__file__).resolve().parents[1]
    raw_visual_dir = Path("data/raw_visual")
    raw_visual_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_visual_dir / f"{creator_id}.json"

    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    else:
        cache = {}

    signals = []
    fast_visual = os.getenv("FAST_VISUAL", "false").lower() == "true"
    workers = int(os.getenv("VISUAL_WORKERS", "1"))

    fast_visual = os.getenv("FAST_VISUAL", "false").lower() == "true"
    if fast_visual:
        print(f"[VISUAL][{creator_id}] FAST_VISUAL=true (OCR/objects/audio disabled)")
    else:
        print(f"[VISUAL][{creator_id}] FAST_VISUAL=false (OCR/objects/audio enabled)")

    # Gather worklist of missing videos
    work = []
    for v in videos:
        video_id = v.get("video_id")
        if not video_id:
            continue
        if video_id in cache:
            signals.append(cache[video_id])
        else:
            work.append(video_id)

    if workers > 1 and work:
        print(f"[VISUAL][{creator_id}] Using {workers} workers")
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            args = [(creator_id, vid, fast_visual) for vid in work]
            for vid, sig in pool.imap_unordered(_extract_visual_for_video, args):
                if sig:
                    cache[vid] = sig
                    signals.append(sig)
    else:
        total = len(work)
        for idx, video_id in enumerate(work, start=1):
            print(f"[VISUAL][{creator_id}] {idx}/{total} video_id={video_id}")
            vid, sig = _extract_visual_for_video(
                (creator_id, video_id, fast_visual)
            )
            if sig:
                cache[vid] = sig
                signals.append(sig)

    # persist cache
    try:
        cache_path.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass

    def avg(key):
        vals = [s[key] for s in signals if key in s]
        return round(mean(vals), 3) if vals else None

    return {
        "avg_face_presence_ratio": avg("face_presence_ratio"),
        "avg_talking_head_confidence": avg("talking_head_confidence"),
        "avg_motion_intensity": avg("motion_intensity"),
        "avg_face_area_stability": avg("face_area_stability"),
        "avg_face_area_ratio": avg("avg_face_area_ratio"),
        "avg_face_area_ratio_std": avg("face_area_ratio_std"),
        "avg_brightness_mean": avg("brightness_mean"),
        "avg_brightness_std": avg("brightness_std"),
        "avg_saturation_mean": avg("saturation_mean"),
        "avg_color_temperature_index": avg("color_temperature_index"),
        "avg_edge_density": avg("edge_density"),
        "avg_text_density_heuristic": avg("text_density_heuristic"),
        "avg_text_density_ocr": avg("text_density_ocr"),
        "avg_shot_change_rate": avg("shot_change_rate"),
        "avg_camera_motion_intensity": avg("camera_motion_intensity"),
        "avg_camera_stability": avg("camera_stability"),
        "avg_object_presence_ratio": avg("object_presence_ratio"),
        "avg_objects_per_frame": avg("avg_objects_per_frame"),
        "avg_audio_energy": avg("audio_energy"),
        "avg_music_confidence": avg("music_confidence"),
    }
