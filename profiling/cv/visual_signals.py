from typing import Dict
import numpy as np

from profiling.cv.frame_sampler import sample_frames
from profiling.cv.face_detector import detect_faces
from utils.trace import trace


@trace
def extract_visual_signals(
    video_path: str,
    num_frames: int = 8,
) -> Dict[str, float]:
    """
    Extract robust visual signals from a video.

    This function intentionally aggregates across frames to
    reduce false positives (e.g. tattoos detected as faces).

    Returns interpretable, downstream-safe signals.
    """

    frames = sample_frames(video_path, num_frames=num_frames)

    face_counts = []
    face_areas = []

    for frame in frames:
        faces = detect_faces(frame)
        face_counts.append(len(faces))

        for f in faces:
            face_area = f["w"] * f["h"]
            face_areas.append(face_area)

    total_frames = len(frames)

    # -----------------------------
    # Core signals
    # -----------------------------
    frames_with_faces = sum(1 for c in face_counts if c > 0)

    face_presence_ratio = (
        frames_with_faces / total_frames if total_frames > 0 else 0.0
    )

    avg_faces_per_frame = (
        sum(face_counts) / total_frames if total_frames > 0 else 0.0
    )

    max_faces = max(face_counts) if face_counts else 0

    # -----------------------------
    # Robust talking-head confidence
    # -----------------------------
    talking_head_confidence = min(
        1.0,
        0.7 * face_presence_ratio
        + 0.3 * min(1.0, avg_faces_per_frame),
    )

    # -----------------------------
    # Optional stability heuristic
    # -----------------------------
    face_area_stability = 0.0
    if len(face_areas) > 1:
        face_area_stability = 1.0 - min(
            1.0,
            np.std(face_areas) / (np.mean(face_areas) + 1e-6)
        )

    return {
        "face_presence_ratio": float(round(face_presence_ratio, 3)),
        "avg_faces_per_frame": float(round(avg_faces_per_frame, 3)),
        "max_faces": int(max_faces),
        "talking_head_confidence": float(round(talking_head_confidence, 3)),
        "face_area_stability": float(round(face_area_stability, 3)),
    }
