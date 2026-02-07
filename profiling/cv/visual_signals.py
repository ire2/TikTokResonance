from typing import Dict
import numpy as np
import cv2

from profiling.cv.frame_sampler import sample_frames
from profiling.cv.face_detector import detect_faces
from utils.trace import trace
from profiling.cv.motion import compute_motion_intensity
from profiling.cv.visual_metrics import (
    compute_shot_change_rate,
    compute_camera_motion_intensity,
)
from profiling.cv.ocr_text import estimate_text_density
from profiling.cv.object_detection import detect_objects
from profiling.cv.audio_metrics import compute_audio_metrics


@trace
def extract_visual_signals(
    video_path: str,
    num_frames: int = 8,
    use_ocr: bool = True,
    use_objects: bool = True,
    use_audio: bool = True,
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
    face_area_ratios = []

    for frame in frames:
        faces = detect_faces(frame)
        face_counts.append(len(faces))

        h, w = frame.shape[:2]
        frame_area = float(h * w)
        for f in faces:
            face_area = f["w"] * f["h"]
            face_areas.append(face_area)
            face_area_ratios.append(face_area / frame_area)

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

    avg_face_area_ratio = (
        float(np.mean(face_area_ratios)) if face_area_ratios else 0.0
    )
    face_area_ratio_std = (
        float(np.std(face_area_ratios)) if len(face_area_ratios) > 1 else 0.0
    )

    # -----------------------------
    # Visual environment signals
    # -----------------------------
    brightness_vals = []
    saturation_vals = []
    edge_densities = []
    text_densities = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_vals.append(float(np.mean(gray) / 255.0))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_vals.append(float(np.mean(hsv[..., 1]) / 255.0))

        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.sum(edges > 0) / edges.size)
        edge_densities.append(edge_density)

        # text-like density (heuristic)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thr = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )
        contours, _ = cv2.findContours(
            thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frame_area = float(gray.shape[0] * gray.shape[1])
        area_sum = 0.0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 30 or area > frame_area * 0.2:
                continue
            ar = w / (h + 1e-6)
            if ar < 0.1 or ar > 10:
                continue
            area_sum += area
        text_densities.append(area_sum / frame_area)

    brightness_mean = float(np.mean(brightness_vals)
                            ) if brightness_vals else 0.0
    brightness_std = float(np.std(brightness_vals)) if len(
        brightness_vals) > 1 else 0.0
    saturation_mean = float(np.mean(saturation_vals)
                            ) if saturation_vals else 0.0
    edge_density_mean = float(np.mean(edge_densities)
                              ) if edge_densities else 0.0
    text_density_mean = float(np.mean(text_densities)
                              ) if text_densities else 0.0

    # Color temperature index: negative = warm, positive = cool
    color_temp_vals = []
    for frame in frames:
        b = float(np.mean(frame[..., 0]))
        r = float(np.mean(frame[..., 2]))
        color_temp_vals.append((b - r) / (b + r + 1e-6))
    color_temperature_index = (
        float(np.mean(color_temp_vals)) if color_temp_vals else 0.0
    )

    # -----------------------------
    # Video-level signals
    # -----------------------------
    shot_change_rate = compute_shot_change_rate(video_path)
    camera_motion_intensity = compute_camera_motion_intensity(video_path)
    camera_stability = round(1.0 - camera_motion_intensity, 3)

    # OCR-based text density
    if use_ocr:
        try:
            text_density_ocr = estimate_text_density(frames)
        except Exception:
            text_density_ocr = 0.0
    else:
        text_density_ocr = 0.0

    # Object detection
    if use_objects:
        try:
            object_stats = detect_objects(frames)
        except Exception:
            object_stats = {
                "object_presence_ratio": 0.0,
                "avg_objects_per_frame": 0.0,
                "top_objects": [],
            }
    else:
        object_stats = {
            "object_presence_ratio": 0.0,
            "avg_objects_per_frame": 0.0,
            "top_objects": [],
        }

    # Audio metrics
    audio_stats = compute_audio_metrics(video_path) if use_audio else {
        "audio_energy": 0.0,
        "music_confidence": 0.0,
    }

    return {
        "face_presence_ratio": float(round(face_presence_ratio, 3)),
        "avg_faces_per_frame": float(round(avg_faces_per_frame, 3)),
        "max_faces": int(max_faces),
        "talking_head_confidence": float(round(talking_head_confidence, 3)),
        "face_area_stability": float(round(face_area_stability, 3)),
        "avg_face_area_ratio": float(round(avg_face_area_ratio, 4)),
        "face_area_ratio_std": float(round(face_area_ratio_std, 4)),
        "brightness_mean": float(round(brightness_mean, 3)),
        "brightness_std": float(round(brightness_std, 3)),
        "saturation_mean": float(round(saturation_mean, 3)),
        "color_temperature_index": float(round(color_temperature_index, 3)),
        "edge_density": float(round(edge_density_mean, 4)),
        "text_density_heuristic": float(round(text_density_mean, 4)),
        "text_density_ocr": float(round(text_density_ocr, 4)),
        "shot_change_rate": float(round(shot_change_rate, 3)),
        "camera_motion_intensity": float(round(camera_motion_intensity, 3)),
        "camera_stability": float(round(camera_stability, 3)),
        "motion_intensity": float(
            round(compute_motion_intensity(video_path), 3)
        ),
        "object_presence_ratio": object_stats["object_presence_ratio"],
        "avg_objects_per_frame": object_stats["avg_objects_per_frame"],
        "top_objects": object_stats["top_objects"],
        "audio_energy": audio_stats["audio_energy"],
        "music_confidence": audio_stats["music_confidence"],
    }
