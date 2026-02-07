from utils.trace import trace
from functools import lru_cache
from typing import List

import numpy as np


@lru_cache(maxsize=1)
def _get_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")


@trace
def detect_objects(frames: List[np.ndarray], conf: float = 0.25) -> dict:
    """
    Run YOLO object detection on frames and return aggregate stats.
    """

    if not frames:
        return {
            "object_presence_ratio": 0.0,
            "avg_objects_per_frame": 0.0,
            "top_objects": [],
        }

    model = _get_model()
    total_frames = len(frames)
    frames_with_objects = 0
    total_objects = 0
    class_counts = {}

    results = model.predict(frames, conf=conf, verbose=False)
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes
        if len(boxes) > 0:
            frames_with_objects += 1
        total_objects += len(boxes)
        for cls_id in boxes.cls.tolist():
            name = r.names.get(int(cls_id), str(int(cls_id)))
            class_counts[name] = class_counts.get(name, 0) + 1

    top_objects = sorted(
        class_counts.items(), key=lambda x: x[1], reverse=True
    )[:5]

    return {
        "object_presence_ratio": round(frames_with_objects / total_frames, 3),
        "avg_objects_per_frame": round(total_objects / total_frames, 3),
        "top_objects": [f"{k}:{v}" for k, v in top_objects],
    }
