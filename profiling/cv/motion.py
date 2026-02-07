import cv2
import numpy as np

from utils.trace import trace


@trace
def compute_motion_intensity(
    video_path: str,
    sample_rate: int = 5,
    resize_width: int = 160,
) -> float:
    """
    Compute normalized motion intensity based on
    frame-to-frame pixel differences.

    Returns value in [0, 1].
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    prev_gray = None
    diffs = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # sample every N frames
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        # resize for speed + stability
        h, w = frame.shape[:2]
        scale = resize_width / w
        frame = cv2.resize(frame, (resize_width, int(h * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            diffs.append(np.mean(diff))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not diffs:
        return 0.0

    # Normalize: max possible pixel diff is 255
    motion = float(np.mean(diffs) / 255.0)

    return round(min(1.0, motion), 4)
