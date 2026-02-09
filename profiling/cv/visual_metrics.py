import cv2
import numpy as np
from utils.trace import trace


@trace
def _iter_sampled_frames(
    video_path: str,
    sample_rate: int = 5,
    resize_width: int = 160,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        scale = resize_width / w
        frame = cv2.resize(frame, (resize_width, int(h * scale)))

        yield frame
        frame_idx += 1

    cap.release()


@trace
def compute_shot_change_rate(
    video_path: str,
    sample_rate: int = 5,
    resize_width: int = 160,
    diff_threshold: float = 0.35,
) -> float:
    """
    Estimate shot change rate using frame-to-frame differences.
    Returns value in [0, 1] over sampled frames.
    """

    prev_gray = None
    changes = 0
    total = 0

    for frame in _iter_sampled_frames(
        video_path, sample_rate=sample_rate, resize_width=resize_width
    ):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            norm = float(np.mean(diff) / 255.0)
            if norm > diff_threshold:
                changes += 1
            total += 1

        prev_gray = gray

    if total == 0:
        return 0.0

    return round(min(1.0, changes / total), 3)


@trace
def compute_camera_motion_intensity(
    video_path: str,
    sample_rate: int = 5,
    resize_width: int = 160,
) -> float:
    """
    Estimate camera motion using optical flow magnitude.
    Returns value in [0, 1].
    """

    prev_gray = None
    mags = []

    for frame in _iter_sampled_frames(
        video_path, sample_rate=sample_rate, resize_width=resize_width
    ):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mags.append(float(np.mean(mag)))

        prev_gray = gray

    if not mags:
        return 0.0

    norm = min(1.0, float(np.mean(mags) / 10.0))
    return round(norm, 3)
