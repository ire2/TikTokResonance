import cv2
import os


def sample_frames(video_path: str, num_frames: int = 5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []

    indices = [
        int(total_frames * i / (num_frames + 1))
        for i in range(1, num_frames + 1)
    ]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if success:
            frames.append(frame)

    cap.release()
    return frames
