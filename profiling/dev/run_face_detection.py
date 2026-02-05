import cv2
from pathlib import Path

from profiling.cv.frame_sampler import sample_frames
from profiling.cv.face_detector import detect_faces


RAW_VIDEO_DIR = Path("profiling/raw_videos")
DEBUG_DIR = Path("profiling/dev/debug_faces")
DEBUG_DIR.mkdir(exist_ok=True)


def main():
    videos = list(RAW_VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise RuntimeError("No videos found in profiling/raw_videos")

    video_path = videos[0]
    print(f"[CV TEST] Using video: {video_path.name}")

    frames = sample_frames(str(video_path), num_frames=5)

    for i, frame in enumerate(frames):
        faces = detect_faces(frame)

        for f in faces:
            x, y, w, h = f["x"], f["y"], f["w"], f["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out_path = DEBUG_DIR / f"frame_{i}.jpg"
        cv2.imwrite(str(out_path), frame)

        print(f"Frame {i}: detected {len(faces)} face(s) → {out_path}")


if __name__ == "__main__":
    main()
