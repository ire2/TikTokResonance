from utils.trace import trace
import cv2
from pathlib import Path
from typing import List, Dict


CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces(frame) -> List[Dict]:
    """
    Detect faces in a single frame.

    Returns:
        List of bounding boxes:
        [{x, y, w, h}]
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    results = []
    for (x, y, w, h) in faces:
        results.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
        })

    return results
