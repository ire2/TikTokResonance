from functools import lru_cache
from typing import List

import numpy as np

from utils.trace import trace


@lru_cache(maxsize=1)
@trace
def _get_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


def estimate_text_density(frames: List[np.ndarray]) -> float:
    """
    Estimate on-screen text density using OCR bounding boxes.
    Returns mean text area ratio across frames in [0, 1].
    """

    if not frames:
        return 0.0

    reader = _get_reader()
    densities = []

    for frame in frames:
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        if frame_area == 0:
            densities.append(0.0)
            continue

        results = reader.readtext(frame)
        area_sum = 0.0
        for bbox, _text, conf in results:
            if conf is None or conf < 0.3:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x0, x1 = max(0, int(min(xs))), min(w, int(max(xs)))
            y0, y1 = max(0, int(min(ys))), min(h, int(max(ys)))
            area_sum += max(0, x1 - x0) * max(0, y1 - y0)

        densities.append(min(1.0, area_sum / frame_area))

    return float(round(sum(densities) / len(densities), 4))
