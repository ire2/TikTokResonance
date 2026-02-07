from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from utils.trace import trace
from profiling.cv.visual_signals import extract_visual_signals


class LearnedFormatClassifier:
    """
    RandomForest-based format classifier using visual signals.
    Falls back to None if no model is available.
    """

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or Path(
            "profiling/models/format_rf.joblib"
        )
        self.bundle = None
        if self.model_path.exists():
            try:
                from joblib import load
                self.bundle = load(self.model_path)
            except Exception:
                self.bundle = None

    def is_ready(self) -> bool:
        return bool(self.bundle)

    def _feature_vector(self, signals: Dict[str, Any]) -> List[float]:
        features = self.bundle["features"]
        vec = []
        for k in features:
            v = signals.get(k, 0.0)
            try:
                vec.append(float(v))
            except Exception:
                vec.append(0.0)
        return vec

    @trace
    def classify(self, video: Dict[str, Any]) -> str | None:
        if not self.bundle:
            return None

        local_path = video.get("local_path")
        if not local_path or not Path(local_path).exists():
            return None

        try:
            signals = extract_visual_signals(
                str(local_path),
                use_ocr=False,
                use_objects=False,
                use_audio=False,
            )
        except Exception:
            return None

        vec = np.array([self._feature_vector(signals)])
        model = self.bundle["model"]
        label_encoder = self.bundle["label_encoder"]
        pred = model.predict(vec)[0]
        return str(label_encoder.inverse_transform([pred])[0])
