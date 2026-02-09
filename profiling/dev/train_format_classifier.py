from pathlib import Path
import csv
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump


LABELS_PATH = Path("data/labels/format_labels.csv")
RAW_VISUAL_DIR = Path("data/raw_visual")
MODEL_PATH = Path("profiling/models/format_rf.joblib")


# Features used for training. Must exist in raw_visual cache.
FEATURES = [
    "face_presence_ratio",
    "avg_faces_per_frame",
    "max_faces",
    "talking_head_confidence",
    "face_area_stability",
    "avg_face_area_ratio",
    "face_area_ratio_std",
    "brightness_mean",
    "brightness_std",
    "saturation_mean",
    "color_temperature_index",
    "edge_density",
    "text_density_heuristic",
    "shot_change_rate",
    "camera_motion_intensity",
    "camera_stability",
    "motion_intensity",
]


def load_cache(creator_id: str) -> dict:
    path = RAW_VISUAL_DIR / f"{creator_id}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def main():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")

    rows = []
    with open(LABELS_PATH, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("creator_id") or not r.get("video_id"):
                continue
            rows.append(r)

    if not rows:
        raise ValueError("No labels found in format_labels.csv")

    X = []
    y = []

    cache_by_creator = {}
    missing = 0

    for r in rows:
        creator_id = r["creator_id"]
        video_id = r["video_id"]
        label = r.get("format_label") or r.get("label")
        if not label:
            continue

        if creator_id not in cache_by_creator:
            cache_by_creator[creator_id] = load_cache(creator_id)

        cache = cache_by_creator[creator_id]
        signals = cache.get(video_id)
        if not signals:
            missing += 1
            continue

        vec = []
        for k in FEATURES:
            v = signals.get(k, 0.0)
            try:
                vec.append(float(v))
            except Exception:
                vec.append(0.0)

        X.append(vec)
        y.append(label)

    if not X:
        raise ValueError(
            "No training samples found (missing raw_visual cache?)")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(np.array(X), y_enc)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "model": model,
            "label_encoder": le,
            "features": FEATURES,
        },
        MODEL_PATH,
    )

    print(f"[OK] trained RF model → {MODEL_PATH}")
    if missing:
        print(
            f"[WARN] missing {missing} labeled videos with no raw_visual cache")


if __name__ == "__main__":
    main()
