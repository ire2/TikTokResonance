# profiling/embeddings/embedding_store.py

from pathlib import Path
import numpy as np
import json
from typing import Dict
from utils.trace import trace


EMBEDDING_DIR = Path("data/embeddings_store")
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)


def _base_path(creator_id: str, model_name: str) -> Path:
    safe_model = model_name.replace("/", "_")
    return EMBEDDING_DIR / f"{creator_id}_{safe_model}"


@trace
def load_creator_embeddings(
    creator_id: str,
    model_name: str,
) -> Dict | None:
    """
    Load creator-level semantic memory.

    Loads:
    - mean creator embedding
    - per-segment embeddings + texts
    """

    base = _base_path(creator_id, model_name)

    mean_vec_path = base.with_suffix(".npy")
    meta_path = base.with_suffix(".json")
    seg_vec_path = base.with_suffix(".segments.npy")
    seg_txt_path = base.with_suffix(".segments.json")

    if not mean_vec_path.exists() or not meta_path.exists():
        return None

    # ---- load mean embedding ----
    creator_embedding = np.load(mean_vec_path)

    with open(meta_path) as f:
        meta = json.load(f)

    # ---- load segment embeddings (optional but expected) ----
    segments = []

    if seg_vec_path.exists() and seg_txt_path.exists():
        seg_embeddings = np.load(seg_vec_path)

        with open(seg_txt_path) as f:
            texts = json.load(f)

        segments = [
            {
                "text": t.get("text"),
                "video_id": t.get("video_id"),
                "embedding": emb,
            }
            for t, emb in zip(texts, seg_embeddings)
        ]

    return {
        "creator_embedding": creator_embedding,
        "segments": segments,
        "num_segments": meta.get("num_segments", len(segments)),
        "dim": creator_embedding.shape[0],
        "model": model_name,
    }


@trace
def save_creator_embedding(
    creator_id: str,
    model_name: str,
    payload: Dict,
):
    base = _base_path(creator_id, model_name)

    # ---- save creator-level embedding ----
    np.save(base.with_suffix(".npy"), payload["embedding"])

    # ---- save segment embeddings ----
    segments = payload.get("segments", [])
    if segments:
        np.save(
            base.with_suffix(".segments.npy"),
            np.stack([s["embedding"] for s in segments])
        )

        with open(base.with_suffix(".segments.json"), "w") as f:
            json.dump(
                [
                    {
                        "text": s.get("text"),
                        "video_id": s.get("video_id"),
                    }
                    for s in segments
                ],
                f,
                indent=2,
            )

    # ---- save metadata ----
    meta = {
        "num_segments": payload.get("num_segments", 0),
        "dim": payload.get("dim", 0),
        "has_segments": bool(segments),
    }

    with open(base.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)


@trace
def load_creator_segments(
    creator_id: str,
    model_name: str,
) -> list[Dict]:
    base = _base_path(creator_id, model_name)

    seg_vec_path = base.with_suffix(".segments.npy")
    seg_txt_path = base.with_suffix(".segments.json")

    if not seg_vec_path.exists() or not seg_txt_path.exists():
        return []

    embeddings = np.load(seg_vec_path)

    with open(seg_txt_path) as f:
        texts = json.load(f)

    return [
        {
            "text": t["text"],
            "embedding": emb,
        }
        for t, emb in zip(texts, embeddings)
    ]
