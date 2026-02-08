import numpy as np
from typing import List, Dict
from utils.trace import trace


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@trace
def compute_segment_similarity(
    idea_embedding: np.ndarray,
    segment_embeddings: List[Dict],
    top_k: int = 10,
) -> Dict:
    """
    Compare an idea embedding against ALL creator transcript segments.

    Returns:
    - top_k similarity score (mean)
    - max similarity
    - top matching segments (for explainability)
    """

    if not segment_embeddings:
        return {
            "top_k_mean": 0.0,
            "max_similarity": 0.0,
            "matches": [],
        }

    scored = []

    for seg in segment_embeddings:
        sim = cosine_similarity(idea_embedding, seg["embedding"])
        scored.append({
            "similarity": sim,
            "text": seg.get("text", ""),
        })

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top = scored[:top_k]

    return {
        "top_k_mean": round(
            sum(s["similarity"] for s in top) / len(top), 4
        ),
        "max_similarity": round(top[0]["similarity"], 4),
        "matches": top[:5],  # keep report readable
    }
