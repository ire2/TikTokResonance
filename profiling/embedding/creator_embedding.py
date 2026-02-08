# profiling/embeddings/creator_embedding.py

import numpy as np
from typing import Dict, List
from utils.trace import trace


@trace
def aggregate_embeddings(
    embeddings: np.ndarray,
    strategy: str = "mean",
) -> np.ndarray:
    """
    Aggregate segment embeddings into a creator embedding.

    Strategies:
    - mean: baseline
    - weighted_mean: placeholder for future attention
    """

    if embeddings.size == 0:
        return embeddings

    if strategy == "mean":
        return embeddings.mean(axis=0)

    raise ValueError(f"Unknown aggregation strategy: {strategy}")


@trace
def build_creator_embedding(
    transcript_segments: List[Dict],
    embedder,
) -> Dict:
    texts = [
        s["text"]
        for s in transcript_segments
        if s.get("text") and len(s["text"].strip()) > 3
    ]

    embeddings = embedder.embed_texts(texts)

    segments = []
    for s, text, emb in zip(transcript_segments, texts, embeddings):
        segments.append({
            "text": text,
            "embedding": emb,
            "video_id": s.get("video_id"),
        })

    creator_vec = embeddings.mean(axis=0)

    return {
        "embedding": creator_vec,
        "segments": segments,          # ← THIS is the missing piece
        "num_segments": len(segments),
        "dim": creator_vec.shape[0],
    }
