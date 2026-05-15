from typing import Dict, List
from utils.trace import trace


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

    if not texts:
        return {
            "embedding": None,
            "segments": [],
            "num_segments": 0,
            "dim": 0,
        }

    embeddings = embedder.embed_texts(texts)

    segment_payloads = [
        {
            "text": text,
            "embedding": emb,
        }
        for text, emb in zip(texts, embeddings)
    ]

    creator_vec = embeddings.mean(axis=0)

    return {
        "embedding": creator_vec,
        "segments": segment_payloads,
        "num_segments": len(texts),
        "dim": creator_vec.shape[0],
    }


def _format_match_label(format_alignment):
    if format_alignment is None:
        return "unknown"
    if format_alignment >= 0.8:
        return "strong"
    if format_alignment >= 0.5:
        return "partial"
    return "weak"


def build_resonance_report(idea: str, resonance: dict) -> dict:
    return {
        "idea_text": idea,
        "resonance": {
            k: v for k, v in resonance.items()
            if k != "evidence"
        },
        "top_similar_moments": resonance.get("evidence", []),
        "interpretation": {
            "semantic_fit": (
                "high" if resonance["semantic_alignment"] > 0.55
                else "medium" if resonance["semantic_alignment"] > 0.35
                else "low"
            ),
            "format_match": (
                _format_match_label(resonance.get("format_alignment"))
            ),
        },
    }
