import numpy as np
from typing import Dict, List
from utils.trace import trace
from sentence_transformers.util import cos_sim


@trace
def compute_resonance(
    idea_embedding: np.ndarray,
    creator_embedding_payload: Dict,
    creator_profile: Dict,
    top_k: int = 5,
    idea_motion_intensity: float | None = None,
) -> Dict:
    """
    Compute resonance between an idea and a creator.

    Key change:
    - Semantic alignment is based on TOP-K matching past segments,
      not the creator mean embedding.
    """

    segment_payloads = creator_embedding_payload.get("segments", [])
    if not segment_payloads:
        semantic_alignment = 0.0
        evidence = []
    else:
        # ---- group segments by video ----
        videos: Dict[str, List[Dict]] = {}
        for seg in segment_payloads:
            vid = seg.get("video_id", "unknown")
            videos.setdefault(vid, []).append(seg)

        per_video_scores = []
        per_video_evidence = []

        for vid, segs in videos.items():
            sims = []
            for seg in segs:
                sim = float(
                    cos_sim(idea_embedding, seg["embedding"])[0][0]
                )
                sims.append({
                    "video_id": vid,
                    "text": seg["text"],
                    "similarity": round(sim, 4),
                })

            sims.sort(key=lambda x: x["similarity"], reverse=True)

            top = sims[:top_k]
            if not top:
                continue

            video_score = sum(s["similarity"] for s in top) / len(top)
            per_video_scores.append(video_score)
            per_video_evidence.extend(top[:1])  # 1 best moment per video

        semantic_alignment = round(
            sum(per_video_scores) / len(per_video_scores),
            4,
        ) if per_video_scores else 0.0

        evidence = per_video_evidence

    # ---- structural affinities ----
    nlp = creator_profile.get("profile_nlp", {})

    dialogue_affinity = nlp.get("dialogue_video_ratio", 0.0)
    solo_rant_affinity = nlp.get("solo_rant_ratio", 0.0)

    visual = creator_profile.get("visual_signals", {})
    talking_head_affinity = (
        visual.get("avg_talking_head_confidence") or 0.0
    )
    creator_motion_intensity = visual.get("avg_motion_intensity")

    motion_alignment = None
    if idea_motion_intensity is not None and creator_motion_intensity is not None:
        motion_alignment = 1.0 - abs(
            float(idea_motion_intensity) - float(creator_motion_intensity)
        )
        motion_alignment = round(max(0.0, motion_alignment), 3)

    # ---- final score ----
    if motion_alignment is None:
        resonance_score = round(
            0.55 * semantic_alignment
            + 0.25 * dialogue_affinity
            + 0.20 * (1 - solo_rant_affinity),
            4,
        )
    else:
        resonance_score = round(
            0.50 * semantic_alignment
            + 0.20 * dialogue_affinity
            + 0.15 * (1 - solo_rant_affinity)
            + 0.15 * motion_alignment,
            4,
        )

    return {
        "semantic_alignment": semantic_alignment,
        "dialogue_affinity": round(dialogue_affinity, 3),
        "solo_rant_affinity": round(solo_rant_affinity, 3),
        "talking_head_affinity": round(talking_head_affinity, 3),
        "motion_alignment": motion_alignment,
        "resonance_score": resonance_score,
        "evidence": evidence,
    }
