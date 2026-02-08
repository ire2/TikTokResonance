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
    idea_text_density: float | None = None,
    idea_format: str | None = None,
    per_video_visuals: Dict | None = None,
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
    creator_text_density = (
        visual.get("avg_text_density_ocr")
        or visual.get("avg_text_density_heuristic")
    )

    format_alignment = None
    dominant_formats = (
        creator_profile.get("observed_patterns", {}).get("dominant_formats")
        or []
    )
    underused_formats = (
        creator_profile.get("observed_patterns", {}).get("underused_formats")
        or []
    )
    if idea_format:
        if idea_format in dominant_formats:
            format_alignment = 1.0
        elif idea_format in underused_formats:
            format_alignment = 0.3
        else:
            format_alignment = 0.6

    motion_alignment = None
    if idea_motion_intensity is not None and creator_motion_intensity is not None:
        motion_alignment = 1.0 - abs(
            float(idea_motion_intensity) - float(creator_motion_intensity)
        )
        motion_alignment = round(max(0.0, motion_alignment), 3)

    text_density_alignment = None
    if idea_text_density is not None and creator_text_density is not None:
        text_density_alignment = 1.0 - abs(
            float(idea_text_density) - float(creator_text_density)
        )
        text_density_alignment = round(
            max(0.0, text_density_alignment), 3
        )

    # ---- per-video visual alignment from top matches ----
    if per_video_visuals and evidence:
        top_video_ids = list({
            e.get("video_id") for e in evidence if e.get("video_id")
        })
        if top_video_ids:
            mot_vals = []
            txt_vals = []
            for vid in top_video_ids:
                sig = per_video_visuals.get(vid) or {}
                if idea_motion_intensity is not None and sig.get("motion_intensity") is not None:
                    mot_vals.append(
                        1.0 - abs(float(idea_motion_intensity) - float(sig.get("motion_intensity")))
                    )
                if idea_text_density is not None:
                    v_txt = sig.get("text_density_ocr") or sig.get("text_density_heuristic")
                    if v_txt is not None:
                        txt_vals.append(
                            1.0 - abs(float(idea_text_density) - float(v_txt))
                        )
            if mot_vals:
                motion_alignment = round(max(0.0, sum(mot_vals) / len(mot_vals)), 3)
            if txt_vals:
                text_density_alignment = round(max(0.0, sum(txt_vals) / len(txt_vals)), 3)

    # ---- semantic gate (embedding similarity) ----
    semantic_gate = None
    creator_vec = (
        creator_embedding_payload.get("creator_embedding")
        or creator_embedding_payload.get("embedding")
    )
    if creator_vec is not None:
        sim = float(cos_sim(idea_embedding, creator_vec)[0][0])
        semantic_gate = round(max(0.0, min(1.0, sim)), 3)

    # ---- final score ----
    if motion_alignment is None and text_density_alignment is None and format_alignment is None:
        resonance_score = round(
            0.55 * semantic_alignment
            + 0.25 * dialogue_affinity
            + 0.20 * (1 - solo_rant_affinity),
            4,
        )
    else:
        fmt = format_alignment if format_alignment is not None else 0.5
        mot = motion_alignment if motion_alignment is not None else 0.5
        txt = text_density_alignment if text_density_alignment is not None else 0.5
        resonance_score = round(
            0.45 * semantic_alignment
            + 0.18 * dialogue_affinity
            + 0.12 * (1 - solo_rant_affinity)
            + 0.10 * fmt
            + 0.08 * txt
            + 0.07 * mot,
            4,
        )

    if semantic_gate is not None:
        gate = max(0.5, semantic_gate)
        resonance_score = round(resonance_score * gate, 4)

    # Hard cap for low semantic alignment
    if semantic_alignment < 0.35:
        resonance_score = min(resonance_score, 0.25)

    return {
        "semantic_alignment": semantic_alignment,
        "dialogue_affinity": round(dialogue_affinity, 3),
        "solo_rant_affinity": round(solo_rant_affinity, 3),
        "talking_head_affinity": round(talking_head_affinity, 3),
        "motion_alignment": motion_alignment,
        "text_density_alignment": text_density_alignment,
        "format_alignment": format_alignment,
        "semantic_gate": semantic_gate,
        "resonance_score": resonance_score,
        "evidence": evidence,
    }
