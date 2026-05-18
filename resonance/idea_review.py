from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from resonance.creator_library import (
    DEFAULT_DATA_DIR,
    build_creator_library,
    get_creator,
    load_label_index,
)


STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "get",
    "go",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "just",
    "like",
    "me",
    "more",
    "not",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "this",
    "to",
    "up",
    "use",
    "was",
    "we",
    "what",
    "when",
    "where",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


class IdeaReviewError(ValueError):
    pass


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", (text or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _text_similarity(idea_tokens: set[str], segment_text: str) -> float:
    segment_tokens = _tokens(segment_text)
    if not idea_tokens or not segment_tokens:
        return 0.0

    overlap = len(idea_tokens & segment_tokens)
    if overlap == 0:
        return 0.0

    idea_coverage = overlap / len(idea_tokens)
    segment_density = overlap / len(segment_tokens)
    return round((0.72 * idea_coverage) + (0.28 * segment_density), 4)


def _load_creator_segments(
    creator_id: str,
    data_dir: Path,
) -> list[dict[str, Any]]:
    embedding_dir = data_dir / "embeddings_store"
    if not embedding_dir.exists():
        return []

    paths = sorted(embedding_dir.glob(f"{creator_id}_*.segments.json"))
    if not paths:
        return []

    try:
        payload = json.loads(paths[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    if not isinstance(payload, list):
        return []
    return [
        item for item in payload
        if isinstance(item, dict) and item.get("text")
    ]


def _safe_demo_cache(cache_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _fit_label(score: float) -> str:
    if score >= 0.45:
        return "high"
    if score >= 0.22:
        return "medium"
    return "low"


def _score_weight_for_confidence(confidence_level: str) -> float:
    return {
        "high": 1.0,
        "medium": 0.86,
        "low": 0.68,
    }.get(confidence_level, 0.68)


def _calibrated_fit_score(raw_score: float, confidence_level: str) -> float:
    raw_score = max(0.0, min(1.0, float(raw_score or 0.0)))
    if raw_score >= 0.35:
        calibrated = 0.75 + min((raw_score - 0.35) / 0.25, 1.0) * 0.15
    elif raw_score >= 0.20:
        calibrated = 0.58 + ((raw_score - 0.20) / 0.15) * 0.17
    elif raw_score >= 0.08:
        calibrated = 0.34 + ((raw_score - 0.08) / 0.12) * 0.24
    else:
        calibrated = raw_score * 4.25

    confidence_weight = _score_weight_for_confidence(confidence_level)
    return round(calibrated * confidence_weight, 4)


def _labeled_evidence(
    *,
    creator_id: str,
    evidence: list[dict[str, Any]],
    data_dir: Path,
) -> list[dict[str, Any]]:
    label_index = load_label_index(data_dir).get(creator_id, {})
    enriched = []
    for item in evidence:
        video_id = str(item.get("video_id") or "")
        label = label_index.get(video_id, {})
        tiktok_url = label.get("tiktok_url") or (
            f"https://www.tiktok.com/@{creator_id}/video/{video_id}"
            if video_id
            else ""
        )
        enriched.append({
            **item,
            "format_label": label.get("format_label") or None,
            "performance_label": label.get("performance_label") or None,
            "tiktok_url": tiktok_url,
        })
    return enriched


def _suggestions(
    *,
    creator: dict[str, Any],
    score: float,
    miss_evidence: list[dict[str, Any]],
) -> list[dict[str, str]]:
    suggestions: list[dict[str, str]] = []

    if score < 0.22:
        suggestions.append({
            "type": "fit",
            "title": "Revise the angle",
            "detail": "The idea has limited overlap with cached creator segments, so treat the score as weak evidence.",
        })

    dominant_formats = creator.get("dominant_formats") or []
    if dominant_formats:
        suggestions.append({
            "type": "format",
            "title": "Check the format",
            "detail": f"Compare the idea against common formats: {', '.join(dominant_formats[:3])}.",
        })

    if miss_evidence:
        suggestions.append({
            "type": "risk",
            "title": "Inspect miss-like examples",
            "detail": "Some retrieved evidence is labeled miss; use the examples as a revision warning, not a prediction.",
        })

    if creator.get("confidence_level") == "low":
        suggestions.append({
            "type": "coverage",
            "title": "Use manual review",
            "detail": "Coverage is thin, so the saved decision should lean more on strategist judgment.",
        })

    return suggestions


def analyze_pasted_idea(
    *,
    creator_id: str,
    idea_text: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    creator_id = (creator_id or "").strip()
    idea_text = (idea_text or "").strip()
    if not idea_text:
        raise IdeaReviewError("idea_text is required")

    creator = get_creator(creator_id, data_dir)
    if creator is None:
        raise IdeaReviewError(f"unknown creator_id: {creator_id}")

    idea_tokens = _tokens(idea_text)
    segments = _load_creator_segments(creator_id, data_dir)
    scored = []
    for segment in segments:
        score = _text_similarity(idea_tokens, segment.get("text", ""))
        if score <= 0:
            continue
        scored.append({
            "video_id": segment.get("video_id"),
            "text": segment.get("text", ""),
            "similarity": score,
        })

    scored.sort(key=lambda item: item["similarity"], reverse=True)
    evidence = _labeled_evidence(
        creator_id=creator_id,
        evidence=scored[:5],
        data_dir=data_dir,
    )

    if not evidence and cache_path:
        cached = _safe_demo_cache(Path(cache_path))
        if cached:
            fallback = dict(cached)
            fallback["creator_id"] = creator_id
            fallback["idea_text"] = idea_text
            fallback["analysis_mode"] = "demo_cache_fallback"
            fallback["analysis_note"] = (
                "No local segment match was available, so this uses the cached demo payload as a clearly labeled fallback."
            )
            fallback["coverage"] = creator
            fallback["hit_evidence"] = []
            fallback["miss_evidence"] = []
            return fallback

    top_scores = [item["similarity"] for item in evidence[:3]]
    semantic_alignment = (
        round(sum(top_scores) / len(top_scores), 4) if top_scores else 0.0
    )
    raw_match_score = round(
        semantic_alignment * _score_weight_for_confidence(creator["confidence_level"]),
        4,
    )
    resonance_score = _calibrated_fit_score(
        raw_match_score,
        creator["confidence_level"],
    )

    hit_evidence = [
        item for item in evidence
        if (item.get("performance_label") or "").lower() == "hit"
    ][:3]
    miss_evidence = [
        item for item in evidence
        if (item.get("performance_label") or "").lower() == "miss"
    ][:3]
    label_counts = Counter(
        (item.get("performance_label") or "unlabeled").lower()
        for item in evidence
    )

    resonance = {
        "semantic_alignment": semantic_alignment,
        "raw_match_score": raw_match_score,
        "format_alignment": None,
        "motion_alignment": None,
        "text_density_alignment": None,
        "resonance_score": resonance_score,
    }

    return {
        "creator_id": creator_id,
        "model_name": "local_segment_overlap",
        "video_path": None,
        "idea_text": idea_text,
        "analysis_mode": "local_segment_overlap",
        "analysis_note": (
            "Uses cached creator segment text, human labels, and local artifacts only. This is fit evidence, not a virality prediction."
        ),
        "coverage": creator,
        "resonance": resonance,
        "interpretation": {
            "semantic_fit": _fit_label(semantic_alignment),
            "format_match": "not evaluated for pasted text",
            "confidence": creator["confidence_level"],
        },
        "evidence": evidence,
        "hit_evidence": hit_evidence,
        "miss_evidence": miss_evidence,
        "label_summary": {
            "hit": label_counts.get("hit", 0),
            "ok": label_counts.get("ok", 0),
            "miss": label_counts.get("miss", 0),
            "unlabeled": label_counts.get("unlabeled", 0),
        },
        "suggestions": _suggestions(
            creator=creator,
            score=semantic_alignment,
            miss_evidence=miss_evidence,
        ),
    }


def default_creator_id(data_dir: Path = DEFAULT_DATA_DIR) -> str | None:
    creators = build_creator_library(data_dir)
    return creators[0]["creator_id"] if creators else None
