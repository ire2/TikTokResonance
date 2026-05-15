from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


ALLOWED_DECISIONS = {"approve", "revise", "reject"}
DEFAULT_DECISIONS_PATH = Path("data/reviews/resonance_decisions.jsonl")
MAX_NOTES_CHARS = 1000
DEFAULT_REVIEWER_ID = "creator_strategist"


class ReviewDecisionError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def idea_fingerprint(payload: dict[str, Any]) -> str:
    source = {
        "creator_id": payload.get("creator_id"),
        "video_path": payload.get("video_path"),
        "idea_text": payload.get("idea_text"),
    }
    encoded = json.dumps(source, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_review_decision(
    payload: dict[str, Any],
    decision: str,
    notes: str = "",
    *,
    source: str = "live",
    reviewer_id: str = DEFAULT_REVIEWER_ID,
    created_at: str | None = None,
) -> dict[str, Any]:
    decision = (decision or "").strip().lower()
    if decision not in ALLOWED_DECISIONS:
        raise ReviewDecisionError(
            f"decision must be one of: {', '.join(sorted(ALLOWED_DECISIONS))}"
        )

    notes = (notes or "").strip()
    if len(notes) > MAX_NOTES_CHARS:
        raise ReviewDecisionError(
            f"notes must be {MAX_NOTES_CHARS} characters or fewer"
        )
    reviewer_id = (reviewer_id or DEFAULT_REVIEWER_ID).strip()

    resonance = payload.get("resonance") or {}
    evidence = payload.get("evidence") or []
    idea_text = payload.get("idea_text") or ""

    return {
        "review_id": uuid4().hex,
        "created_at": created_at or _utc_now(),
        "decision": decision,
        "notes": notes,
        "source": source,
        "reviewer_id": reviewer_id,
        "creator_id": payload.get("creator_id"),
        "video_path": payload.get("video_path"),
        "idea_fingerprint": idea_fingerprint(payload),
        "idea_text": idea_text,
        "idea_snippet": idea_text[:140],
        "model_name": payload.get("model_name"),
        "analysis_mode": payload.get("analysis_mode"),
        "resonance_score": resonance.get("resonance_score"),
        "semantic_alignment": resonance.get("semantic_alignment"),
        "format_alignment": resonance.get("format_alignment"),
        "motion_alignment": resonance.get("motion_alignment"),
        "text_density_alignment": resonance.get("text_density_alignment"),
        "evidence_video_ids": [
            e.get("video_id") for e in evidence[:3] if e.get("video_id")
        ],
    }


def save_review_decision(
    payload: dict[str, Any],
    decision: str,
    notes: str = "",
    *,
    path: Path = DEFAULT_DECISIONS_PATH,
    source: str = "live",
    reviewer_id: str = DEFAULT_REVIEWER_ID,
) -> dict[str, Any]:
    record = build_review_decision(
        payload=payload,
        decision=decision,
        notes=notes,
        source=source,
        reviewer_id=reviewer_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def load_review_decisions(
    path: Path = DEFAULT_DECISIONS_PATH,
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records[-limit:][::-1]


def reset_review_decisions(
    path: Path = DEFAULT_DECISIONS_PATH,
    *,
    seed_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = seed_records or []
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    return {
        "path": str(path),
        "records_written": len(records),
        "touched": [str(path)],
    }
