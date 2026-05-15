from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from resonance.creator_library import (
    DEFAULT_DATA_DIR,
    PERFORMANCE_LABELS,
    load_label_rows,
)


DEFAULT_MARKET_INDEX_PATH = Path("data/artifacts/market_index.json")
VALID_PERFORMANCE_LABELS = set(PERFORMANCE_LABELS)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _creator_from_caption(path: Path) -> tuple[str, str] | None:
    stem = path.stem
    if stem.startswith("idea_") or "_" not in stem:
        return None
    creator_id, video_id = stem.rsplit("_", 1)
    if not creator_id or not video_id:
        return None
    return creator_id, video_id


def _creator_from_embedding(path: Path, known_creators: set[str]) -> str | None:
    for creator_id in sorted(known_creators, key=len, reverse=True):
        if path.name.startswith(f"{creator_id}_"):
            return creator_id
    if "_" not in path.name:
        return None
    return path.name.rsplit("_", 1)[0]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    number = _safe_float(value)
    return int(number) if number is not None else None


def _median(values: list[float | int]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return round(float(median(clean)), 4)


def _load_metadata(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    path = data_dir / "raw_data" / "creator_metadata.json"
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {}

    metadata: dict[str, list[dict[str, Any]]] = {}
    for creator_id, rows in payload.items():
        if not isinstance(rows, list):
            continue
        metadata[creator_id] = [
            row for row in rows
            if isinstance(row, dict) and row.get("video_id")
        ]
    return metadata


def _caption_paths(data_dir: Path) -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = defaultdict(dict)
    caption_dir = data_dir / "raw_captions"
    if not caption_dir.exists():
        return {}

    for path in caption_dir.glob("*.json"):
        parsed = _creator_from_caption(path)
        if not parsed:
            continue
        creator_id, video_id = parsed
        paths[creator_id][video_id] = path
    return dict(paths)


def _visual_video_ids(data_dir: Path) -> dict[str, set[str]]:
    videos: dict[str, set[str]] = defaultdict(set)
    visual_dir = data_dir / "raw_visual"
    if not visual_dir.exists():
        return {}

    for path in visual_dir.glob("*.json"):
        payload = _read_json(path)
        if isinstance(payload, dict):
            videos[path.stem].update(str(k) for k in payload.keys())
    return dict(videos)


def _embedding_video_ids(
    data_dir: Path,
    known_creators: set[str],
) -> tuple[dict[str, set[str]], dict[str, bool]]:
    segment_videos: dict[str, set[str]] = defaultdict(set)
    creator_embeddings: dict[str, bool] = defaultdict(bool)
    embedding_dir = data_dir / "embeddings_store"
    if not embedding_dir.exists():
        return {}, {}

    for path in embedding_dir.iterdir():
        if not path.is_file():
            continue
        creator_id = _creator_from_embedding(path, known_creators)
        if not creator_id:
            continue
        if path.name.endswith(".npy") or path.name.endswith(".json"):
            creator_embeddings[creator_id] = True
        if not path.name.endswith(".segments.json"):
            continue
        payload = _read_json(path)
        if isinstance(payload, list):
            segment_videos[creator_id].update(
                str(item["video_id"])
                for item in payload
                if isinstance(item, dict) and item.get("video_id")
            )
    return dict(segment_videos), dict(creator_embeddings)


def _label_rows_by_creator(
    data_dir: Path,
) -> dict[str, dict[str, dict[str, str]]]:
    labels: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in load_label_rows(data_dir):
        labels[row["creator_id"]][row["video_id"]] = row
    return dict(labels)


def infer_creator_local_performance_labels(
    videos: list[dict[str, Any]],
    human_labels: dict[str, str] | None = None,
    *,
    min_videos: int = 5,
) -> dict[str, dict[str, Any]]:
    human_labels = human_labels or {}
    results: dict[str, dict[str, Any]] = {}

    for video in videos:
        video_id = str(video.get("video_id") or "")
        if not video_id:
            continue
        human_label = (human_labels.get(video_id) or "").strip().lower()
        if human_label in VALID_PERFORMANCE_LABELS:
            results[video_id] = {"label": human_label, "source": "human"}

    all_view_candidates = [
        (str(video.get("video_id")), _safe_float(video.get("views")))
        for video in videos
        if video.get("video_id")
    ]
    all_view_candidates = [
        (video_id, views)
        for video_id, views in all_view_candidates
        if views is not None
    ]
    missing_candidates = [
        (video_id, views)
        for video_id, views in all_view_candidates
        if video_id not in results
    ]

    if len(all_view_candidates) < min_videos:
        for video_id, _ in missing_candidates:
            results[video_id] = {"label": None, "source": "insufficient"}
        return results

    ranked = sorted(all_view_candidates, key=lambda item: item[1])
    edge_count = max(1, math.ceil(len(ranked) * 0.2))
    bottom_ids = {video_id for video_id, _ in ranked[:edge_count]}
    top_ids = {video_id for video_id, _ in ranked[-edge_count:]}

    for video_id, _views in missing_candidates:
        if video_id in bottom_ids:
            label = "miss"
        elif video_id in top_ids:
            label = "hit"
        else:
            label = "ok"
        results[video_id] = {
            "label": label,
            "source": "inferred_creator_local_views",
        }

    return results


def _top_topics_from_draft(draft: dict[str, Any], limit: int = 5) -> list[str]:
    topics = (
        draft.get("profile_nlp", {})
        .get("topic_profile", {})
        .get("topics", [])
    )
    top_topics = []
    for topic in topics:
        terms = topic.get("top_terms") if isinstance(topic, dict) else None
        if terms:
            top_topics.append(", ".join(str(term) for term in terms[:3]))
        if len(top_topics) >= limit:
            break
    return top_topics


def calculate_coverage_tier(
    *,
    metadata_count: int,
    caption_count: int,
    visual_count: int,
    label_count: int,
    embedding_video_count: int,
) -> str:
    if (
        visual_count > 0
        and label_count > 0
        and (embedding_video_count > 0 or caption_count > 0)
    ):
        return "deep_style_ready"
    if caption_count > 0 or embedding_video_count > 0:
        return "semantic_ready"
    if metadata_count > 0:
        return "metadata_only"
    return "metadata_only"


def calculate_confidence_level(
    *,
    video_count: int,
    caption_count: int,
    visual_count: int,
    label_count: int,
    embedding_video_count: int,
) -> str:
    if (
        video_count >= 10
        and caption_count >= 10
        and visual_count >= 10
        and label_count >= 10
        and embedding_video_count >= 5
    ):
        return "high"
    signals = sum([
        video_count >= 5,
        caption_count >= 5,
        visual_count >= 5,
        label_count >= 3,
        embedding_video_count >= 3,
    ])
    if signals >= 3:
        return "medium"
    return "low"


def recommend_next_processing_step(
    *,
    video_count: int,
    caption_count: int,
    visual_count: int,
    label_count: int,
    embedding_video_count: int,
) -> str:
    if video_count == 0:
        return "add_metadata"
    if caption_count < video_count:
        return "generate_captions"
    if embedding_video_count < caption_count:
        return "build_embeddings"
    if visual_count < video_count:
        return "run_visual_processing"
    if label_count < min(video_count, 5):
        return "label_creator_videos"
    return "ready_for_idea_review"


def _creator_record(
    *,
    creator_id: str,
    metadata_rows: list[dict[str, Any]],
    caption_paths: dict[str, Path],
    visual_ids: set[str],
    label_rows: dict[str, dict[str, str]],
    embedding_ids: set[str],
    creator_embedding_exists: bool,
    draft: dict[str, Any],
) -> dict[str, Any]:
    metadata_by_id = {str(row["video_id"]): row for row in metadata_rows if row.get("video_id")}
    label_ids = set(label_rows.keys())
    video_ids = set(metadata_by_id)
    video_ids.update(caption_paths.keys())
    video_ids.update(visual_ids)
    video_ids.update(label_ids)
    video_ids.update(embedding_ids)

    human_performance = {
        video_id: row.get("performance_label", "")
        for video_id, row in label_rows.items()
    }
    inferred = infer_creator_local_performance_labels(
        list(metadata_by_id.values()),
        human_performance,
    )
    final_labels = {
        video_id: info
        for video_id, info in inferred.items()
        if info.get("label") in VALID_PERFORMANCE_LABELS
    }

    distribution = Counter(info["label"] for info in final_labels.values())
    label_sources = Counter(info["source"] for info in inferred.values())

    format_counts = Counter(
        row.get("format_label")
        for row in label_rows.values()
        if row.get("format_label")
    )
    if not format_counts:
        format_counts.update(
            row.get("format_pred")
            for row in metadata_rows
            if row.get("format_pred")
        )
    if not format_counts:
        format_counts.update(
            draft.get("observed_patterns", {}).get("dominant_formats") or []
        )

    views = [
        _safe_int(row.get("views"))
        for row in metadata_rows
        if _safe_int(row.get("views")) is not None
    ]
    engagement_rates = []
    for row in metadata_rows:
        views_value = _safe_float(row.get("views"))
        if not views_value:
            continue
        likes = _safe_float(row.get("likes")) or 0.0
        comments = _safe_float(row.get("comments")) or 0.0
        engagement_rates.append((likes + comments) / views_value)

    caption_count = len(caption_paths)
    visual_count = len(visual_ids)
    label_count = len(label_rows)
    embedding_video_count = len(embedding_ids)
    video_count = len(video_ids)
    metadata_count = len(metadata_by_id)

    coverage_tier = calculate_coverage_tier(
        metadata_count=metadata_count,
        caption_count=caption_count,
        visual_count=visual_count,
        label_count=label_count,
        embedding_video_count=embedding_video_count,
    )
    confidence = calculate_confidence_level(
        video_count=video_count,
        caption_count=caption_count,
        visual_count=visual_count,
        label_count=label_count,
        embedding_video_count=embedding_video_count,
    )
    next_step = recommend_next_processing_step(
        video_count=video_count,
        caption_count=caption_count,
        visual_count=visual_count,
        label_count=label_count,
        embedding_video_count=embedding_video_count,
    )

    if label_sources.get("human", 0) == len(final_labels) and final_labels:
        label_source = "human"
    elif label_sources.get("human", 0) and label_sources.get("inferred_creator_local_views", 0):
        label_source = "mixed_human_inferred"
    elif label_sources.get("inferred_creator_local_views", 0):
        label_source = "inferred_creator_local_views"
    else:
        label_source = "insufficient"

    return {
        "creator_id": creator_id,
        "video_count": video_count,
        "metadata_video_count": metadata_count,
        "videos_with_captions": caption_count,
        "videos_with_visual_artifacts": visual_count,
        "videos_with_labels": label_count,
        "videos_with_embeddings": embedding_video_count,
        "creator_embedding_exists": bool(creator_embedding_exists),
        "coverage_tier": coverage_tier,
        "dominant_formats": [fmt for fmt, _ in format_counts.most_common(5)],
        "top_topics": _top_topics_from_draft(draft),
        "hit_ok_miss_distribution": {
            label: distribution.get(label, 0)
            for label in PERFORMANCE_LABELS
        },
        "label_source": label_source,
        "label_source_counts": dict(label_sources),
        "median_views": _median(views),
        "engagement_proxies": {
            "median_like_comment_rate": _median(engagement_rates),
            "videos_with_engagement": len(engagement_rates),
        },
        "confidence_level": confidence,
        "recommended_next_processing_step": next_step,
    }


def build_market_index(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    data_dir = Path(data_dir)
    metadata = _load_metadata(data_dir)
    captions = _caption_paths(data_dir)
    visuals = _visual_video_ids(data_dir)
    label_rows = _label_rows_by_creator(data_dir)

    known_creators = set(metadata) | set(captions) | set(visuals) | set(label_rows)
    drafts = {}
    draft_dir = data_dir / "drafts"
    if draft_dir.exists():
        for path in draft_dir.glob("*_draft.yaml"):
            creator_id = path.name[: -len("_draft.yaml")]
            known_creators.add(creator_id)
            drafts[creator_id] = _read_yaml(path)

    embeddings, creator_embedding_flags = _embedding_video_ids(data_dir, known_creators)
    known_creators.update(embeddings)
    known_creators.update(creator_embedding_flags)

    creators = [
        _creator_record(
            creator_id=creator_id,
            metadata_rows=metadata.get(creator_id, []),
            caption_paths=captions.get(creator_id, {}),
            visual_ids=visuals.get(creator_id, set()),
            label_rows=label_rows.get(creator_id, {}),
            embedding_ids=embeddings.get(creator_id, set()),
            creator_embedding_exists=creator_embedding_flags.get(creator_id, False),
            draft=drafts.get(creator_id, {}),
        )
        for creator_id in sorted(known_creators)
    ]

    return {
        "schema_version": "market_index.v1",
        "data_dir": str(data_dir),
        "summary": summarize_market(creators, label_rows, metadata, drafts),
        "creators": creators,
    }


def summarize_market(
    creators: list[dict[str, Any]],
    label_rows: dict[str, dict[str, dict[str, str]]] | None = None,
    metadata: dict[str, list[dict[str, Any]]] | None = None,
    drafts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    label_rows = label_rows or {}
    metadata = metadata or {}
    drafts = drafts or {}
    tier_counts = Counter(c["coverage_tier"] for c in creators)

    format_totals: dict[str, Counter] = defaultdict(Counter)
    for creator_id, rows_by_video in label_rows.items():
        for row in rows_by_video.values():
            fmt = row.get("format_label") or "unknown"
            performance = (row.get("performance_label") or "").lower()
            if performance in VALID_PERFORMANCE_LABELS:
                format_totals[fmt][performance] += 1

    if not format_totals:
        for rows in metadata.values():
            for row in rows:
                fmt = row.get("format_pred") or "unknown"
                format_totals[fmt]["ok"] += 1

    formats_by_hit_rate = []
    for fmt, counts in format_totals.items():
        total = sum(counts.values())
        if not total:
            continue
        formats_by_hit_rate.append({
            "format": fmt,
            "hit_rate": round(counts.get("hit", 0) / total, 4),
            "hit_count": counts.get("hit", 0),
            "ok_count": counts.get("ok", 0),
            "miss_count": counts.get("miss", 0),
            "labeled_count": total,
        })
    formats_by_hit_rate.sort(
        key=lambda row: (row["hit_rate"], row["labeled_count"]),
        reverse=True,
    )

    topic_counts = Counter()
    topic_ready_counts = Counter()
    coverage_by_creator = {c["creator_id"]: c["coverage_tier"] for c in creators}
    for creator_id, draft in drafts.items():
        for topic in _top_topics_from_draft(draft):
            topic_counts[topic] += 1
            if coverage_by_creator.get(creator_id) == "deep_style_ready":
                topic_ready_counts[topic] += 1

    return {
        "total_creators": len(creators),
        "total_videos": sum(c["video_count"] for c in creators),
        "creators_by_coverage_tier": dict(tier_counts),
        "formats_by_hit_rate": formats_by_hit_rate[:10],
        "topics_by_coverage": [
            {
                "topic": topic,
                "creator_count": count,
                "deep_style_ready_creators": topic_ready_counts.get(topic, 0),
            }
            for topic, count in topic_counts.most_common(10)
        ],
        "creators_needing_captions": [
            c["creator_id"] for c in creators
            if c["recommended_next_processing_step"] == "generate_captions"
        ],
        "creators_needing_visual_processing": [
            c["creator_id"] for c in creators
            if c["recommended_next_processing_step"] == "run_visual_processing"
        ],
        "creators_needing_labels": [
            c["creator_id"] for c in creators
            if c["recommended_next_processing_step"] == "label_creator_videos"
        ],
        "creators_ready_for_idea_review": [
            c["creator_id"] for c in creators
            if c["recommended_next_processing_step"] == "ready_for_idea_review"
        ],
        "recommended_processing_queue": [
            {
                "creator_id": c["creator_id"],
                "next_step": c["recommended_next_processing_step"],
                "coverage_tier": c["coverage_tier"],
            }
            for c in creators
            if c["recommended_next_processing_step"] != "ready_for_idea_review"
        ][:20],
    }


def write_market_index(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_path: Path = DEFAULT_MARKET_INDEX_PATH,
) -> dict[str, Any]:
    index = build_market_index(data_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(index, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return index
