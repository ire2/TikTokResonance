from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_DATA_DIR = Path("data")
PERFORMANCE_LABELS = ("hit", "ok", "miss")


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
    if stem.startswith("idea_"):
        return None
    if "_" not in stem:
        return None
    creator_id, video_id = stem.rsplit("_", 1)
    if not creator_id or not video_id:
        return None
    return creator_id, video_id


def _strip_embedding_suffix(name: str) -> str | None:
    for suffix in (".segments.json", ".segments.npy", ".json", ".npy"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _creator_from_embedding_name(name: str) -> str | None:
    base = _strip_embedding_suffix(name)
    if not base:
        return None

    for marker in ("_BAAI_", "_all-", "_sentence-", "_intfloat_"):
        if marker in base:
            return base.split(marker, 1)[0]

    if "_" not in base:
        return None
    return base.rsplit("_", 1)[0]


def load_label_rows(data_dir: Path = DEFAULT_DATA_DIR) -> list[dict[str, str]]:
    labels_path = data_dir / "labels" / "format_labels.csv"
    if not labels_path.exists():
        return []

    rows: list[dict[str, str]] = []
    try:
        with labels_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                creator_id = (row.get("creator_id") or "").strip()
                video_id = (row.get("video_id") or "").strip()
                if not creator_id or not video_id:
                    continue
                rows.append({k: (v or "").strip() for k, v in row.items()})
    except OSError:
        return []
    return rows


def load_label_index(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, dict[str, dict[str, str]]]:
    index: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in load_label_rows(data_dir):
        index[row["creator_id"]][row["video_id"]] = row
    return dict(index)


def _discover_caption_videos(
    data_dir: Path,
) -> dict[str, set[str]]:
    videos: dict[str, set[str]] = defaultdict(set)
    caption_dir = data_dir / "raw_captions"
    if not caption_dir.exists():
        return {}

    for path in caption_dir.glob("*.json"):
        parsed = _creator_from_caption(path)
        if not parsed:
            continue
        creator_id, video_id = parsed
        videos[creator_id].add(video_id)
    return dict(videos)


def _discover_visual_videos(
    data_dir: Path,
) -> dict[str, set[str]]:
    videos: dict[str, set[str]] = defaultdict(set)
    visual_dir = data_dir / "raw_visual"
    if not visual_dir.exists():
        return {}

    for path in visual_dir.glob("*.json"):
        payload = _read_json(path)
        if isinstance(payload, dict):
            videos[path.stem].update(str(k) for k in payload.keys())
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and item.get("video_id"):
                    videos[path.stem].add(str(item["video_id"]))
    return dict(videos)


def _discover_embedding_files(
    data_dir: Path,
    known_creators: set[str],
) -> dict[str, list[Path]]:
    embedding_dir = data_dir / "embeddings_store"
    files_by_creator: dict[str, list[Path]] = defaultdict(list)
    if not embedding_dir.exists():
        return {}

    known_by_length = sorted(known_creators, key=len, reverse=True)
    for path in embedding_dir.iterdir():
        if not path.is_file():
            continue

        creator_id = None
        for known in known_by_length:
            if path.name.startswith(f"{known}_"):
                creator_id = known
                break
        if creator_id is None:
            creator_id = _creator_from_embedding_name(path.name)
        if creator_id:
            files_by_creator[creator_id].append(path)

    return dict(files_by_creator)


def _count_segment_memory(paths: list[Path]) -> int:
    segment_paths = [p for p in paths if p.name.endswith(".segments.json")]
    if not segment_paths:
        return 0

    payload = _read_json(sorted(segment_paths)[0])
    return len(payload) if isinstance(payload, list) else 0


def _confidence_level(
    *,
    caption_count: int,
    human_label_count: int,
    visual_signal_count: int,
    embeddings_exist: bool,
) -> tuple[str, str, list[str]]:
    drivers = []
    if caption_count >= 20:
        drivers.append("20+ captions")
    elif caption_count >= 5:
        drivers.append("some captions")

    if human_label_count >= 10:
        drivers.append("10+ human labels")
    elif human_label_count >= 3:
        drivers.append("some human labels")

    if visual_signal_count >= 10:
        drivers.append("10+ visual analyses")
    elif visual_signal_count >= 3:
        drivers.append("some visual analyses")

    if embeddings_exist:
        drivers.append("segment embeddings")

    if (
        caption_count >= 20
        and human_label_count >= 10
        and visual_signal_count >= 10
        and embeddings_exist
    ):
        return (
            "high",
            "High confidence because captions, labels, visual signals, and segment memory are all present.",
            drivers,
        )

    medium_signals = sum(
        [
            caption_count >= 5,
            human_label_count >= 3,
            visual_signal_count >= 3,
            embeddings_exist,
        ]
    )
    if medium_signals >= 2:
        return (
            "medium",
            "Medium confidence because useful evidence exists, but at least one coverage pillar is thin.",
            drivers,
        )

    return (
        "low",
        "Low confidence because there is too little local evidence for a strong recommendation.",
        drivers,
    )


def discover_creator_ids(data_dir: Path = DEFAULT_DATA_DIR) -> set[str]:
    creator_ids: set[str] = set()

    drafts_dir = data_dir / "drafts"
    if drafts_dir.exists():
        for path in drafts_dir.glob("*_draft.yaml"):
            creator_ids.add(path.name[: -len("_draft.yaml")])

    creator_ids.update(_discover_caption_videos(data_dir).keys())
    creator_ids.update(_discover_visual_videos(data_dir).keys())
    creator_ids.update(row["creator_id"] for row in load_label_rows(data_dir))
    creator_ids.update(_discover_embedding_files(data_dir, creator_ids).keys())
    return creator_ids


def build_creator_library(data_dir: Path = DEFAULT_DATA_DIR) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    caption_videos = _discover_caption_videos(data_dir)
    visual_videos = _discover_visual_videos(data_dir)
    label_rows = load_label_rows(data_dir)

    labels_by_creator: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in label_rows:
        labels_by_creator[row["creator_id"]].append(row)

    creator_ids = discover_creator_ids(data_dir)
    embedding_files = _discover_embedding_files(data_dir, creator_ids)

    creators: list[dict[str, Any]] = []
    for creator_id in sorted(creator_ids):
        draft_path = data_dir / "drafts" / f"{creator_id}_draft.yaml"
        draft = _read_yaml(draft_path) if draft_path.exists() else {}

        rows = labels_by_creator.get(creator_id, [])
        label_video_ids = {row["video_id"] for row in rows if row.get("video_id")}
        video_ids = set()
        video_ids.update(caption_videos.get(creator_id, set()))
        video_ids.update(visual_videos.get(creator_id, set()))
        video_ids.update(label_video_ids)

        perf_counts = Counter(
            (row.get("performance_label") or "").strip().lower()
            for row in rows
        )
        hit_ok_miss = {
            label: perf_counts.get(label, 0)
            for label in PERFORMANCE_LABELS
        }

        format_counts = Counter(
            row.get("format_label")
            for row in rows
            if row.get("format_label")
        )
        dominant_formats = [label for label, _ in format_counts.most_common(3)]
        if not dominant_formats:
            dominant_formats = (
                draft.get("observed_patterns", {}).get("dominant_formats") or []
            )[:3]

        files = embedding_files.get(creator_id, [])
        embeddings_exist = any(
            p.name.endswith(".npy") or p.name.endswith(".json")
            for p in files
        )
        segment_count = _count_segment_memory(files)
        if segment_count:
            video_ids.update(
                str(item.get("video_id"))
                for p in files
                if p.name.endswith(".segments.json")
                for item in (_read_json(p) or [])
                if isinstance(item, dict) and item.get("video_id")
            )

        confidence, confidence_reason, drivers = _confidence_level(
            caption_count=len(caption_videos.get(creator_id, set())),
            human_label_count=len(rows),
            visual_signal_count=len(visual_videos.get(creator_id, set())),
            embeddings_exist=embeddings_exist,
        )

        creators.append({
            "creator_id": creator_id,
            "videos_analyzed": len(video_ids),
            "visual_signal_count": len(visual_videos.get(creator_id, set())),
            "caption_count": len(caption_videos.get(creator_id, set())),
            "human_label_count": len(rows),
            "hit_count": hit_ok_miss["hit"],
            "ok_count": hit_ok_miss["ok"],
            "miss_count": hit_ok_miss["miss"],
            "dominant_formats": dominant_formats,
            "embeddings_exist": embeddings_exist,
            "segment_count": segment_count,
            "confidence_level": confidence,
            "confidence_reason": confidence_reason,
            "confidence_drivers": drivers,
            "artifact_paths": {
                "draft": str(draft_path) if draft_path.exists() else None,
                "raw_visual": str(data_dir / "raw_visual" / f"{creator_id}.json")
                if (data_dir / "raw_visual" / f"{creator_id}.json").exists()
                else None,
                "labels": str(data_dir / "labels" / "format_labels.csv")
                if rows
                else None,
                "embeddings": sorted(str(p) for p in files),
            },
        })

    confidence_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        creators,
        key=lambda c: (
            confidence_order.get(c["confidence_level"], 3),
            -c["videos_analyzed"],
            c["creator_id"],
        ),
    )


def get_creator(
    creator_id: str,
    data_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, Any] | None:
    creator_id = (creator_id or "").strip()
    for creator in build_creator_library(data_dir):
        if creator["creator_id"] == creator_id:
            return creator
    return None
