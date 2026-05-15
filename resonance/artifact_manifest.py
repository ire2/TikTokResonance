from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from resonance.creator_library import DEFAULT_DATA_DIR, load_label_rows
from resonance.market_cohort import (
    calculate_coverage_tier,
    write_market_index,
)


DEFAULT_MANIFEST_PATH = Path("data/artifacts/media_manifest.json")
MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _creator_from_caption(path: Path) -> tuple[str, str] | None:
    stem = path.stem
    if stem.startswith("idea_") or "_" not in stem:
        return None
    creator_id, video_id = stem.rsplit("_", 1)
    if not creator_id or not video_id:
        return None
    return creator_id, video_id


def _known_creator_from_name(path: Path, known_creators: set[str]) -> str | None:
    for creator_id in sorted(known_creators, key=len, reverse=True):
        if path.name.startswith(f"{creator_id}_"):
            return creator_id
    return None


def _metadata_index(data_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    path = data_dir / "raw_data" / "creator_metadata.json"
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {}

    index: dict[str, dict[str, dict[str, Any]]] = {}
    for creator_id, rows in payload.items():
        if not isinstance(rows, list):
            continue
        index[creator_id] = {
            str(row["video_id"]): row
            for row in rows
            if isinstance(row, dict) and row.get("video_id")
        }
    return index


def _caption_index(data_dir: Path) -> dict[str, dict[str, Path]]:
    index: dict[str, dict[str, Path]] = {}
    caption_dir = data_dir / "raw_captions"
    if not caption_dir.exists():
        return index

    for path in caption_dir.glob("*.json"):
        parsed = _creator_from_caption(path)
        if not parsed:
            continue
        creator_id, video_id = parsed
        index.setdefault(creator_id, {})[video_id] = path
    return index


def _visual_index(data_dir: Path) -> dict[str, dict[str, Path]]:
    index: dict[str, dict[str, Path]] = {}
    visual_dir = data_dir / "raw_visual"
    if not visual_dir.exists():
        return index

    for path in visual_dir.glob("*.json"):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        index[path.stem] = {str(video_id): path for video_id in payload.keys()}
    return index


def _label_index(data_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
    index: dict[str, dict[str, dict[str, str]]] = {}
    for row in load_label_rows(data_dir):
        index.setdefault(row["creator_id"], {})[row["video_id"]] = row
    return index


def _embedding_index(
    data_dir: Path,
    known_creators: set[str],
) -> tuple[dict[str, set[str]], set[str]]:
    segment_index: dict[str, set[str]] = {}
    creator_embedding_ids: set[str] = set()
    embedding_dir = data_dir / "embeddings_store"
    if not embedding_dir.exists():
        return segment_index, creator_embedding_ids

    for path in embedding_dir.iterdir():
        if not path.is_file():
            continue
        creator_id = _known_creator_from_name(path, known_creators)
        if not creator_id:
            continue
        if path.suffix in {".json", ".npy"}:
            creator_embedding_ids.add(creator_id)
        if not path.name.endswith(".segments.json"):
            continue
        payload = _read_json(path)
        if isinstance(payload, list):
            segment_index.setdefault(creator_id, set()).update(
                str(item["video_id"])
                for item in payload
                if isinstance(item, dict) and item.get("video_id")
            )
    return segment_index, creator_embedding_ids


def _raw_media_paths(data_dir: Path) -> dict[tuple[str, str], Path]:
    found: dict[tuple[str, str], Path] = {}
    for folder in (data_dir / "raw_videos", data_dir / "test" / "video"):
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            if "_" not in path.stem:
                continue
            creator_id, video_id = path.stem.rsplit("_", 1)
            found[(creator_id, video_id)] = path
    return found


def _relative_or_none(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def build_artifact_manifest(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
    data_dir = Path(data_dir)
    metadata = _metadata_index(data_dir)
    captions = _caption_index(data_dir)
    visuals = _visual_index(data_dir)
    labels = _label_index(data_dir)
    known_creators = set(metadata) | set(captions) | set(visuals) | set(labels)

    draft_dir = data_dir / "drafts"
    if draft_dir.exists():
        known_creators.update(path.name[: -len("_draft.yaml")] for path in draft_dir.glob("*_draft.yaml"))

    embeddings, creator_embedding_ids = _embedding_index(data_dir, known_creators)
    raw_media = _raw_media_paths(data_dir)

    rows = []
    for creator_id in sorted(known_creators):
        video_ids = set(metadata.get(creator_id, {}))
        video_ids.update(captions.get(creator_id, {}))
        video_ids.update(visuals.get(creator_id, {}))
        video_ids.update(labels.get(creator_id, {}))
        video_ids.update(embeddings.get(creator_id, set()))

        for video_id in sorted(video_ids):
            metadata_row = metadata.get(creator_id, {}).get(video_id)
            caption_path = captions.get(creator_id, {}).get(video_id)
            visual_path = visuals.get(creator_id, {}).get(video_id)
            label_row = labels.get(creator_id, {}).get(video_id)
            has_segment = video_id in embeddings.get(creator_id, set())
            raw_path = raw_media.get((creator_id, video_id))
            metadata_local_path = Path(metadata_row["local_path"]) if metadata_row and metadata_row.get("local_path") else None
            raw_media_present = bool(
                (raw_path and raw_path.exists())
                or (metadata_local_path and metadata_local_path.exists())
            )

            metadata_count = 1 if metadata_row else 0
            caption_count = 1 if caption_path else 0
            visual_count = 1 if visual_path else 0
            label_count = 1 if label_row else 0
            embedding_count = 1 if has_segment else 0
            coverage_tier = calculate_coverage_tier(
                metadata_count=metadata_count,
                caption_count=caption_count,
                visual_count=visual_count,
                label_count=label_count,
                embedding_video_count=embedding_count,
            )

            source_url = None
            if label_row and label_row.get("tiktok_url"):
                source_url = label_row["tiktok_url"]

            if has_segment:
                embedding_status = "segment_memory"
            elif creator_id in creator_embedding_ids:
                embedding_status = "creator_embedding_only"
            else:
                embedding_status = "missing"

            notes = []
            if not caption_path:
                notes.append("caption_missing")
            if not visual_path:
                notes.append("visual_artifact_missing")
            if not label_row:
                notes.append("human_label_missing")
            if not has_segment:
                notes.append("segment_embedding_missing")

            rows.append({
                "creator_id": creator_id,
                "video_id": video_id,
                "source_url": source_url,
                "metadata_present": bool(metadata_row),
                "caption_path": _relative_or_none(caption_path),
                "visual_artifact_path": _relative_or_none(visual_path),
                "embedding_status": embedding_status,
                "label_status": "human" if label_row else "missing",
                "raw_media_present": raw_media_present,
                "coverage_tier": coverage_tier,
                "processed_at": None,
                "notes": notes,
            })

    return {
        "schema_version": "media_manifest.v1",
        "data_dir": str(data_dir),
        "summary": {
            "videos": len(rows),
            "raw_media_present": sum(1 for row in rows if row["raw_media_present"]),
            "metadata_present": sum(1 for row in rows if row["metadata_present"]),
            "captions_present": sum(1 for row in rows if row["caption_path"]),
            "visual_artifacts_present": sum(1 for row in rows if row["visual_artifact_path"]),
            "segment_memory_present": sum(1 for row in rows if row["embedding_status"] == "segment_memory"),
            "human_labels_present": sum(1 for row in rows if row["label_status"] == "human"),
        },
        "videos": rows,
    }


def write_artifact_manifest(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_artifact_manifest(data_dir)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def ensure_artifact_indexes(
    data_dir: Path = DEFAULT_DATA_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    market_index = write_market_index(data_dir)
    manifest = write_artifact_manifest(data_dir, manifest_path)
    return market_index, manifest
