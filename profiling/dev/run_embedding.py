# profiling/dev/run_embedding.py

from pathlib import Path

from profiling.embedding.embedder import TextEmbedder
from profiling.embedding.creator_embedding import build_creator_embedding
from profiling.embedding.embedding_store import save_creator_embedding
from profiling.nlp.transcript_loader import load_transcript
from profiling.nlp.dialogue_chunker import chunk_dialogue
from profiling.utils.creator_config import (
    get_active_creator,
    get_default_caption_limit,
    get_default_model_name,
)

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parents[2]
CREATOR_ID = get_active_creator()
MODEL_NAME = get_default_model_name()
CAPTION_DIR = Path("data/raw_captions")
LIMIT = get_default_caption_limit()
# ---------------------------------------


def main():
    caption_files = sorted(
        CAPTION_DIR.glob(f"{CREATOR_ID}_*.json")
    )[:LIMIT]

    if not caption_files:
        print("[EMBED DEBUG] No caption files found.")
        return

    segments = []

    for cap_path in caption_files:
        transcript = load_transcript(cap_path)
        result = chunk_dialogue(transcript)

        for d in result.get("dialogue", []):
            d["video_id"] = cap_path.stem.replace(f"{CREATOR_ID}_", "")
            segments.append(d)

    if not segments:
        print("[EMBED DEBUG] No dialogue segments extracted.")
        return

    embedder = TextEmbedder(model_name=MODEL_NAME)

    print(f"[DEBUG] segments passed to embedder: {len(segments)}")

    payload = build_creator_embedding(
        transcript_segments=segments,
        embedder=embedder,
    )

    save_creator_embedding(
        creator_id=CREATOR_ID,
        model_name=MODEL_NAME,
        payload=payload,
    )

    print("\n[EMBED DEBUG RESULT]")
    print({
        "creator": CREATOR_ID,
        "model": MODEL_NAME,
        "segments_used": payload["num_segments"],
        "embedding_dim": payload["dim"],
    })


if __name__ == "__main__":
    main()
