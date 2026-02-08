import os
from pathlib import Path
import multiprocessing as mp

from profiling.run_pipeline import run_profiling_for_creator
from profiling.utils.creator_config import (
    get_training_creators,
    get_active_creator,
    get_default_model_name,
    get_default_caption_limit,
)
from profiling.embedding.embedder import TextEmbedder
from profiling.embedding.creator_embedding import build_creator_embedding
from profiling.embedding.embedding_store import save_creator_embedding
from profiling.nlp.transcript_loader import load_transcript
from profiling.nlp.dialogue_chunker import chunk_dialogue

from profiling.dev.generate_label_queue import main as generate_label_queue
from profiling.dev.train_format_classifier import main as train_format_classifier
from profiling.dev.run_resonance_from_video import main as run_resonance_from_video
from profiling.dev.pick_random_video import main as pick_random_video


CAPTION_DIR = Path("data/raw_captions")


def build_embeddings_for_creator(creator_id: str):
    limit = get_default_caption_limit()
    caption_files = sorted(CAPTION_DIR.glob(f"{creator_id}_*.json"))[:limit]
    if not caption_files:
        print(f"[EMBED] No captions found for {creator_id}. Skipping.")
        return

    segments = []
    for cap_path in caption_files:
        transcript = load_transcript(cap_path)
        result = chunk_dialogue(transcript)
        for d in result.get("dialogue", []):
            d["video_id"] = cap_path.stem.replace(f"{creator_id}_", "")
            segments.append(d)

    if not segments:
        print(f"[EMBED] No dialogue segments for {creator_id}. Skipping.")
        return

    model_name = get_default_model_name()
    embedder = TextEmbedder(model_name=model_name)

    payload = build_creator_embedding(
        transcript_segments=segments,
        embedder=embedder,
    )

    save_creator_embedding(
        creator_id=creator_id,
        model_name=model_name,
        payload=payload,
    )

    print(
        f"[EMBED] {creator_id} segments={payload['num_segments']} "
        f"dim={payload['dim']} model={model_name}"
    )


def main():
    training_creators = get_training_creators()
    if not training_creators:
        training_creators = [get_active_creator()]

    print(f"[MAIN] Active creator: {get_active_creator()}")
    print(f"[MAIN] Training creators: {', '.join(training_creators)}")

    run_labels = os.getenv("RUN_LABELS", "false").lower() == "true"
    run_train = os.getenv("RUN_TRAIN", "false").lower() == "true"
    run_resonance = os.getenv("RUN_RESONANCE_TEST", "false").lower() == "true"
    pick_random = os.getenv("PICK_RANDOM_TEST", "false").lower() == "true"
    build_embeddings = os.getenv("BUILD_EMBEDDINGS", "true").lower() == "true"
    run_profiles = os.getenv("RUN_PROFILES", "true").lower() == "true"
    force_profiles = os.getenv("FORCE_PROFILES", "false").lower() == "true"

    ingest_workers = int(os.getenv("INGEST_WORKERS", "1"))

    def _profile_one(cid: str):
        draft_path = Path("data/drafts") / f"{cid}_draft.yaml"
        if draft_path.exists() and not force_profiles:
            print(f"[PROFILE] Skipping {cid} (draft exists)")
            return
        print(f"[PROFILE] Building profile for {cid}")
        run_profiling_for_creator(creator_id=cid, video_limit=30)

    if run_profiles:
        if ingest_workers > 1 and len(training_creators) > 1:
            print(f"[PROFILE] Parallel ingest with {ingest_workers} workers")
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=ingest_workers) as pool:
                pool.map(_profile_one, training_creators)
        else:
            for creator_id in training_creators:
                _profile_one(creator_id)

    for creator_id in training_creators:
        if build_embeddings:
            build_embeddings_for_creator(creator_id)

    if run_labels:
        generate_label_queue()

    if run_train:
        train_format_classifier()

    if pick_random:
        pick_random_video()

    if run_resonance:
        run_resonance_from_video()


if __name__ == "__main__":
    main()
