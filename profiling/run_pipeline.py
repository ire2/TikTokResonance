import yaml
from pathlib import Path
import shutil
import os

from profiling.ingestion.ingest_creator import ingest_creator
from .profile_generator import generate_profile, write_profile
from profiling.dev.generate_label_queue import main as generate_label_queue
from profiling.utils.creator_config import get_active_creator, get_training_creators


# ------------------------
# Configuration
# ------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "input" / "creators.yaml"
RAW_DATA_PATH = Path("data/raw_data/creator_metadata.json")
DRAFTS_DIR = Path("data/drafts")

CLEAN_RUN = os.getenv("CLEAN_RUN", "false").lower() == "true"
GENERATE_LABELS = os.getenv("GENERATE_LABELS", "false").lower() == "true"


# ------------------------
# Utilities
# ------------------------

def clear_pycache(base_dir: Path):
    """
    Development-only helper to ensure clean imports during iteration.
    """
    for pycache in base_dir.rglob("__pycache__"):
        shutil.rmtree(pycache)


# ------------------------
# Core profiling logic
# ------------------------

def run_profiling_for_creator(creator_id: str, video_limit: int = 30) -> Path:
    """
    Run profiling pipeline for a single creator.

    This function is:
    - non-interactive
    - deterministic
    - callable by TikTokResonance

    Returns:
        Path to draft profile YAML
    """

    DRAFTS_DIR.mkdir(exist_ok=True)

    print(f"\n[1/2] Ingesting creator: {creator_id}")
    ingest_creator(creator_id, video_limit=video_limit)

    print(f"[2/2] Generating profile for: {creator_id}")
    profile = generate_profile(
        creator_id=creator_id,
        raw_data_path=RAW_DATA_PATH
    )

    output_path = DRAFTS_DIR / f"{creator_id}_draft.yaml"
    write_profile(profile, output_path)

    print(f"Draft profile written → {output_path}")

    if GENERATE_LABELS:
        print("[LABELS] Generating label queue...")
        generate_label_queue()

    return output_path


def run_profiling_from_yaml():
    """
    Batch profiling entrypoint.
    Reads creators.yaml and generates draft profiles.
    """

    if CLEAN_RUN:
        clear_pycache(BASE_DIR)

    training_creators = get_training_creators()
    if training_creators:
        for creator_id in training_creators:
            run_profiling_for_creator(
                creator_id=creator_id,
                video_limit=30,
            )
    else:
        creator_id = get_active_creator()
        run_profiling_for_creator(
            creator_id=creator_id,
            video_limit=30,
        )

    print("\nProfiling complete. All drafts ready.")


# ------------------------
# Script entrypoint
# ------------------------

if __name__ == "__main__":
    run_profiling_from_yaml()
