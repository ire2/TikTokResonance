import yaml
from pathlib import Path
import shutil
import os

from profiling.ingestion.ingest_creator import ingest_creator
from .profile_generator import generate_profile, write_profile


# ------------------------
# Configuration
# ------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "input" / "creators.yaml"
RAW_DATA_PATH = BASE_DIR / "metadata" / "raw_data" / "creator_metadata.json"
DRAFTS_DIR = BASE_DIR / "drafts"

CLEAN_RUN = os.getenv("CLEAN_RUN", "false").lower() == "true"


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
    - callable by ConstraintSpace

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
    return output_path


def run_profiling_from_yaml():
    """
    Batch profiling entrypoint.
    Reads creators.yaml and generates draft profiles.
    """

    if CLEAN_RUN:
        clear_pycache(BASE_DIR)

    with open(INPUT_PATH, "r") as f:
        config = yaml.safe_load(f)

    if not config or "creators" not in config:
        raise ValueError("Invalid creators.yaml format")

    for creator in config["creators"]:
        creator_id = creator["id"]
        video_limit = creator.get("video_limit", 30)

        run_profiling_for_creator(
            creator_id=creator_id,
            video_limit=video_limit
        )

    print("\nProfiling complete. All drafts ready.")


# ------------------------
# Script entrypoint
# ------------------------

if __name__ == "__main__":
    run_profiling_from_yaml()
