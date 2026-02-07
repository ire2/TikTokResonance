from pathlib import Path
import yaml

from profiling.embedding.embedder import TextEmbedder
from profiling.resonance.idea_encoder import encode_idea
from profiling.resonance.resonance_score import compute_resonance
from profiling.resonance.resonance_report import build_resonance_report
from profiling.embedding.embedding_store import load_creator_embeddings
from profiling.utils.creator_config import (
    get_active_creator,
    get_default_model_name,
)


# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parents[2]
CREATOR_ID = get_active_creator()
MODEL_NAME = get_default_model_name()
PROFILE_PATH = BASE_DIR / "profiling" / "drafts" / f"{CREATOR_ID}_draft.yaml"
IDEA_MOTION_INTENSITY = 0.2

IDEA_TEXT = """
Mukbang sushi. Mukbang is a genre of social media content where creators eat large quantities of food while interacting with their audience. Sushi is a popular type of Japanese cuisine that typically consists of vinegared rice, seafood, and vegetables. A "mukbang sushi" video would likely feature a creator eating various types of sushi while engaging with their viewers.
"""
# --------------------------------------


def main():
    print("\n[RESONANCE DEBUG] Starting\n")

    if not PROFILE_PATH.exists():
        print("[ERROR] Creator profile not found.")
        return

    creator_profile = yaml.safe_load(PROFILE_PATH.read_text())

    creator_embedding_payload = load_creator_embeddings(
        creator_id=CREATOR_ID,
        model_name=MODEL_NAME,
    )

    if creator_embedding_payload is None:
        print("[ERROR] Creator embedding not found.")
        return

    embedder = TextEmbedder(model_name=MODEL_NAME)
    idea = encode_idea(IDEA_TEXT.strip(), embedder)

    resonance = compute_resonance(
        idea_embedding=idea["embedding"],
        creator_embedding_payload=creator_embedding_payload,
        creator_profile=creator_profile,
        idea_motion_intensity=IDEA_MOTION_INTENSITY,
    )

    report = build_resonance_report(
        idea=idea["text"],
        resonance=resonance,
    )

    print("\n========== IDEA ==========")
    print(report["idea_text"].strip())

    print("\n====== RESONANCE ======")
    for k, v in report["resonance"].items():
        print(f"{k:>25}: {v}")

    print("\n=== INTERPRETATION ===")
    for k, v in report["interpretation"].items():
        print(f"{k:>25}: {v}")

    print("\n[RESONANCE DEBUG] Done\n")


if __name__ == "__main__":
    main()
