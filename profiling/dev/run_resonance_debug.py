from pathlib import Path
import yaml

from profiling.embedding.embedder import TextEmbedder
from profiling.resonance.idea_encoder import encode_idea
from profiling.resonance.resonance_score import compute_resonance
from profiling.resonance.resonance_report import build_resonance_report
from profiling.embedding.embedding_store import load_creator_embeddings


# ---------------- CONFIG ----------------
CREATOR_ID = "expoparker"
MODEL_NAME = "BAAI/bge-large-en-v1.5"

PROFILE_PATH = Path("profiling/drafts/expoparker_draft.yaml")

IDEA_TEXT = """
Awkward public interaction where the guy misunderstands a normal request
and escalates it into an uncomfortable flirt, then gets called out.
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
