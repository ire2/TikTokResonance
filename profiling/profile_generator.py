import json
import yaml
from pathlib import Path
from utils.trace import trace

from profiling.profile.visual_profile import compute_visual_profile
from profiling.profile.nlp_profile import compute_nlp_profile
from profiling.profile.metadata_profile import compute_metadata_profile
from profiling.profile.nlp_gate import compute_nlp_gate
from profiling.embedding.embedding_store import load_creator_embeddings
from profiling.utils.creator_config import get_default_model_name


@trace
def generate_profile(creator_id: str, raw_data_path: str) -> dict:
    data = json.load(open(raw_data_path))
    videos = data[creator_id]

    # ---- compute components ----
    visual = compute_visual_profile(creator_id, videos)
    nlp = compute_nlp_profile(creator_id, videos)
    meta = compute_metadata_profile(videos)
    gate = compute_nlp_gate(visual)

    embedding_meta = load_creator_embeddings(
        creator_id=creator_id,
        model_name=get_default_model_name(),
    )

    # ---- assemble profile ----
    profile = {
        "creator_id": creator_id,
        "generated_by": "Tik Tok Resonance Profiler v0.1",
        "status": "draft",
        "analysis_window": f"last_{len(videos)}_videos",

        "observed_patterns": {
            "dominant_formats": meta["dominant_formats"],
            "underused_formats": meta["underused_formats"],
            "avg_duration_sec": meta["avg_duration_sec"],
        },

        "modality_bias": {
            "voice": "high" if meta["voice_pct"] > 0.75 else "medium",
            "text": "high" if meta["text_pct"] > 0.75 else "medium",
        },

        "visual_signals": visual,
        "nlp_captioning_gate": gate,
        "profile_nlp": nlp,


        "creator_embedding": (
            {
                "model": embedding_meta["model"],
                "dim": embedding_meta["dim"],
                "num_segments": embedding_meta["num_segments"],
                "has_segment_memory": bool(embedding_meta["segments"]),
            }
            if embedding_meta
            else None
        ),

        "human_review_required": True,
    }

    return profile


@trace
def write_profile(profile: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        yaml.dump(profile, f, sort_keys=False)
