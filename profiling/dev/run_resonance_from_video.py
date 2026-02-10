from pathlib import Path
import json
import os
import yaml

from profiling.embedding.embedder import TextEmbedder
from utils.warnings import silence_common_warnings
from profiling.embedding.embedding_store import load_creator_embeddings
from resonance.idea_encoder import encode_idea
from resonance.resonance_score import compute_resonance
from resonance.resonance_report import build_resonance_report
from profiling.cv.visual_signals import extract_visual_signals
from profiling.cv.learned_classifier import LearnedFormatClassifier
from profiling.nlp.asr import ensure_captions
from profiling.nlp.transcript_loader import load_transcript
from profiling.utils.creator_config import (
    get_active_creator,
    get_default_model_name,
)


TEST_VIDEO_DIR = Path("data/test/video")


def find_latest_video() -> Path:
    if not TEST_VIDEO_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {TEST_VIDEO_DIR}")
    candidates = []
    for ext in ("*.mp4", "*.mov", "*.mkv"):
        candidates.extend(TEST_VIDEO_DIR.glob(ext))
    if not candidates:
        raise FileNotFoundError(f"No videos found in {TEST_VIDEO_DIR}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_idea_text(video_path: Path) -> str:
    creator_id = "idea"
    video_id = video_path.stem
    caption = ensure_captions(
        creator_id=creator_id,
        video_id=video_id,
        video_path=str(video_path),
    )
    if not caption:
        return ""
    transcript = load_transcript(caption)
    return " ".join(s.get("text", "") for s in transcript.get("segments", []))


def main():
    silence_common_warnings()
    creator_id = get_active_creator()
    model_name = get_default_model_name()

    video_path = find_latest_video()
    print(f"[VIDEO] Using {video_path}")
    print(f"[RESONANCE] creator={creator_id} model={model_name}")

    # Load creator profile and embeddings
    profile_path = Path("data/drafts") / f"{creator_id}_draft.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing profile: {profile_path}")
    print(f"[RESONANCE] Loading profile {profile_path}")
    creator_profile = yaml.safe_load(profile_path.read_text())

    print("[RESONANCE] Loading creator embeddings")
    creator_embedding_payload = load_creator_embeddings(
        creator_id=creator_id,
        model_name=model_name,
    )
    if creator_embedding_payload is None:
        raise ValueError("Creator embeddings not found. Run run_embedding.py")
    print(
        f"[RESONANCE] Embeddings loaded (segments={creator_embedding_payload.get('num_segments')}, "
        f"dim={creator_embedding_payload.get('dim')})"
    )

    # Extract idea signals from video
    print("[RESONANCE] Extracting visual signals")
    visual = extract_visual_signals(str(video_path))
    idea_motion_intensity = visual.get("motion_intensity")
    idea_text_density = (
        visual.get("text_density_ocr")
        or visual.get("text_density_heuristic")
    )
    print(
        f"[RESONANCE] Visual signals ready (motion={idea_motion_intensity}, text_density={idea_text_density})"
    )

    # Format via learned classifier (if available)
    format_classifier = LearnedFormatClassifier()
    idea_format = None
    if format_classifier.is_ready():
        print("[RESONANCE] Predicting format with learned classifier")
        idea_format = format_classifier.classify(
            {"local_path": str(video_path)}
        )
        print(f"[RESONANCE] Predicted format={idea_format}")
    else:
        print("[RESONANCE] Format classifier not ready; skipping format prediction")

    print("[RESONANCE] Extracting transcript (ASR if needed)")
    idea_text = extract_idea_text(video_path)
    if not idea_text:
        idea_text = " "

    print("[RESONANCE] Encoding idea text")
    embedder = TextEmbedder(model_name=model_name)
    idea = encode_idea(idea_text.strip(), embedder)

    print("[RESONANCE] Computing resonance")
    resonance = compute_resonance(
        idea_embedding=idea["embedding"],
        creator_embedding_payload=creator_embedding_payload,
        creator_profile=creator_profile,
        idea_motion_intensity=idea_motion_intensity,
        idea_text_density=idea_text_density,
        idea_format=idea_format,
    )

    report = build_resonance_report(
        idea=idea_text.strip(),
        resonance=resonance,
    )

    if os.getenv("RESONANCE_WRITE_CACHE", "false").lower() == "true":
        cache_path = Path(os.getenv(
            "RESONANCE_CACHE_PATH", "data/demo/resonance_cache.json"
        ))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        evidence = report.get("top_similar_moments", [])
        for e in evidence:
            vid = e.get("video_id")
            if vid:
                e["tiktok_url"] = f"https://www.tiktok.com/@{creator_id}/video/{vid}"
        payload = {
            "creator_id": creator_id,
            "model_name": model_name,
            "video_path": str(video_path),
            "idea_text": report["idea_text"],
            "resonance": report["resonance"],
            "interpretation": report["interpretation"],
            "evidence": evidence,
        }
        cache_path.write_text(json.dumps(payload, indent=2))
        print(f"[RESONANCE] Cached results → {cache_path}")

    print("\n========== IDEA ==========")
    print((report["idea_text"] or "").strip()[:1000])

    print("\n====== RESONANCE ======")
    for k, v in report["resonance"].items():
        print(f"{k:>25}: {v}")

    print("\n=== INTERPRETATION ===")
    for k, v in report["interpretation"].items():
        print(f"{k:>25}: {v}")

    print("\n=== TOP EVIDENCE ===")
    evidence = report.get("top_similar_moments", [])
    if not evidence:
        print("No evidence available.")
    else:
        for e in evidence[:5]:
            vid = e.get("video_id", "-")
            sim = e.get("similarity", "-")
            text = (e.get("text") or "").strip()
            print(f"[{vid}] sim={sim} :: {text[:200]}")

    print("\n[RESONANCE VIDEO] Done\n")


if __name__ == "__main__":
    main()
