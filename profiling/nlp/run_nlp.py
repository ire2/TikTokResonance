import json
from pathlib import Path
from utils.trace import trace

from profiling.nlp.asr import ensure_captions
from profiling.utils.video_paths import video_path


@trace
def run_nlp_for_creator(
    creator_id: str,
    raw_data_path: Path,
    profile: dict,
):
    """
    Run NLP only if CV gate allows it.
    """

    gate = profile.get("nlp_captioning_gate", {})
    weight = gate.get("weight", 0)

    if weight == 0.0:
        print("[NLP] Skipped — CV gate disabled captioning.")
        return {}

    print(f"[NLP] Running captioning with weight={weight}")

    data = json.loads(raw_data_path.read_text())
    videos = data.get(creator_id, [])

    results = []

    for v in videos:
        video_id = v.get("video_id")
        if not video_id:
            continue

        path = video_path(creator_id, video_id)
        if not path.exists():
            continue

        caption_path = ensure_captions(
            creator_id=creator_id,
            video_id=video_id,
            video_path=str(path),
        )

        if caption_path:
            text = caption_path.read_text()
            results.append({
                "video_id": video_id,
                "text": text,
            })

    return {
        "creator_id": creator_id,
        "nlp_inputs": results,
    }
