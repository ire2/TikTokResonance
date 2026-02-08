from statistics import mean
from utils.trace import trace
import os
from profiling.utils.video_paths import video_path
from profiling.nlp.asr import ensure_captions
from profiling.nlp.transcript_loader import load_transcript
from profiling.nlp.nlp_aggregate import compute_nlp_signals
from profiling.nlp.topic_profile import compute_topic_profile


@trace
def compute_nlp_profile(creator_id: str, videos: list) -> dict:
    outputs = []
    docs = []
    debug_video = os.getenv("DEBUG_VIDEO", "false").lower() == "true"
    total = len(videos)
    processed = 0

    for v in videos:
        video_id = v.get("video_id")
        if not video_id:
            continue

        path = video_path(creator_id, video_id)
        if not path.exists():
            continue

        caption = ensure_captions(
            creator_id=creator_id,
            video_id=video_id,
            video_path=str(path),
        )

        if not caption:
            continue

        transcript = load_transcript(caption)
        signals = compute_nlp_signals(transcript)

        # Build per-video document for topic modeling
        doc = " ".join(s.get("text", "") for s in transcript.get("segments", []))
        if doc:
            docs.append(doc)

        if signals:
            outputs.append(signals)
            if debug_video:
                print(f"[VIDEO][{creator_id}:{video_id}] nlp_signals=ok")
        processed += 1
        if processed % 5 == 0:
            print(f"[NLP][{creator_id}] processed {processed}/{total}")

    if not outputs:
        return {}

    def pct(key):
        return round(
            sum(1 for o in outputs if o.get(key)) / len(outputs),
            3,
        )

    topic_profile = compute_topic_profile(docs)

    return {
        "dialogue_video_ratio": pct("has_dialogue"),
        "solo_rant_ratio": round(
            mean(o["solo_rant"]["is_solo_rant"]
                 for o in outputs if "solo_rant" in o),
            3,
        ),
        "avg_speakers": round(
            mean(o["speaker_signals"]["num_speakers"]
                 for o in outputs if "speaker_signals" in o),
            2,
        ),
        "topic_profile": topic_profile,
    }
