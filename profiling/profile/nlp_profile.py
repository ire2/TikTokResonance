from statistics import mean
from utils.trace import trace
from profiling.utils.video_paths import video_path
from profiling.nlp.asr import ensure_captions
from profiling.nlp.transcript_loader import load_transcript
from profiling.nlp.nlp_aggregate import compute_nlp_signals


@trace
def compute_nlp_profile(creator_id: str, videos: list) -> dict:
    outputs = []

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

        if signals:
            outputs.append(signals)

    if not outputs:
        return {}

    def pct(key):
        return round(
            sum(1 for o in outputs if o.get(key)) / len(outputs),
            3,
        )

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
    }
