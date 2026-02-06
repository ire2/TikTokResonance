# profiling/nlp/dialogue_chunker.py

from typing import Dict, List
from utils.trace import trace


@trace
def chunk_dialogue(
    transcript: Dict,
    merge_gap_sec: float = 1.5,
    break_gap_sec: float = 2.5,
    min_chars: int = 80,
) -> Dict:
    """
    Convert transcript segments into semantic dialogue beats.

    - merge_gap_sec: silence allowed within a beat
    - break_gap_sec: silence that strongly implies topic shift
    - min_chars: force-merge short beats to preserve meaning
    """

    segments = transcript.get("segments", [])
    if not segments:
        return {**transcript, "dialogue": []}

    dialogue: List[Dict] = []

    current = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": segments[0]["text"],
        "pause_before": 0.0,
        "duration": segments[0]["end"] - segments[0]["start"]
    }

    for prev, seg in zip(segments, segments[1:]):
        gap = seg["start"] - prev["end"]

        should_merge = (
            gap <= merge_gap_sec
            or len(current["text"]) < min_chars
        )

        if should_merge:
            current["end"] = seg["end"]
            current["text"] += " " + seg["text"]
        else:
            dialogue.append(current)

            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "pause_before": gap,
            }

    dialogue.append(current)

    return {
        **transcript,
        "dialogue": dialogue,
    }
