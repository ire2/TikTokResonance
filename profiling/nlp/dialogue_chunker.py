# profiling/nlp/dialogue_chunker.py

from typing import Dict, List
from utils.trace import trace


@trace
def chunk_dialogue(
    transcript: Dict,
    merge_gap_sec: float = 0.8,
    break_gap_sec: float = 1.2,
) -> Dict:
    """
    Convert transcript segments into dialogue units.

    - merge_gap_sec: max silence to merge lines
    - break_gap_sec: silence that implies a new beat
    """

    segments = transcript["segments"]
    if not segments:
        return {**transcript, "dialogue": []}

    dialogue: List[Dict] = []

    current = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": segments[0]["text"],
        "pause_before": 0.0,
    }

    for prev, seg in zip(segments, segments[1:]):
        gap = seg["start"] - prev["end"]

        if gap <= merge_gap_sec:
            # same beat
            current["end"] = seg["end"]
            current["text"] += " " + seg["text"]

        else:
            # close current beat
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
