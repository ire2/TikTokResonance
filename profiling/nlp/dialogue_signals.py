# profiling/nlp/speaker_signals.py

from utils.trace import trace
from collections import Counter
from typing import Dict, List
import re


# -----------------------------
# Dialogue-level signals
# -----------------------------

def _empty_dialogue():
    return {
        "num_turns": 0,
        "avg_turn_length": 0.0,
        "short_turn_ratio": 0.0,
        "question_ratio": 0.0,
        "dialogue_density": 0.0,
        "is_dialogue": False,
    }


@trace
def compute_dialogue_signals(transcript: Dict) -> Dict[str, float]:
    """
    Extract dialogue / interaction structure from ASR transcript JSON.

    Models conversational geometry only.
    """

    if not transcript or "segments" not in transcript:
        return _empty_dialogue()

    segments = transcript.get("segments", [])
    if not segments:
        return _empty_dialogue()

    # -----------------------------
    # 1. Normalize turns
    # -----------------------------
    lines = [
        s["text"].strip()
        for s in segments
        if s.get("text") and s["text"].strip()
    ]

    num_turns = len(lines)
    if num_turns == 0:
        return _empty_dialogue()

    # -----------------------------
    # 2. Turn-level stats
    # -----------------------------
    turn_lengths = [len(l.split()) for l in lines]
    avg_turn_length = sum(turn_lengths) / num_turns

    short_turn_ratio = sum(
        1 for l in turn_lengths if l <= 5
    ) / num_turns

    # -----------------------------
    # 3. Question rhythm
    # -----------------------------
    question_ratio = sum(
        1 for l in lines
        if l.endswith("?")
        or re.search(r"\b(what|why|how|who|where)\b", l.lower())
    ) / num_turns

    # -----------------------------
    # 4. Dialogue density heuristic
    # -----------------------------
    dialogue_density = (
        0.5 * short_turn_ratio
        + 0.3 * question_ratio
        + 0.2 * min(1.0, num_turns / 12)
    )

    dialogue_density = round(min(1.0, dialogue_density), 3)

    is_dialogue = (
        num_turns >= 6
        and short_turn_ratio > 0.3
        and dialogue_density > 0.45
    )

    return {
        "num_turns": num_turns,
        "avg_turn_length": round(avg_turn_length, 2),
        "short_turn_ratio": round(short_turn_ratio, 3),
        "question_ratio": round(question_ratio, 3),
        "dialogue_density": dialogue_density,
        "is_dialogue": is_dialogue,
    }


# -----------------------------
# Speaker-level signals
# -----------------------------

def compute_speaker_signals(dialogue: List[Dict]) -> Dict[str, float]:
    """
    Compute speaker interaction signals from diarized dialogue turns.
    """

    if not dialogue:
        return {}

    speakers = [d["speaker_id"] for d in dialogue]
    counts = Counter(speakers)

    num_turns = len(dialogue)
    num_speakers = len(counts)

    turn_lengths = [
        len(d["text"].split()) for d in dialogue
    ]

    dominant_ratio = max(counts.values()) / num_turns

    back_and_forth = sum(
        1
        for i in range(1, num_turns)
        if dialogue[i]["speaker_id"] != dialogue[i - 1]["speaker_id"]
    )

    return {
        "num_speakers": num_speakers,
        "turn_count": num_turns,
        "avg_turn_length": round(sum(turn_lengths) / num_turns, 2),
        "back_and_forth_ratio": round(back_and_forth / max(1, num_turns - 1), 3),
        "dominant_speaker_ratio": round(dominant_ratio, 3),
    }
