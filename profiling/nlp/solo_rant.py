# profiling/nlp/solo_rant.py

from typing import Dict
from utils.trace import trace


@trace
def compute_solo_rant_signal(
    dialogue_signals: Dict[str, float],
    speaker_signals: Dict[str, float],
) -> Dict[str, float]:
    """
    Detect whether content is a solo rant (monologic delivery).

    Uses only structural signals — no semantic understanding.
    """

    if not dialogue_signals or not speaker_signals:
        return {
            "is_solo_rant": False,
            "confidence": 0.0,
            "reason": "insufficient signals",
        }

    score = 0.0
    reasons = []

    # --- Speaker dominance ---
    if speaker_signals.get("dominant_speaker_ratio", 0) >= 0.75:
        score += 0.3
        reasons.append("dominant speaker")

    # --- Low interaction ---
    if speaker_signals.get("back_and_forth_ratio", 1.0) <= 0.3:
        score += 0.25
        reasons.append("low back-and-forth")

    # --- Long turns ---
    if speaker_signals.get("avg_turn_length", 0) >= 8:
        score += 0.2
        reasons.append("long turns")

    # --- Not dialogue-like ---
    if not dialogue_signals.get("is_dialogue", False):
        score += 0.15
        reasons.append("non-dialogue structure")

    # --- Few short reactions ---
    if dialogue_signals.get("short_turn_ratio", 1.0) <= 0.3:
        score += 0.1
        reasons.append("few short turns")

    score = round(min(score, 1.0), 3)

    return {
        "is_solo_rant": score >= 0.6,
        "confidence": score,
        "reasons": reasons,
    }
