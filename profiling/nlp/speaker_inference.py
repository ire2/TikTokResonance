# profiling/nlp/speaker_inference.py

from typing import List, Dict

TURN_GAP_SEC = 0.9
SHORT_REPLY_WORDS = 4


def infer_speakers(dialogue: List[Dict]) -> List[Dict]:
    """
    Assign inferred speaker IDs to dialogue turns.

    Adds:
        speaker_id: S0, S1, ...
    """

    if not dialogue:
        return dialogue

    speaker_idx = 0
    last_end = dialogue[0]["end"]
    dialogue[0]["speaker_id"] = f"S{speaker_idx}"

    for i in range(1, len(dialogue)):
        curr = dialogue[i]
        prev = dialogue[i - 1]

        pause = curr["start"] - last_end
        curr_words = len(curr["text"].split())
        prev_words = len(prev["text"].split())

        speaker_switch = False

        if pause > TURN_GAP_SEC:
            speaker_switch = True
        elif curr_words <= SHORT_REPLY_WORDS and prev_words > 6:
            speaker_switch = True
        elif prev_words <= SHORT_REPLY_WORDS and curr_words > 6:
            speaker_switch = True

        if speaker_switch:
            speaker_idx += 1

        curr["speaker_id"] = f"S{speaker_idx}"
        last_end = curr["end"]

    return dialogue
