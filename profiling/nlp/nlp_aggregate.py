# profiling/nlp/nlp_aggregate.py

from typing import Dict
from utils.trace import trace

from profiling.nlp.dialogue_chunker import chunk_dialogue
from profiling.nlp.speaker_inference import infer_speakers

from profiling.nlp.dialogue_signals import compute_dialogue_signals, compute_speaker_signals
from profiling.nlp.solo_rant import compute_solo_rant_signal


@trace
def compute_nlp_signals(transcript: Dict) -> Dict:
    """
    Compute all NLP-derived structural signals for a single video.
    """

    # 1. Chunk dialogue
    chunked = chunk_dialogue(transcript)
    dialogue = chunked.get("dialogue", [])

    if not dialogue:
        return {
            "has_dialogue": False,
        }

    # 2. Infer speakers
    dialogue = infer_speakers(dialogue)

    # 3. Compute signals
    speaker_signals = compute_speaker_signals(dialogue)
    dialogue_signals = compute_dialogue_signals(transcript)

    solo_rant = compute_solo_rant_signal(
        dialogue_signals=dialogue_signals,
        speaker_signals=speaker_signals,
    )

    return {
        "has_dialogue": True,
        "speaker_signals": speaker_signals,
        "dialogue_signals": dialogue_signals,
        "solo_rant": solo_rant,
    }
