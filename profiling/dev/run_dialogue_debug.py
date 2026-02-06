# profiling/dev/run_dialogue_debug.py

from pathlib import Path


from profiling.nlp.nlp_aggregate import compute_nlp_signals
from profiling.nlp.solo_rant import compute_solo_rant_signal
from profiling.nlp.speaker_inference import infer_speakers
from profiling.nlp.transcript_loader import load_transcript
from profiling.nlp.dialogue_chunker import chunk_dialogue
from profiling.nlp.dialogue_signals import (
    compute_speaker_signals,
    compute_dialogue_signals,
)
from profiling.nlp.solo_rant import compute_solo_rant_signal

CREATOR_ID = "expoparker"
CAPTION_DIR = Path("profiling/raw_captions")
LIMIT = 3


def main():
    caption_files = sorted(
        CAPTION_DIR.glob(f"{CREATOR_ID}_*.json")
    )[:LIMIT]

    if not caption_files:
        print("[DEBUG] No caption files found.")
        return

    for cap_path in caption_files:
        print("\n" + "=" * 80)
        print(f"[VIDEO] {cap_path.stem}")
        print("=" * 80)

        transcript = load_transcript(cap_path)

        # ✅ dialogue-level signals FIRST
        dialogue_signals = compute_dialogue_signals(transcript)
        print("\n[DIALOGUE SIGNALS]")
        for k, v in dialogue_signals.items():
            print(f"  {k}: {v}")

        # Build dialogue turns
        result = chunk_dialogue(transcript)
        dialogue = result.get("dialogue", [])

        if not dialogue:
            print("\n(no dialogue detected)")
            continue

        nlp = compute_nlp_signals(transcript)

        print("\n[NLP SIGNALS]")
        for k, v in nlp.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
