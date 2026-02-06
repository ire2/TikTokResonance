# profiling/nlp/transcript_loader.py

from pathlib import Path
from typing import Dict, List
import json
import re

from utils.trace import trace


@trace
def normalize_text(text: str) -> str:
    """
    Light normalization only.
    DO NOT add NLP logic here.
    """
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@trace
def load_transcript(caption_path: Path) -> Dict:
    """
    Load captions (Whisper or platform) into a canonical transcript format.

    Output:
    {
      video_id: str,
      segments: [
        { start: float, end: float, text: str }
      ]
    }
    """

    raw = json.loads(caption_path.read_text())

    segments: List[Dict] = []

    for seg in raw.get("segments", []):
        text = normalize_text(seg.get("text", ""))
        if not text:
            continue

        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": text,
        })

    return {
        "video_id": caption_path.stem,
        "segments": segments,
    }
