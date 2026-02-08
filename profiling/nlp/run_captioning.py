from pathlib import Path
from utils.trace import trace

from profiling.ingestion.fetch_raw import fetch_captions
from profiling.nlp.asr import run_asr  # your ASR file
from profiling.utils.video_paths import video_path


RAW_CAPTION_DIR = Path("data/raw_captions")
RAW_CAPTION_DIR.mkdir(parents=True, exist_ok=True)


@trace
def ensure_captions(
    creator_id: str,
    video_id: str,
):
    """
    Ensure captions exist for a single video.
    """

    vtt = RAW_CAPTION_DIR / f"{creator_id}_{video_id}.en.vtt"
    asr = RAW_CAPTION_DIR / f"{creator_id}_{video_id}.asr.json"

    # 1. Platform captions already exist
    if vtt.exists():
        return {"source": "platform", "path": vtt}

    # 2. Try fetching captions (best effort)
    fetch_captions(creator_id, video_limit=1)

    if vtt.exists():
        return {"source": "platform", "path": vtt}

    # 3. ASR fallback
    if asr.exists():
        return {"source": "asr", "path": asr}

    video = video_path(creator_id, video_id)
    run_asr(video, output_path=asr)

    return {"source": "asr", "path": asr}
