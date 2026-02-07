from pathlib import Path

import numpy as np

from utils.trace import trace


@trace
def compute_audio_metrics(
    video_path: str,
    target_sr: int = 16000,
    duration_limit: float | None = 60.0,
) -> dict:
    """
    Compute audio energy and music likelihood using librosa.
    Returns zeros if audio can't be loaded.
    """

    try:
        import librosa
    except Exception:
        return {
            "audio_energy": 0.0,
            "music_confidence": 0.0,
        }

    if not Path(video_path).exists():
        return {
            "audio_energy": 0.0,
            "music_confidence": 0.0,
        }

    try:
        y, sr = librosa.load(
            video_path,
            sr=target_sr,
            mono=True,
            duration=duration_limit,
        )
    except Exception:
        return {
            "audio_energy": 0.0,
            "music_confidence": 0.0,
        }

    if y is None or len(y) == 0:
        return {
            "audio_energy": 0.0,
            "music_confidence": 0.0,
        }

    rms = librosa.feature.rms(y=y).mean()
    audio_energy = float(np.tanh(rms * 10.0))

    # crude music proxy: spectral flatness + centroid
    flatness = float(librosa.feature.spectral_flatness(y=y).mean())
    centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    centroid_norm = min(1.0, centroid / 4000.0)
    music_confidence = float(
        np.clip(0.6 * centroid_norm + 0.4 * flatness, 0.0, 1.0)
    )

    return {
        "audio_energy": round(audio_energy, 3),
        "music_confidence": round(music_confidence, 3),
    }
