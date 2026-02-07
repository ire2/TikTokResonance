from utils.trace import trace
from .format_classifier import FormatClassifier
from pathlib import Path
from profiling.cv.visual_signals import extract_visual_signals
from profiling.cv.learned_classifier import LearnedFormatClassifier


class SimpleFormatClassifier(FormatClassifier):
    """
    LEGACY FORMAT CLASSIFIER (metadata-only)

    Used only when CV signals are unavailable.
    DO NOT extend this class further.
    """

    @trace
    def classify(self, video):
        learned = LearnedFormatClassifier()
        if learned.is_ready():
            pred = learned.classify(video)
            if pred:
                return pred

        # Presence of audio
        has_audio = video.get("acodec") not in (None, "none")

        # Text proxy (description + subtitles)
        has_text = bool(video.get("description")) or bool(
            video.get("subtitles"))

        # Duration proxy
        duration = video.get("duration", 0)

        # Optional: fast visual signals (no OCR/objects/audio)
        signals = None
        local_path = video.get("local_path")
        if local_path and Path(local_path).exists():
            try:
                signals = extract_visual_signals(
                    str(local_path),
                    use_ocr=False,
                    use_objects=False,
                    use_audio=False,
                )
            except Exception:
                signals = None

        if signals:
            talking_head = (
                (signals.get("talking_head_confidence") or 0.0) > 0.6
                and (signals.get("avg_face_area_ratio") or 0.0) > 0.08
                and (signals.get("shot_change_rate") or 0.0) < 0.2
            )
            text_heavy = (
                (signals.get("text_density_heuristic") or 0.0) > 0.12
            )
            dance = (
                (signals.get("motion_intensity") or 0.0) > 0.35
                and (signals.get("camera_motion_intensity") or 0.0) > 0.35
                and (signals.get("talking_head_confidence") or 0.0) < 0.4
            )
            broll = (
                (signals.get("shot_change_rate") or 0.0) > 0.35
                and (signals.get("talking_head_confidence") or 0.0) < 0.4
            )

            if text_heavy:
                return "text_heavy"
            if talking_head:
                return "talking_head"
            if dance:
                return "dance"
            if broll:
                return "broll"
            if has_audio:
                return "voiceover"
            return "mixed"

        if has_audio and has_text and duration < 60:
            return "talking_head"

        if has_audio and not has_text:
            return "voiceover"
        if not has_audio and has_text:
            return "text_heavy"

        if duration > 60:
            return "broll"

        return "unknown"
