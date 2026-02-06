from utils.trace import trace
from .format_classifier import FormatClassifier


class SimpleFormatClassifier(FormatClassifier):
    """
    LEGACY FORMAT CLASSIFIER (metadata-only)

    Used only when CV signals are unavailable.
    DO NOT extend this class further.
    """

    @trace
    def classify(self, video):
        # Presence of audio
        has_audio = video.get("acodec") not in (None, "none")

        # Text proxy (description + subtitles)
        has_text = bool(video.get("description")) or bool(
            video.get("subtitles"))

        # Duration proxy
        duration = video.get("duration", 0)

        if has_audio and has_text and duration < 60:
            return "talking_head"

        if has_audio and not has_text:
            return "voiceover"
        if not has_audio and has_text:
            return "text_heavy"

        if duration > 60:
            return "broll"

        return "unknown"
