from typing import Dict, Any


class FormatClassifier:
    """
    Interface for classifying video format using CV / multimodal signals.

    v1: heuristic + frame sampling
    v2: learned CV model
    """

    def classify(self, video: Dict[str, Any]) -> str:
        """
        Returns one of:
        - talking_head
        - voiceover
        - broll
        - text_heavy
        - dance
        - mixed
        - unknown
        """
        raise NotImplementedError
