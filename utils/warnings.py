import warnings
import os


def silence_common_warnings():
    """
    Suppress noisy third-party warnings when QUIET_WARNINGS=true.
    """
    if os.getenv("QUIET_WARNINGS", "true").lower() != "true":
        return

    warnings.filterwarnings(
        "ignore",
        message=".*pin_memory.*not supported on MPS.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="PySoundFile failed.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="librosa.core.audio.__audioread_load.*",
        category=FutureWarning,
    )
