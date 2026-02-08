import os


def get_preferred_device() -> str:
    """
    Returns preferred torch device: cuda, mps, or cpu.
    Environment override: CV_DEVICE
    """
    override = os.getenv("CV_DEVICE")
    if override:
        return override

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"
