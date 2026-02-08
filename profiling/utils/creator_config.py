from pathlib import Path
import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BASE_DIR.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
FALLBACK_CONFIG_PATH = BASE_DIR / "resonance" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text()) or {}
    if FALLBACK_CONFIG_PATH.exists():
        return yaml.safe_load(FALLBACK_CONFIG_PATH.read_text()) or {}
    raise FileNotFoundError(
        f"Config not found: {path} (also checked {FALLBACK_CONFIG_PATH})"
    )


def get_active_creator(path: Path = CONFIG_PATH) -> str:
    config = load_config(path)
    creator_id = config.get("active_creator")
    if not creator_id:
        raise ValueError("active_creator missing in profiling/config.yaml")
    return creator_id


def get_training_creators(path: Path = CONFIG_PATH) -> list[str]:
    config = load_config(path)
    creators = config.get("training_creators")
    if not creators:
        return []
    if isinstance(creators, list):
        return [str(c) for c in creators]
    return [str(creators)]


def get_test_creator(path: Path = CONFIG_PATH) -> list[str]:
    config = load_config(path)
    test_creators = config.get("test_creator")
    if not test_creators:
        raise ValueError("test_creators missing in profiling/config.yaml")
    return test_creators


def get_default_model_name(path: Path = CONFIG_PATH) -> str:
    config = load_config(path)
    defaults = config.get("defaults", {})
    model_name = defaults.get("model_name")
    if not model_name:
        raise ValueError(
            "defaults.model_name missing in profiling/config.yaml")
    return model_name


def get_default_caption_limit(path: Path = CONFIG_PATH) -> int:
    config = load_config(path)
    defaults = config.get("defaults", {})
    limit = defaults.get("caption_limit")
    if limit is None:
        raise ValueError(
            "defaults.caption_limit missing in profiling/config.yaml")
    return int(limit)


def get_default_scan_limit(path: Path = CONFIG_PATH) -> int:
    config = load_config(path)
    defaults = config.get("defaults", {})
    limit = defaults.get("scan_limit")
    if limit is None:
        return 0
    return int(limit)


def get_selection_mode(path: Path = CONFIG_PATH) -> str | None:
    config = load_config(path)
    defaults = config.get("defaults", {})
    return defaults.get("selection_mode")


def get_selection_percentile(path: Path = CONFIG_PATH) -> float | None:
    config = load_config(path)
    defaults = config.get("defaults", {})
    val = defaults.get("selection_percentile")
    if val is None:
        return None
    return float(val)


def get_selection_metric(path: Path = CONFIG_PATH) -> str | None:
    config = load_config(path)
    defaults = config.get("defaults", {})
    return defaults.get("selection_metric")
