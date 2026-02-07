from pathlib import Path
import yaml

from profiling.nlp.run_nlp import run_nlp_for_creator
from profiling.utils.creator_config import get_active_creator

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = BASE_DIR / "profiling" / "metadata" / "raw_data" / "creator_metadata.json"
CREATOR_ID = get_active_creator()
PROFILE_PATH = BASE_DIR / "profiling" / "drafts" / f"{CREATOR_ID}_draft.yaml"


def main():
    profile = yaml.safe_load(PROFILE_PATH.read_text())

    creator_id = profile["creator_id"]

    result = run_nlp_for_creator(
        creator_id=creator_id,
        raw_data_path=RAW_DATA_PATH,
        profile=profile,
    )

    print("\n[NLP DEBUG RESULT]")
    print(result)


if __name__ == "__main__":
    main()
