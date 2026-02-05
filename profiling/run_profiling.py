import yaml
from profile_generator import generate_profile, write_profile

INPUT_PATH = "profiling/input/creators.yaml"
RAW_DATA_PATH = "profiling/raw_data/creator_metadata.json"
OUTPUT_PATH = "profiling/drafts/creator_profile_draft.yaml"


def main():
    with open(INPUT_PATH, "r") as f:
        config = yaml.safe_load(f)

    # For now: process the first creator only
    creator = config["creators"][0]
    creator_id = creator["id"]

    profile = generate_profile(
        creator_id=creator_id,
        raw_data_path=RAW_DATA_PATH
    )

    write_profile(profile, OUTPUT_PATH)

    print("\nDraft creative profile generated")
    print(f"Creator: {creator_id}")
    print(f"Output: {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
