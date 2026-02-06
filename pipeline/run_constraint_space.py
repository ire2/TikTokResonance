from pathlib import Path
import yaml
import shutil
import os

from resonance.select_ideas import select_ideas
from pipeline.human_review import human_review_checkpoint
from profiling.run_pipeline import run_profiling_for_creator
from resonance.build_constraints import build_constraints
from resonance.evaluate_idea import evaluate_idea_against_constraints

CLEAN_RUN = os.getenv("CLEAN_RUN", "false").lower() == "true"
BASE_DIR = Path(__file__).resolve().parent.parent

CREATORS_PATH = BASE_DIR / "profiling" / "input" / "creators.yaml"
DRAFTS_DIR = BASE_DIR / "profiling" / "drafts"
REVIEWED_DIR = BASE_DIR / "profiling" / "reviewed"
IDEAS_DIR = BASE_DIR / "constraint_space" / "ideas"


def clear_pycache(base_dir: Path):
    """
    Development-only helper to ensure clean imports during iteration.
    """
    for pycache in base_dir.rglob("__pycache__"):
        shutil.rmtree(pycache)


def run_constraint_space():
    if CLEAN_RUN:
        clear_pycache(BASE_DIR)
    with open(CREATORS_PATH, "r") as f:
        config = yaml.safe_load(f)

    for creator in config["creators"]:
        creator_id = creator["id"]
        video_limit = creator.get("video_limit", 30)

        draft_path = DRAFTS_DIR / f"{creator_id}_draft.yaml"
        reviewed_path = REVIEWED_DIR / f"{creator_id}_profile.yaml"

        # 1. Ensure profiling draft exists
        if not draft_path.exists():
            run_profiling_for_creator(
                creator_id=creator_id,
                video_limit=video_limit
            )

        # 2. Ensure human-reviewed profile exists
        if not reviewed_path.exists():
            human_review_checkpoint(
                draft_path=draft_path,
                reviewed_path=reviewed_path
            )

        # 3. Load reviewed profile
        with open(reviewed_path, "r") as f:
            reviewed_profile = yaml.safe_load(f)

        # 4. Build constraint space (v0)
        constraints = build_constraints(reviewed_profile)
        print(f"\nConstraintSpace ready for creator: {creator_id}")

        print(yaml.dump(constraints, sort_keys=False))

        # 5. Evaluate ideas
        evaluations = []
        for idea in IDEAS_DIR.glob("example_idea*.yaml"):
            idea_path = idea
            if idea_path.exists():
                with open(idea_path, "r") as f:
                    idea = yaml.safe_load(f)

                result = evaluate_idea_against_constraints(
                    idea=idea,
                    constraints=constraints
                )

            #     print("\nIDEA EVALUATION")
            #     print(f"Idea: {idea.get('title')}")
            #     print(f"Decision: {result['decision'].upper()}")
            #     print(f"Score: {result['score']}")
            #     print(f"Confidence: {result['confidence_band']}")

            #     if result["reasons"]:
            #         print("Reasons:")
            #         for r in result["reasons"]:
            #             print(f"- {r}")

            #     if result["signals"]:
            #         print("Signals:")
            #         for k, v in result["signals"].items():
            #             print(f"- {k}: {round(v, 3)}")
            # else:
            #     print("\nNo ideas found for evaluation.")
                evaluations.append({
                    "idea": idea,
                    "result": result
                })
    selection = select_ideas(
        evaluations,
        constraints["experiment_policy"]
    )

    print("\nSELECTION PLAN")

    for group, items in selection.items():
        print(f"\n{group.upper()}:")
        for item in items:
            print(
                f"- {item['idea']['title']} "
                f"({item['result']['score']})"
            )


if __name__ == "__main__":
    run_constraint_space()
