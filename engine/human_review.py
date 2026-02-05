import yaml
import subprocess
from pathlib import Path
from datetime import datetime


def open_file(path: Path):
    """
    Open file in default editor (macOS-friendly for OA).
    """
    try:
        subprocess.call(["open", path])
    except Exception:
        print(f"Please open and edit manually: {path}")


def human_review_checkpoint(draft_path: Path, reviewed_path: Path):
    """
    Blocking human review step.
    Pipeline pauses here until review is resolved.
    """

    with open(draft_path) as f:
        draft = yaml.safe_load(f)

    print("\n" + "─" * 40)
    print("CREATOR PROFILE — DRAFT\n")
    print(f"Creator: {draft.get('creator_id')}")
    print(f"Status: DRAFT\n")
    print("Summary:")
    print(draft.get("creative_interpretation", {}).get("summary", "—"))
    print("─" * 40)

    choice = input(
        "\nApprove profile for downstream use?\n"
        "[a] Approve\n"
        "[f] Flag for edit\n"
        "Choice: "
    ).strip().lower()

    if choice == "a":
        status = "approved"

    elif choice == "f":
        print("\nOpening draft for editing...\n")
        open_file(draft_path)

        input(
            "Make edits in the opened file.\n"
            "When finished, return here and press ENTER to continue."
        )

        with open(draft_path) as f:
            draft = yaml.safe_load(f)

        final_choice = input(
            "\nApprove edited profile?\n"
            "[a] Approve\n"
            "[r] Reject\n"
            "Choice: "
        ).strip().lower()

        if final_choice == "a":
            status = "approved"
        else:
            status = "rejected"

    else:
        print("Invalid choice. Review aborted.")
        return

    reviewed = {
        "creator_id": draft.get("creator_id"),
        "review_status": status,
        "reviewed_at": datetime.utcnow().isoformat(),
        "reviewer": "human",
        "profile": draft,
    }

    reviewed_path.parent.mkdir(exist_ok=True)
    with open(reviewed_path, "w") as f:
        yaml.dump(reviewed, f, sort_keys=False)

    print(f"\nProfile {status.upper()}. Pipeline may continue.\n")

    if status != "approved":
        raise RuntimeError("Pipeline stopped: profile not approved.")
