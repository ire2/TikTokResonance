from __future__ import annotations

import argparse
import json
from pathlib import Path

from resonance.artifact_manifest import DEFAULT_MANIFEST_PATH, write_artifact_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build derived media artifact manifest.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args()

    manifest = write_artifact_manifest(Path(args.data_dir), Path(args.output))
    print(json.dumps({
        "path": args.output,
        "videos": manifest["summary"]["videos"],
        "raw_media_present": manifest["summary"]["raw_media_present"],
        "captions_present": manifest["summary"]["captions_present"],
        "visual_artifacts_present": manifest["summary"]["visual_artifacts_present"],
        "human_labels_present": manifest["summary"]["human_labels_present"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
