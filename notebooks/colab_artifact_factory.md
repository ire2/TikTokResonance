# Colab Artifact Factory

This is the scale path for building TikTokResonance market memory on an online GPU machine while keeping the local demo artifact-first. The GPU worker may temporarily hold raw media, but the durable output is a derived artifact pack.

## 1. Start Runtime

Use a GPU runtime only when running CV or ASR. Metadata-only and caption/embedding tiers can run on CPU if dependencies are available.

## 2. Get The Project

Option A: clone the repo.

```bash
git clone <repo-url> TikTokResonance
cd TikTokResonance
```

Option B: upload a project zip, unzip it, and `cd` into the repo.

## 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

If a dependency is too heavy for the notebook, process only metadata/caption/embedding tiers and leave deep CV for RunPod/Modal.

## 4. Mount Drive Or Upload Config

```python
from google.colab import drive
drive.mount("/content/drive")
```

Copy or upload a batch config like:

```text
configs/market_batch.example.yaml
```

## 5. Process Creators In Tiers

Recommended tiering:

- `metadata_only`: collect or load creator/video metadata. No raw video should be committed.
- `semantic`: ensure captions/transcripts and embeddings.
- `deep_style`: run expensive CV/audio/OCR/style extraction only for selected creators.

Current repo support is artifact-first. A full batch processor is future work; today, use the config as an execution checklist and run the existing pipeline commands per creator/tier.

Example manual loop:

```bash
# Metadata breadth
python -m scripts.build_market_index

# Per selected creator, run existing project pipeline commands as needed.
# Avoid network-dependent commands unless you explicitly intend to fetch data.
make labels
make train

# Build derived indexes before export.
make market-index
make artifact-manifest
```

## 6. Export Derived Artifact Pack

```bash
python -m scripts.export_artifact_pack \
  --output data/artifact_packs/market_artifacts.zip
```

Include reviews only when you explicitly want to move local strategy decisions:

```bash
python -m scripts.export_artifact_pack \
  --include-reviews \
  --output data/artifact_packs/market_artifacts_with_reviews.zip
```

The exporter excludes raw videos such as `.mp4`, `.mov`, `.mkv`, and `.webm`.

## 7. Delete Raw Media On The Worker

Dry run first:

```bash
make clean-media-dry-run
```

Then delete only raw media files from known raw/test video folders:

```bash
make clean-media
```

This does not remove captions, labels, visual JSON, embeddings, reviews, demo cache, manifest, or market index.

## 8. Bring Artifacts Back Locally

Download `data/artifact_packs/market_artifacts.zip`, then import into the local checkout:

```bash
python -m scripts.import_artifact_pack data/artifact_packs/market_artifacts.zip --dry-run
python -m scripts.import_artifact_pack data/artifact_packs/market_artifacts.zip
make market-index
make artifact-manifest
make dashboard-demo
```

## TODOs

- Add a true batch processor that reads `configs/market_batch.example.yaml`.
- Add per-tier resume state so failed creators can be retried.
- Add cloud storage export targets for Drive, S3, or GCS.
- Add artifact pack checksums and schema version validation before import.
