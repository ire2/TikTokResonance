# TikTokResonance

TikTokResonance is a local Creator Strategy Workspace for reviewing whether a short-form video idea or uploaded clip fits a creator's observed style.

It does not predict virality, auto-approve content, or scrape TikTok during the interview demo. The demo path uses local creator artifacts, evidence panels, and an explicit human approve/revise/reject decision.

---

## Current Demo

The interview-safe demo is a FastAPI dashboard that supports:

- creator selection from a curated local cohort
- coverage confidence for each creator
- pasted-idea review against cached creator segment evidence
- local video upload review using the video's extracted transcript
- evidence buckets for top matches, hit-like examples, and miss-like risks
- presentation-safe masking for explicit evidence text
- append-only human decisions in `data/reviews/resonance_decisions.jsonl`

Recommended demo creator:

```text
gray.davis
```

Recommended local demo video:

```text
data/test/video/gray.davis_7639092976248999181.mp4
```

That clip has a clean Gray Davis garden/fruit transcript and is a better demo candidate than the older placeholder test clip.

Live download/upload shortcut:

```bash
upload gray.davis 7639092976248999181
```

This downloads the video into `data/test/video/` and, if the dashboard is running, sends it through the same upload/transcript/scoring endpoint. You can also pass a quoted TikTok URL, but the numeric video id is easiest during a live demo.

If you only have the URL, this also works:

```bash
upload 'https://www.tiktok.com/@gray.davis/video/7639092976248999181'
```

---

## Quick Start

Activate the project environment:

```bash
source scripts/activate_env.sh
```

Reset only the local review trail:

```bash
make demo-reset
```

Run the deterministic dashboard:

```bash
make dashboard-demo
```

Open:

```text
http://127.0.0.1:8000
```

The prompt should show `(resonance)`. A harmless zsh/VS Code warning such as `RPROMPT: parameter not set` does not mean activation failed.

---

## Demo Script

1. Select `gray.davis`.
2. Confirm high coverage: captions, visual signals, human labels, and embeddings are present.
3. Upload `data/test/video/gray.davis_7639092976248999181.mp4`.
4. Click `Analyze Uploaded Video`.
5. Explain that the uploaded video is transcribed locally and compared against cached creator evidence.
6. Review top evidence plus hit-like and miss-risk examples.
7. Choose `Revise` or `Approve`, add a note, and save the decision.
8. Show Decision History and mention the append-only JSONL artifact.

If the upload path is slow because Whisper is extracting captions, use the pasted-idea fallback:

```text
A practical garden walkthrough explaining a rare tropical fruit, how to tell when it is ripe, what it tastes like, and one tip for growing it at home.
```

Optional upstream loop to show if asked where the evidence comes from:

```bash
make labels
make ui PORT=8001
```

Then open:

```text
http://127.0.0.1:8001/?creator=washingtonpost
```

This opens the human labeling UI for `data/labels/format_labels.csv`, filtered to the small Washington Post sample. A reviewer labels each creator video with a format label and a performance label (`hit`, `ok`, or `miss`). Those labels are what power the dashboard's hit-like and miss-risk evidence buckets.

The label UI and dashboard can run on different ports. For example: dashboard on `8000`, label UI on `8001`.

---

## Score Calibration

The raw transcript/idea match signal is intentionally conservative. Lexical segment overlap often produces values around `0.20` to `0.35` even when the retrieved evidence is clearly relevant.

For the dashboard, the user-facing `resonance_score` is a calibrated creator-local fit score. The raw signal is still exposed as `raw_match_score` inside the `resonance` payload.

Use this framing in the demo:

```text
The displayed fit score is calibrated from a compressed raw match signal. The number helps triage the idea, but the evidence panel is the source of truth: which prior creator examples did this resemble, and were those examples hits or misses?
```

Do not frame calibration as "tuning the score higher." Frame it as making a raw technical signal interpretable for a human review workflow.

Current limits:

- uploaded videos are scored primarily from extracted transcript text
- visual artifacts support creator coverage and the broader pipeline, but the live upload review is transcript-first
- the score is creator-local, not a global TikTok quality score
- hit/miss labels are evidence labels, not outcome predictions

---

## Makefile Shortcuts

```bash
make env            # open a shell with conda env activated
make run            # full pipeline: profiles + embeddings
make run-skip       # skip profiling + embeddings
make labels         # generate label queue for creators
make ui             # launch label UI
make train          # train format classifier
make random         # pick a random test video
make resonance      # run resonance + write demo cache
make dashboard      # live dashboard recompute path
make dashboard-demo # deterministic dashboard from local demo settings
make demo-reset     # truncate only local review decisions
```

---

## Data Layout

```text
data/
  raw_videos/        # downloaded/raw source videos when available
  raw_captions/      # platform or Whisper transcript JSON
  raw_visual/        # cached visual signals
  raw_data/          # normalized creator metadata
  embeddings_store/  # creator and segment embeddings/text
  labels/            # human format/performance labels
  reviews/           # approve/revise/reject decision trail
  drafts/            # creator profiles
  demo/              # cached resonance output
  test/video/        # local test/upload demo videos
```

`make demo-reset` truncates only:

```text
data/reviews/resonance_decisions.jsonl
```

It does not touch training data, labels, raw visuals, captions, embeddings, uploaded videos, or the demo cache.

---

## What The System Computes

For the local demo, TikTokResonance computes:

- creator coverage and confidence from local artifacts
- transcript or pasted-idea fit against cached creator segment evidence
- human-label evidence buckets: hit-like, miss-like, and top matches
- revision suggestions from low fit, format mismatch, miss-like evidence, or thin coverage
- a saved decision artifact with reviewer id, timestamp, score, notes, and evidence ids

For the broader pipeline, the project also includes:

- creator ingestion and profiling
- CV/OCR/audio/NLP feature extraction
- creator embeddings and segment memory
- resonance scoring for local test videos
- label UI and classifier training

---

## Human Labeling Loop

The upstream workflow is:

1. Add or edit creators in `config.yaml`.
2. Run ingestion/profiling to fetch creator metadata, selected videos, captions, and visual artifacts.
3. Generate the label queue.
4. Open the label UI.
5. Human reviewers assign:
   - `format_label`: what kind of content it is
   - `performance_label`: `hit`, `ok`, or `miss`
6. The dashboard uses those labels as evidence, not as automatic truth.

Commands:

```bash
make run      # fetch/profile configured training creators
make labels   # generate or refresh data/labels/format_labels.csv
make ui PORT=8001 # open the labeling UI at http://127.0.0.1:8001
make train    # train the format classifier from human labels
```

For an interview, avoid live-ingesting a brand-new creator unless specifically requested. Network downloads can be slow or blocked. The safer demo is to show the existing label UI and explain that this is how new creator evidence enters the system.

---

## Verification

Use the focused interview-safe test slice:

```bash
python -m compileall -q pipeline profiling resonance utils scripts
python -m pytest \
  profiling/ingestion/test_select_videos.py \
  profiling/label_ui/test_label_ui_app.py \
  resonance/test_creator_library.py \
  resonance/test_idea_review.py \
  resonance/test_resonance_report.py \
  resonance/test_resonance_score.py \
  resonance/test_review_decisions.py \
  resonance/dashboard/test_dashboard_app.py \
  -q
```

Avoid network-dependent smoke tests during the interview.

---

## Project Status

Working now:

- local Creator Strategy Workspace
- curated creator cohort
- Gray Davis upload demo path
- transcript-based uploaded video review
- pasted-idea fallback review
- calibrated dashboard fit score with raw match score retained
- presentation-safe evidence masking
- append-only local review decisions

Known limits:

- not a production multi-user app
- no auth or role-based review queue
- uploaded-video review is transcript-first, not full live multimodal scoring
- calibration is product-level interpretability, not post-publication outcome calibration

Interview details live in `docs/interview_demo.md`.
Product framing lives in `docs/product_brief.md`.
