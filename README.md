# TikTokResonance

A creator profiling and fit‑scoring pipeline. It ingests creator videos, extracts CV + NLP signals, builds embeddings, and produces a resonance score for any test video or idea.

---

## Highlights

- Multi‑modal pipeline (CV, OCR, audio, NLP)
- Creator profiles with structured signals and evidence
- Resonance scoring with explainable drivers
- Human‑in‑the‑loop labeling for a format classifier
- Creator Strategy Workspace with creator selection, coverage confidence, pasted-idea review, and review decisions

---

## Quick Start

### 1) Configure creators
Edit `config.yaml`:

```yaml
active_creator: expoparker
training_creators:
  - expoparker
  - cleoabram
test_creator: cleoabram
defaults:
  model_name: BAAI/bge-large-en-v1.5
  caption_limit: 30
  scan_limit: 100
  selection_mode: both
  selection_percentile: 0.2
  selection_metric: views
```

### 2) Run the pipeline
```bash
make run
```

### 3) Label videos (optional)
```bash
make labels
make ui
```

### 4) Train the classifier
```bash
make train
```

### 5) Run resonance + dashboard
```bash
make random
make resonance
make dashboard
```

---

## Makefile Shortcuts

```
make env            # open a shell with conda env activated
make run            # full pipeline (profiles + embeddings)
make run-skip       # skip profiling + embeddings
make labels         # generate label queue for all creators
make ui             # launch label UI
make train          # train format classifier
make random         # pick a random test video
make resonance      # run resonance + write cache
make dashboard      # live dashboard (recompute)
make dashboard-demo # dashboard from cached results
make demo-reset     # reset only local review decisions
```

---

## Data Layout

```
data/
  raw_videos/
  raw_captions/
  raw_visual/
  raw_data/
  embeddings_store/
  labels/
  reviews/          # local approve/revise/reject decision trail
  drafts/
  demo/             # cached resonance output
```

`make demo-reset` truncates only `data/reviews/resonance_decisions.jsonl`. It does not touch training data, labels, raw visuals, captions, embeddings, or the demo cache.

---

## Resonance Scoring

Resonance combines:
- Semantic alignment (idea ↔ creator embeddings)
- Format alignment (idea format ↔ creator dominant formats)
- Visual alignment (motion, text density)
- A semantic gate to reduce false positives

Key outputs:
```
semantic_alignment
format_alignment
motion_alignment
text_density_alignment
resonance_score
```

---

## Notes

- CV signals are cached in `data/raw_visual` for fast iteration.
- For full‑quality CV runs:
  ```
  FAST_VISUAL=false FORCE_PROFILES=true
  ```

---

## Status

- End‑to‑end pipeline working
- Multi‑creator ingestion and profiling
- Dashboard with creator library, coverage confidence, pasted-idea evidence, and human review decisions

Interview demo notes live in `docs/interview_demo.md`.
Product framing lives in `docs/product_brief.md`.
