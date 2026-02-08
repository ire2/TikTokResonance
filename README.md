# Resonance Lab

This repository is a **creator resonance lab** for evaluating whether an idea or draft video will fit (and potentially perform for) a specific creator.  
It combines **CV signals**, **NLP signals**, and **semantic embeddings** into a single scoring pipeline.

---

## What It Does

- Ingests creator metadata and videos
- Builds creator profiles (CV + NLP + metadata)
- Trains a format classifier from labeled videos
- Runs resonance on a **test video** or **idea text**

---

## Quick Start

### 1) Configure creators
Edit root `config.yaml`:

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

### 2) Run the pipeline (profiles + embeddings)
```bash
python -m pipeline.run_main
```

### 3) Label videos (optional, for training)
```bash
make labels
make ui
```

### 4) Train format classifier
```bash
make train
```

### 5) Run resonance on a test video
Put a test video in:
```
profiling/test/video
```
Then:
```bash
make resonance
```

---

## Makefile Shortcuts

```
make run           # full pipeline
make run-skip      # skip profiling + embeddings
make labels        # generate label queue
make ui            # launch label UI
make train         # train format classifier
make resonance     # run resonance on test video
make random        # pick random test video
```

---

## Data Layout

All runtime artifacts live in `data/`:

```
data/
  raw_videos/
  raw_captions/
  raw_visual/
  raw_data/
  embeddings_store/
  labels/
  drafts/
```

---

## Resonance Scoring

Resonance combines:
- **Semantic alignment** (idea ↔ creator embeddings)
- **Format alignment** (idea format ↔ creator dominant formats)
- **CV alignment** (motion, text density, etc.)
- **Semantic gate** (embedding similarity)

Output example:
```
semantic_alignment
format_alignment
motion_alignment
text_density_alignment
resonance_score
```

---

## Notes

- OCR/YOLO/audio are expensive. Use `FAST_VISUAL=true` for iteration.
- If you want full quality CV signals, run once with:
  ```
  FAST_VISUAL=false FORCE_PROFILES=true
  ```

---

## Status

- Working end‑to‑end pipeline
- Multi‑creator ingestion
- Video‑based resonance testing
---
