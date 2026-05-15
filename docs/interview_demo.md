# TikTokResonance Interview Demo

## 60-Second Pitch

TikTokResonance is a human-in-the-loop Creator Strategy Workspace with Market Memory. A strategist can see the local creator cohort, understand data coverage, choose a creator, paste an idea, inspect evidence, and save an approve/revise/reject decision.

“I started treating short-form strategy like a quant research problem: creators are instruments, videos are observations, and the useful product is not a magic virality predictor, it is an evidence system for human strategy decisions.”

The product does not auto-approve ideas, scrape live TikTok during the demo, call an LLM, or claim global TikTok coverage.

## Implemented Demo Workflow

1. Read the Market Map.
2. Choose a creator from the curated demo cohort.
3. Read the coverage panel: videos analyzed, captions, visual signals, labels, hit/ok/miss counts, embeddings, and confidence level.
4. Paste or edit an idea.
5. Click Analyze.
6. Review local fit evidence, hit-like examples, miss-like risks, and revision suggestions.
7. Choose Approve, Revise, or Reject.
8. Add notes and save the decision.
9. Confirm the decision appears in Decision History and is appended to `data/reviews/resonance_decisions.jsonl`.

## Market Map

The dashboard starts with Market Memory, a compact cohort view built from local artifacts. It shows:

- total creators and videos
- coverage tier counts
- creators ready for idea review
- top formats with observed hit rates
- recommended next processing queue

This is local market memory, not a claim about all of TikTok. It answers: “What have we studied, how deeply, and what should we process next?”

## Coverage Tiers

- `metadata_only`: metadata exists, so the creator/video can be tracked and roughly ranked.
- `semantic_ready`: captions or segment memory exist, so idea/text review can work.
- `deep_style_ready`: semantic artifacts, visual artifacts, and labels exist, so richer style evidence is available.

Confidence is shown separately. A creator can be semantic-ready but low confidence if evidence is sparse.

## Creator-Local Hit / Ok / Miss Normalization

Human labels are preserved. When labels are absent or incomplete, TikTokResonance can infer hit / ok / miss from the creator’s own performance distribution:

- top 20 percent by creator-local views: hit
- middle 60 percent: ok
- bottom 20 percent: miss

Raw views are not compared globally across creators. This avoids treating a small creator’s breakout hit as worse than a large creator’s average post.

## Creator Library

The creator library is built from local artifacts only:

- `data/drafts/*_draft.yaml`
- `data/raw_visual/*.json`
- `data/raw_captions/*.json`
- `data/labels/format_labels.csv`
- `data/embeddings_store/*`

For each creator, the dashboard shows coverage, confidence, dominant formats, and hit / ok / miss counts.

## Paste-Idea Flow

`POST /api/idea-review` accepts:

```json
{
  "creator_id": "expoparker",
  "idea_text": "A short creator-specific idea"
}
```

The demo-safe path uses local segment memory and human labels. It returns a fit score, top evidence, prior-hit evidence, prior-miss risk evidence, suggestions, coverage, and a clear analysis note. If segment memory is unavailable, fallback behavior is labeled instead of hidden.

## Decision History

`POST /api/review-decision` saves an append-only JSONL record with:

- decision
- creator
- score
- idea text and snippet
- notes
- timestamp
- reviewer id, defaulting to `creator_strategist`
- evidence video ids
- analysis mode

`GET /api/review-decisions` returns the latest local decisions for the dashboard.

## Artifact-First Scale Path

The Colab/GPU artifact factory is the path to broader market coverage:

1. Clone or upload the repo on Colab, RunPod, or Modal.
2. Install dependencies.
3. Mount Drive or upload a batch config.
4. Process creators in tiers:
   - metadata breadth
   - captions/embeddings
   - deep visual/CV for selected creators
5. Build `data/artifacts/market_index.json`.
6. Build `data/artifacts/media_manifest.json`.
7. Export a derived artifact pack.
8. Delete raw media on the worker.
9. Import the artifact pack locally and demo without raw videos.

See `notebooks/colab_artifact_factory.md` and `configs/market_batch.example.yaml`.

## Demo Reset

Run:

```bash
make demo-reset
```

This truncates only:

```text
data/reviews/resonance_decisions.jsonl
```

It does not delete or modify training data, labels, raw visual data, captions, embeddings, demo cache, manifest, or market index.

## Demo Commands

Use the project environment:

```bash
source scripts/activate_env.sh
```

Build local market artifacts:

```bash
make market-index
make artifact-manifest
```

Reset the local decision trail:

```bash
make demo-reset
```

Run the interview-safe dashboard:

```bash
make dashboard-demo
```

Open:

```text
http://127.0.0.1:8000
```

Safe verification:

```bash
python -m compileall -q pipeline profiling resonance utils
python -m pytest \
  profiling/ingestion/test_select_videos.py \
  profiling/label_ui/test_label_ui_app.py \
  resonance/test_artifact_manifest.py \
  resonance/test_creator_library.py \
  resonance/test_idea_review.py \
  resonance/test_market_cohort.py \
  resonance/test_resonance_report.py \
  resonance/test_resonance_score.py \
  resonance/test_review_decisions.py \
  resonance/dashboard/test_dashboard_app.py \
  -q
```

Avoid network-dependent smoke tests during the interview:

```text
profiling/ingestion/test_fetch.py
profiling/dev/test_fetch_captions.py
```

## Exact Demo Sequence

1. Run `make market-index && make artifact-manifest`.
2. Run `make demo-reset`.
3. Run `make dashboard-demo`.
4. Point to Market Memory and say it is local market memory, not a global TikTok claim.
5. Explain coverage tiers and the recommended processing queue.
6. Select `expoparker`, then `cleoabram`, to show confidence and coverage change by creator.
7. Paste or keep the sample idea and click Analyze.
8. Explain the score as evidence-based fit, not virality prediction.
9. Open Top Evidence and the hit/miss evidence buckets.
10. Choose `Revise`, add a note like `The topic fits, but the hook needs a clearer creator-native setup.`
11. Save the decision.
12. Show Decision History and mention the append-only JSONL artifact.
13. If asked about scale, open `notebooks/colab_artifact_factory.md`.

## Files To Keep Open

- `resonance/dashboard/app.py`: FastAPI endpoints and dashboard UI.
- `resonance/market_cohort.py`: Market Memory, creator-local labels, summaries.
- `resonance/artifact_manifest.py`: per-video artifact manifest.
- `resonance/creator_library.py`: creator coverage and confidence.
- `resonance/idea_review.py`: pasted-idea local evidence retrieval.
- `resonance/review_decisions.py`: append-only decision artifact and reset helper.
- `scripts/export_artifact_pack.py`: derived-only artifact export.
- `scripts/import_artifact_pack.py`: safe artifact import.
- `scripts/clean_media.py`: raw media cleanup.

## Known Limits

- The dashboard is still a compact FastAPI-rendered page, not a full React app.
- Pasted-idea analysis uses lexical overlap over cached segment text rather than full semantic recomputation in demo mode.
- Hit-vs-miss evidence is based on retrieved local examples and creator-local labels, not outcome prediction.
- The batch config is a schema/checklist today; the full cloud batch processor is future work.
- Persistence is JSONL and JSON artifacts, not a multi-user database.

## Future Improvements

- Implement a full batch runner for `configs/market_batch.example.yaml`.
- Add semantic retrieval for pasted ideas using existing embeddings once startup is demo-safe.
- Add artifact pack checksums and schema validation.
- Move labels and decisions into SQLite or Postgres with migrations.
- Calibrate confidence against real post-publication outcomes.

## Interview Framing

The upgrade makes the project more product-like because it no longer feels like a single score page over a tiny memory. It now shows the market coverage layer, the creator-specific evidence layer, the human decision loop, and the cloud artifact workflow needed to scale responsibly.
