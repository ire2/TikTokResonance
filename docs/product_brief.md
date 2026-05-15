# Creator Strategy Workspace Product Brief

## Product Thesis

TikTokResonance treats short-form video strategy like quant research:

- creators are instruments
- videos are observations
- captions, hooks, formats, motion, text density, posting time, and engagement are features
- hit / ok / miss labels are normalized within each creator
- the product supports a human strategist with evidence, not a guaranteed prediction

The useful product is not a magic virality predictor. It is market memory: a local, inspectable evidence system that helps a strategist decide what to approve, revise, or reject.

## Target User

Creator strategists, brand strategists, and creator operations teammates who review short-form video ideas before production.

## User Pain Point

Strategists make fit calls from memory, scattered links, spreadsheet labels, and subjective taste. They need a faster way to answer:

- Which creator is this idea for?
- How much evidence do we have for that creator?
- What creator-specific baseline should we compare against?
- What prior examples support the idea?
- What prior examples warn against it?
- What decision did we make and why?

## Current Alternatives

- Manually scanning TikTok profiles and saved links.
- Using spreadsheets of labels without a review surface.
- Asking a teammate who remembers the creator well.
- Using generic AI ideation tools that do not know creator-specific evidence.

## Why Creator-Local Normalization Matters

Raw views should not be compared globally across creators. A 500k-view video may be a hit for one creator and a miss for another. TikTokResonance normalizes hit / ok / miss labels within each creator using human labels first, then creator-local view distribution when labels are incomplete. The default inferred split is top 20 percent as hit, middle 60 percent as ok, and bottom 20 percent as miss when enough creator-local data exists.

This makes creator baselines fairer and keeps the product honest about what it knows.

## Market Memory

The Market Map expands the product beyond isolated creator scoring. It indexes local artifacts into a cohort-level view:

- total creators and videos
- coverage tiers
- creators ready for idea review
- formats by observed hit rate
- topics by available coverage
- recommended next processing queue

More creators improve market coverage, comparison, and workflow prioritization. They do not guarantee prediction. The product becomes more useful because the strategist can see what has been studied, what remains thin, and where to invest processing time next.

## Coverage Tiers

- `metadata_only`: enough to know the creator/video exists and reason about rough engagement distribution.
- `semantic_ready`: captions or segment memory exist, so text/idea review is possible.
- `deep_style_ready`: semantic artifacts, visual artifacts, and labels exist, so richer creator-style evidence is available.

Confidence is separate from the tier. A creator can be semantic-ready but still low confidence if only a few videos are covered.

## Trustworthy Product Shape

The workspace is trustworthy because it shows its evidence boundary:

- Creator confidence depends on local coverage.
- Recommendations show retrieved examples rather than only a score.
- Hit and miss labels are evidence buckets, not prediction labels.
- Decisions require a human approve/revise/reject action.
- Notes and timestamps are saved in append-only JSONL for auditability.
- The demo path avoids live scraping, LLM calls, and hidden network behavior.

## Artifact-First Cloud Workflow

Scaling should happen off the laptop on Colab, RunPod, Modal, or another GPU machine. The cloud worker can temporarily hold raw media, but the durable output is a derived artifact pack.

Tiered processing strategy:

- Metadata breadth: collect or load creator/video metadata for many creators.
- Semantic tier: generate captions/transcripts and embeddings for selected creators.
- Deep style tier: run expensive CV/audio/OCR/style extraction only for creators worth deeper study.

Durable artifacts:

- raw metadata JSON
- captions
- raw visual JSON
- embeddings store
- drafts
- labels
- manifest
- market index/report

Raw media lifecycle:

- raw videos are temporary working files
- derived artifacts are durable
- raw videos should not be committed
- `make clean-media-dry-run` previews deletion
- `make clean-media` deletes only media files from known raw/test video folders

## Implemented Today

- Creator library discovered from local drafts, captions, visuals, labels, and embeddings.
- Coverage and confidence calculation.
- Market cohort index with creator-local hit/ok/miss normalization.
- Market Map dashboard section.
- Pasted-idea review flow.
- Demo-safe local segment-overlap analysis.
- Hit-like and miss-like evidence buckets using human labels.
- Decision history in the dashboard.
- Append-only JSONL persistence with reviewer id.
- Artifact manifest at `data/artifacts/media_manifest.json`.
- Market index at `data/artifacts/market_index.json`.
- Derived-only artifact export/import scripts.
- Safe raw-media cleanup scripts.
- Colab artifact factory plan and example batch config.

## Future Work

- Full batch processor for `configs/market_batch.example.yaml`.
- Semantic retrieval for pasted ideas using existing embeddings when model startup is safe.
- Better hit-vs-miss retrieval that combines format, topic, motion, text density, and visual structure.
- SQLite/Postgres persistence with migrations.
- Per-creator review queues with statuses and owners.
- Confidence calibration against real post-publication outcomes.
- Artifact pack checksums and schema validation before import.

## Validation Plan

Interview five to eight creator strategists and run a workflow test:

1. Give each strategist a creator and three candidate ideas.
2. Ask them to make decisions with their normal workflow.
3. Ask them to make decisions with TikTokResonance.
4. Compare time-to-decision, confidence, number of useful revisions, and whether the evidence changed their decision.
5. Review saved decision notes for clarity and repeatability.
6. Ask whether the Market Map helps them prioritize which creators need more processing.

## Success Metrics

- Median time to first review decision decreases.
- Strategists report higher confidence when coverage is high.
- Strategists correctly distrust or manually inspect low-coverage recommendations.
- More reviewed ideas have concrete revision notes.
- Repeat reviewers can understand prior decisions from history without asking the original reviewer.
- The processing queue reduces wasted GPU/CV work on low-priority creators.
