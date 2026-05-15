# Creator Strategy Workspace Product Brief

## Target User

Creator strategists, brand strategists, and creator operations teammates who review short-form video ideas before production.

## User Pain Point

Strategists make fit calls from memory, scattered links, spreadsheet labels, and subjective taste. They need a faster way to answer:

- Which creator is this idea for?
- How much evidence do we have for that creator?
- What prior examples support the idea?
- What prior examples warn against it?
- What decision did we make and why?

## Current Alternatives

- Manually scanning TikTok profiles and saved links.
- Using spreadsheets of labels without a review surface.
- Asking a teammate who remembers the creator well.
- Using generic AI ideation tools that do not know the creator-specific evidence.

## Trustworthy Product Shape

The workspace is trustworthy because it shows its evidence boundary:

- Creator confidence depends on local coverage.
- Recommendations show retrieved examples rather than only a score.
- Hit and miss labels are shown as evidence buckets, not prediction labels.
- Decisions require a human approve/revise/reject action.
- Notes and timestamps are saved in append-only JSONL for auditability.
- The demo path avoids live scraping, LLM calls, and hidden network behavior.

## Honest Limited Coverage

The demo cohort is intentionally small and curated. The UI says “Curated demo cohort” and shows coverage per creator so the user can separate strong local evidence from thin evidence. Low confidence does not block review; it changes how much the strategist should trust the tool.

## Human-In-The-Loop Trust Boundary

The system can organize evidence and suggest where an idea may fit or need revision. It cannot decide what a creator should post, predict virality, or replace a strategist’s judgment. The product boundary is: evidence-assisted review, then explicit human decision.

## Implemented Today

- Creator library discovered from local drafts, captions, visuals, labels, and embeddings.
- Coverage and confidence calculation.
- Dashboard creator selection.
- Pasted-idea review flow.
- Demo-safe local segment-overlap analysis.
- Hit-like and miss-like evidence buckets using human labels.
- Decision history in the dashboard.
- Append-only JSONL persistence with reviewer id.
- `make demo-reset` to reset only the review decision artifact.
- Focused tests for coverage, confidence, idea review, decision history, and reset behavior.

## Future Work

- Semantic retrieval for pasted text using existing embeddings when model startup is safe for demo and production.
- Per-creator review queues with statuses and owners.
- SQLite/Postgres persistence with migrations.
- Confidence calibration against real post-publication outcomes.
- Better hit-vs-miss retrieval that accounts for format, topic, and visual structure together.
- Role-based review permissions only after the single-user workflow proves useful.

## Validation Plan

Interview five to eight creator strategists and run a workflow test:

1. Give each strategist a creator and three candidate ideas.
2. Ask them to make decisions with their normal workflow.
3. Ask them to make decisions with TikTokResonance.
4. Compare time-to-decision, confidence, number of useful revisions, and whether the evidence changed their decision.
5. Review saved decision notes for clarity and repeatability.

## Success Metrics

- Median time to first review decision decreases.
- Strategists report higher confidence when coverage is high.
- Strategists correctly distrust or manually inspect low-coverage recommendations.
- More reviewed ideas have concrete revision notes.
- Repeat reviewers can understand prior decisions from the history without asking the original reviewer.
