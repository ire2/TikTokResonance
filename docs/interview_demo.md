# TikTokResonance Interview Demo

## 60-Second Pitch

TikTokResonance is a human-in-the-loop Creator Strategy Workspace. A strategist picks a creator, checks how much local evidence exists for that creator, pastes a new idea, reviews fit evidence from cached creator memory, and saves an approve/revise/reject decision. The saved decision becomes strategy memory for the next review.

The product does not predict virality, auto-approve ideas, scrape live TikTok during the demo, or call an LLM. It turns scattered creator examples into an auditable review workflow.

## Implemented Demo Workflow

1. Choose a creator from the curated demo cohort.
2. Read the coverage panel: videos analyzed, captions, visual signals, labels, hit/ok/miss counts, embeddings, and confidence level.
3. Paste or edit an idea.
4. Click Analyze.
5. Review local fit evidence, hit-like examples, miss-like risks, and revision suggestions.
6. Choose Approve, Revise, or Reject.
7. Add notes and save the decision.
8. Confirm the decision appears in Decision History and is appended to `data/reviews/resonance_decisions.jsonl`.

## Creator Library

The creator library is built from local artifacts only:

- `data/drafts/*_draft.yaml`
- `data/raw_visual/*.json`
- `data/raw_captions/*.json`
- `data/labels/format_labels.csv`
- `data/embeddings_store/*`

For each creator, the dashboard shows:

- videos analyzed
- visual signal count
- caption count
- human label count
- hit / ok / miss count
- dominant formats
- whether embeddings and segment memory exist
- confidence level: `low`, `medium`, or `high`

Confidence is based on coverage. High confidence requires useful depth across captions, labels, visual signals, and embeddings. Medium confidence means partial evidence. Low confidence means the reviewer should treat the result as thin evidence.

## Paste-Idea Flow

`POST /api/idea-review` accepts:

```json
{
  "creator_id": "expoparker",
  "idea_text": "A short creator-specific idea"
}
```

The demo-safe path uses local segment memory and human labels. It returns a fit score, top evidence, prior-hit evidence, prior-miss risk evidence, suggestions, coverage, and a clear analysis note. If segment memory is unavailable, the code can fall back to the cached demo payload and labels that fallback behavior explicitly.

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

## Demo Reset

Run:

```bash
make demo-reset
```

This truncates only:

```text
data/reviews/resonance_decisions.jsonl
```

It does not delete or modify training data, labels, raw visual data, captions, embeddings, or `data/demo/resonance_cache.json`.

## Demo Commands

Use the project environment:

```bash
source scripts/activate_env.sh
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
  resonance/test_creator_library.py \
  resonance/test_idea_review.py \
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

1. Run `make demo-reset`.
2. Run `make dashboard-demo`.
3. Point to Creator Library and say this is a curated demo cohort, not a claim of full market coverage.
4. Select `expoparker`, then `cleoabram`, to show confidence and coverage change by creator.
5. Paste or keep the sample idea and click Analyze.
6. Explain the score as evidence-based fit, not virality prediction.
7. Open Top Evidence and the hit/miss evidence buckets.
8. Choose `Revise`, add a note like `The topic fits, but the hook needs a clearer creator-native setup.`
9. Save the decision.
10. Show Decision History and mention the append-only JSONL artifact.
11. Open `resonance/creator_library.py`, `resonance/idea_review.py`, and `resonance/review_decisions.py` if asked how it works.

## Files To Keep Open

- `resonance/dashboard/app.py`: FastAPI endpoints and dashboard UI.
- `resonance/creator_library.py`: coverage and confidence calculation.
- `resonance/idea_review.py`: pasted-idea local evidence retrieval.
- `resonance/review_decisions.py`: append-only decision artifact and reset helper.
- `scripts/demo_reset.py`: demo reset command.
- `data/demo/resonance_cache.json`: deterministic fallback payload.

## Known Limits

- The dashboard is still a compact FastAPI-rendered page, not a full React app.
- Pasted-idea analysis uses lexical overlap over cached segment text rather than full semantic recomputation in demo mode.
- Hit-vs-miss evidence is based on retrieved local examples and human labels, not outcome prediction.
- Persistence is JSONL, not a multi-user database.
- There is no auth, review queue assignment, or live TikTok ingestion in the demo flow.

## Future Improvements

- Add semantic retrieval for pasted ideas using existing embeddings once model loading is demo-safe.
- Move labels and decisions into SQLite or Postgres with migrations.
- Add per-creator review queues and filters for pending/revised/rejected ideas.
- Calibrate recommendations against post-publication outcomes.
- Validate the workflow with creator strategists using time-to-decision, confidence lift, and revision quality.

## Interview Framing

The upgrade makes the project more product-like because it starts with the user decision loop. The strategist can see what data supports the recommendation, where confidence is weak, which examples support or weaken the idea, and what decision was made. The system stays honest by showing limited coverage and keeping the human as the decision maker.
