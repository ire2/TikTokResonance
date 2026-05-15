from pathlib import Path
import os
import json
import yaml

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse

from profiling.utils.creator_config import get_active_creator, get_default_model_name
from utils.warnings import silence_common_warnings
from profiling.embedding.embedder import TextEmbedder
from profiling.embedding.embedding_store import load_creator_embeddings
from resonance.idea_encoder import encode_idea
from resonance.resonance_score import compute_resonance
from resonance.resonance_report import build_resonance_report
from resonance.creator_library import build_creator_library, get_creator
from resonance.idea_review import (
    IdeaReviewError,
    analyze_pasted_idea,
    default_creator_id,
)
from resonance.review_decisions import (
    DEFAULT_REVIEWER_ID,
    DEFAULT_DECISIONS_PATH,
    ReviewDecisionError,
    load_review_decisions,
    save_review_decision,
)
from profiling.cv.visual_signals import extract_visual_signals
from profiling.cv.learned_classifier import LearnedFormatClassifier
from profiling.nlp.asr import ensure_captions
from profiling.nlp.transcript_loader import load_transcript


DATA_DIR = Path(os.getenv("RESONANCE_DATA_DIR", "data"))
TEST_VIDEO_DIR = DATA_DIR / "test" / "video"
RAW_VISUAL_PATH = DATA_DIR / "raw_visual"
DRAFTS_DIR = DATA_DIR / "drafts"
CACHE_PATH = Path(os.getenv("RESONANCE_CACHE_PATH", str(DATA_DIR / "demo" / "resonance_cache.json")))
REVIEW_DECISIONS_PATH = Path(
    os.getenv("REVIEW_DECISIONS_PATH", str(DEFAULT_DECISIONS_PATH))
)
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
app = FastAPI()
silence_common_warnings()


class ReviewDecisionRequest(BaseModel):
    decision: str
    notes: str = ""
    creator_id: str | None = None
    idea_text: str | None = None
    reviewer_id: str = DEFAULT_REVIEWER_ID


class IdeaReviewRequest(BaseModel):
    creator_id: str
    idea_text: str


def _with_creator_context(payload: dict) -> dict:
    creator_id = payload.get("creator_id")
    if creator_id:
        creator = get_creator(creator_id, DATA_DIR)
        if creator:
            payload = dict(payload)
            payload["coverage"] = creator
            payload.setdefault("analysis_mode", "demo_cache" if DEMO_MODE else "live_video")
            payload.setdefault(
                "analysis_note",
                "Uses local project artifacts only in demo mode.",
            )
    return payload


def _find_latest_video() -> Path:
    if not TEST_VIDEO_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {TEST_VIDEO_DIR}")
    candidates = []
    for ext in ("*.mp4", "*.mov", "*.mkv"):
        candidates.extend(TEST_VIDEO_DIR.glob(ext))
    if not candidates:
        raise FileNotFoundError(f"No videos found in {TEST_VIDEO_DIR}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _extract_idea_text(video_path: Path) -> str:
    creator_id = "idea"
    video_id = video_path.stem
    caption = ensure_captions(
        creator_id=creator_id,
        video_id=video_id,
        video_path=str(video_path),
    )
    if not caption:
        return ""
    transcript = load_transcript(caption)
    return " ".join(s.get("text", "") for s in transcript.get("segments", []))


def _suggestions(resonance: dict, creator_profile: dict, idea_visual: dict) -> list[dict]:
    suggestions = []

    semantic = resonance.get("semantic_alignment", 0.0)
    if semantic < 0.35:
        topics = (
            creator_profile.get("profile_nlp", {})
            .get("topic_profile", {})
            .get("topics", [])
        )
        if topics:
            top_terms = topics[0].get("top_terms", [])
            if top_terms:
                suggestions.append({
                    "type": "topic",
                    "title": "Align with creator topics",
                    "detail": f"Consider framing around: {', '.join(top_terms[:6])}",
                })

    fmt = resonance.get("format_alignment")
    if fmt is not None and fmt < 0.6:
        dominant = (
            creator_profile.get("observed_patterns", {})
            .get("dominant_formats", [])
        )
        if dominant:
            suggestions.append({
                "type": "format",
                "title": "Switch format",
                "detail": f"Try a format the creator uses often: {', '.join(dominant[:3])}",
            })

    motion_align = resonance.get("motion_alignment")
    if motion_align is not None and motion_align < 0.7:
        creator_motion = (
            creator_profile.get("visual_signals", {}).get("avg_motion_intensity")
        )
        idea_motion = idea_visual.get("motion_intensity")
        if creator_motion is not None and idea_motion is not None:
            direction = "lower" if idea_motion > creator_motion else "higher"
            suggestions.append({
                "type": "motion",
                "title": "Adjust motion energy",
                "detail": f"Motion feels too {direction}; aim closer to creator baseline.",
            })

    text_align = resonance.get("text_density_alignment")
    if text_align is not None and text_align < 0.7:
        creator_text = (
            creator_profile.get("visual_signals", {}).get("avg_text_density_ocr")
            or creator_profile.get("visual_signals", {}).get("avg_text_density_heuristic")
        )
        idea_text = (
            idea_visual.get("text_density_ocr")
            or idea_visual.get("text_density_heuristic")
        )
        if creator_text is not None and idea_text is not None:
            direction = "less" if idea_text > creator_text else "more"
            suggestions.append({
                "type": "text",
                "title": "Adjust on-screen text",
                "detail": f"Use {direction} on-screen text to match creator style.",
            })

    return suggestions


def _compute_resonance_from_video():
    if DEMO_MODE:
        if not CACHE_PATH.exists():
            raise FileNotFoundError(f"Demo cache not found: {CACHE_PATH}")
        payload = json.loads(CACHE_PATH.read_text())
        creator_id = payload.get("creator_id")
        if creator_id:
            for e in payload.get("evidence", []):
                if e.get("video_id") and not e.get("tiktok_url"):
                    e["tiktok_url"] = f"https://www.tiktok.com/@{creator_id}/video/{e['video_id']}"
        return _with_creator_context(payload)

    creator_id = get_active_creator()
    model_name = get_default_model_name()
    video_path = _find_latest_video()

    profile_path = DRAFTS_DIR / f"{creator_id}_draft.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing profile: {profile_path}")
    creator_profile = yaml.safe_load(profile_path.read_text())

    creator_embedding_payload = load_creator_embeddings(
        creator_id=creator_id,
        model_name=model_name,
    )
    if creator_embedding_payload is None:
        raise ValueError("Creator embeddings not found. Run embeddings first.")

    idea_visual = extract_visual_signals(str(video_path))
    idea_motion_intensity = idea_visual.get("motion_intensity")
    idea_text_density = (
        idea_visual.get("text_density_ocr")
        or idea_visual.get("text_density_heuristic")
    )

    format_classifier = LearnedFormatClassifier()
    idea_format = None
    if format_classifier.is_ready():
        idea_format = format_classifier.classify(
            {"local_path": str(video_path)}
        )

    idea_text = _extract_idea_text(video_path).strip()
    if not idea_text:
        idea_text = " "

    embedder = TextEmbedder(model_name=model_name)
    idea = encode_idea(idea_text, embedder)

    per_video_visuals = None
    visual_path = RAW_VISUAL_PATH / f"{creator_id}.json"
    if visual_path.exists():
        per_video_visuals = json.loads(visual_path.read_text())

    resonance = compute_resonance(
        idea_embedding=idea["embedding"],
        creator_embedding_payload=creator_embedding_payload,
        creator_profile=creator_profile,
        idea_motion_intensity=idea_motion_intensity,
        idea_text_density=idea_text_density,
        idea_format=idea_format,
        per_video_visuals=per_video_visuals,
    )

    report = build_resonance_report(
        idea=idea_text,
        resonance=resonance,
    )

    evidence = report.get("top_similar_moments", [])
    for e in evidence:
        vid = e.get("video_id")
        if vid:
            e["tiktok_url"] = f"https://www.tiktok.com/@{creator_id}/video/{vid}"

    return _with_creator_context({
        "creator_id": creator_id,
        "model_name": model_name,
        "video_path": str(video_path),
        "idea_text": report["idea_text"],
        "resonance": report["resonance"],
        "interpretation": report["interpretation"],
        "evidence": evidence,
        "suggestions": _suggestions(resonance, creator_profile, idea_visual),
    })


@app.get("/api/resonance")
def api_resonance():
    try:
        return JSONResponse(_compute_resonance_from_video())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/creators")
def api_creators():
    creators = build_creator_library(DATA_DIR)
    default_id = get_active_creator()
    if not any(c["creator_id"] == default_id for c in creators):
        default_id = creators[0]["creator_id"] if creators else None
    return JSONResponse({
        "cohort_label": "Curated demo cohort",
        "default_creator_id": default_id,
        "creators": creators,
    })


@app.get("/api/creator/{creator_id}")
def api_creator(creator_id: str):
    creator = get_creator(creator_id, DATA_DIR)
    if creator is None:
        raise HTTPException(status_code=404, detail="creator not found")
    return JSONResponse({"creator": creator})


@app.post("/api/idea-review")
def api_idea_review(request: IdeaReviewRequest):
    try:
        payload = analyze_pasted_idea(
            creator_id=request.creator_id,
            idea_text=request.idea_text,
            data_dir=DATA_DIR,
            cache_path=CACHE_PATH,
        )
        return JSONResponse(payload)
    except IdeaReviewError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/review-decisions")
def api_review_decisions():
    return JSONResponse({
        "decisions": load_review_decisions(REVIEW_DECISIONS_PATH, limit=5),
        "path": str(REVIEW_DECISIONS_PATH),
    })


@app.post("/api/review-decision")
def api_review_decision(request: ReviewDecisionRequest):
    try:
        if (request.idea_text or "").strip():
            creator_id = request.creator_id or default_creator_id(DATA_DIR)
            if not creator_id:
                raise ReviewDecisionError("No creator available for review")
            payload = analyze_pasted_idea(
                creator_id=creator_id,
                idea_text=request.idea_text or "",
                data_dir=DATA_DIR,
                cache_path=CACHE_PATH,
            )
            source = "demo-local" if DEMO_MODE else "local"
        else:
            payload = _compute_resonance_from_video()
            source = "demo" if DEMO_MODE else "live"
        record = save_review_decision(
            payload=payload,
            decision=request.decision,
            notes=request.notes,
            path=REVIEW_DECISIONS_PATH,
            source=source,
            reviewer_id=request.reviewer_id,
        )
        return JSONResponse({
            "saved": True,
            "decision": record,
            "path": str(REVIEW_DECISIONS_PATH),
        })
    except ReviewDecisionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
def index():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Creator Strategy Workspace</title>
      <style>
        :root {
          --bg: #0d0f12;
          --panel: #15191f;
          --panel-2: #10141a;
          --text: #f1f5f2;
          --muted: #a8b3ad;
          --border: #2a3430;
          --accent: #3ee7b6;
          --accent-2: #ffcf5a;
          --danger: #ff8d8d;
          --ok: #8ec5ff;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          color: var(--text);
          background: linear-gradient(180deg, #0d0f12 0%, #111417 100%);
        }
        button, textarea, select { font: inherit; }
        .page { max-width: 1260px; margin: 0 auto; padding: 24px 18px 40px; }
        .header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          padding-bottom: 14px;
          border-bottom: 1px solid var(--border);
        }
        .title { font-size: 28px; font-weight: 760; letter-spacing: 0; }
        .sub { color: var(--muted); font-size: 13px; line-height: 1.45; }
        .workspace {
          display: grid;
          grid-template-columns: 310px minmax(0, 1fr);
          gap: 14px;
          margin-top: 14px;
        }
        .panel {
          background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 14px;
        }
        .panel-title {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 10px;
        }
        .panel-title h2, .panel-title h3 { margin: 0; font-size: 16px; letter-spacing: 0; }
        .stack { display: grid; gap: 12px; }
        .creator-list { display: grid; gap: 8px; }
        .creator-option {
          width: 100%;
          text-align: left;
          background: #11161b;
          color: var(--text);
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 10px;
          cursor: pointer;
        }
        .creator-option.selected { border-color: var(--accent); background: #12211d; }
        .creator-name { font-weight: 720; }
        .coverage-mini {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 6px;
          margin-top: 8px;
        }
        .mini {
          border: 1px solid #26332e;
          border-radius: 6px;
          padding: 6px;
          color: var(--muted);
          font-size: 12px;
          min-height: 46px;
        }
        .mini strong { display: block; color: var(--text); font-size: 16px; }
        .pill {
          border: 1px solid var(--border);
          border-radius: 999px;
          color: var(--muted);
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-size: 12px;
          line-height: 1;
          padding: 6px 9px;
          white-space: nowrap;
        }
        .confidence-high { color: #102019; background: var(--accent); border-color: var(--accent); }
        .confidence-medium { color: #1d1604; background: var(--accent-2); border-color: var(--accent-2); }
        .confidence-low { color: #2a0909; background: var(--danger); border-color: var(--danger); }
        textarea {
          width: 100%;
          min-height: 112px;
          padding: 10px 12px;
          border-radius: 8px;
          border: 1px solid var(--border);
          background: #0e1318;
          color: var(--text);
          resize: vertical;
        }
        .actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
        button.primary {
          background: var(--accent);
          color: #071713;
          border: 0;
          border-radius: 8px;
          font-weight: 760;
          padding: 10px 13px;
          cursor: pointer;
        }
        button.secondary, .decision-btn {
          background: #11161b;
          color: var(--text);
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 10px 12px;
          cursor: pointer;
        }
        .decision-btn.selected { border-color: var(--accent); background: #12241f; color: var(--accent); }
        .analysis-grid {
          display: grid;
          grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
          gap: 14px;
        }
        .score-row {
          display: grid;
          grid-template-columns: 128px minmax(0, 1fr);
          gap: 14px;
          align-items: center;
        }
        .score {
          font-size: 40px;
          font-weight: 780;
          letter-spacing: 0;
        }
        .score small { display: block; color: var(--muted); font-size: 12px; font-weight: 500; }
        .metric-row {
          display: grid;
          grid-template-columns: 118px minmax(0, 1fr) 48px;
          gap: 8px;
          align-items: center;
          margin-top: 8px;
          color: var(--muted);
          font-size: 12px;
        }
        .bar {
          height: 8px;
          border-radius: 999px;
          background: #0a0f13;
          border: 1px solid #25322f;
          overflow: hidden;
        }
        .bar > div { height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent) 0%, var(--ok) 100%); }
        .evidence-list { display: grid; gap: 8px; }
        .evidence-item, .recent-decision, .suggestion {
          border: 1px solid #26332e;
          border-radius: 8px;
          background: #10161b;
          padding: 10px;
          color: var(--muted);
          font-size: 13px;
          line-height: 1.42;
        }
        .evidence-item strong, .recent-decision strong, .suggestion strong { color: var(--text); }
        .evidence-item a { color: var(--ok); text-decoration: none; font-size: 12px; }
        .evidence-item a:hover { text-decoration: underline; }
        .columns { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
        .review-controls { display: flex; flex-wrap: wrap; gap: 8px; }
        .review-status, .error { font-size: 12px; color: var(--muted); }
        .error { color: var(--danger); }
        .section-label { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
        .muted { color: var(--muted); }
        @media (max-width: 940px) {
          .workspace, .analysis-grid, .score-row, .columns { grid-template-columns: 1fr; }
        }
      </style>
    </head>
    <body>
      <div class="page">
        <div class="header">
          <div>
            <div class="title">Creator Strategy Workspace</div>
            <div class="sub" id="subtitle">Curated demo cohort · evidence-based idea review · append-only strategy memory</div>
          </div>
          <button class="secondary" onclick="refreshWorkspace()">Refresh</button>
        </div>

        <div class="workspace">
          <aside class="panel stack">
            <div>
              <div class="panel-title">
                <h2>Creator Library</h2>
                <span class="pill" id="cohortLabel">Curated demo cohort</span>
              </div>
              <div class="creator-list" id="creatorList"></div>
            </div>
            <div>
              <div class="panel-title">
                <h3>Coverage</h3>
                <span class="pill" id="confidencePill">--</span>
              </div>
              <div class="coverage-mini" id="coverageGrid"></div>
              <div class="sub" style="margin-top:8px;" id="confidenceReason"></div>
            </div>
          </aside>

          <main class="stack">
            <section class="panel stack">
              <div class="panel-title">
                <h2>Idea Review</h2>
                <span class="pill" id="selectedCreatorPill">--</span>
              </div>
              <textarea id="ideaInput" placeholder="Paste a creator-specific video idea"></textarea>
              <div class="actions">
                <button class="primary" type="button" onclick="analyzeIdea()">Analyze</button>
                <span class="sub" id="analysisMode">Local analysis only</span>
              </div>
            </section>

            <section class="analysis-grid">
              <div class="panel stack">
                <div>
                  <div class="section-label">Fit Analysis</div>
                  <div class="score-row">
                    <div class="score" id="score">--<small>fit score</small></div>
                    <div id="metrics"></div>
                  </div>
                </div>
                <div>
                  <div class="section-label">Interpretation</div>
                  <div class="sub" id="interpretation">No analysis yet.</div>
                </div>
                <div>
                  <div class="section-label">Evidence Note</div>
                  <div class="sub" id="analysisNote"></div>
                </div>
                <div>
                  <div class="section-label">Idea Surgery</div>
                  <div class="evidence-list" id="suggestions"></div>
                </div>
              </div>

              <div class="panel stack">
                <div>
                  <div class="section-label">Top Evidence</div>
                  <div class="evidence-list" id="evidence"></div>
                </div>
                <div class="columns">
                  <div>
                    <div class="section-label">Resembles Prior Hits</div>
                    <div class="evidence-list" id="hitEvidence"></div>
                  </div>
                  <div>
                    <div class="section-label">Risk: Prior Misses</div>
                    <div class="evidence-list" id="missEvidence"></div>
                  </div>
                </div>
              </div>
            </section>

            <section class="analysis-grid">
              <div class="panel stack">
                <div class="panel-title">
                  <h2>Human Decision</h2>
                  <span class="pill">creator_strategist</span>
                </div>
                <div class="review-controls">
                  <button class="decision-btn" id="decision-approve" type="button" onclick="setDecision('approve')">Approve</button>
                  <button class="decision-btn" id="decision-revise" type="button" onclick="setDecision('revise')">Revise</button>
                  <button class="decision-btn" id="decision-reject" type="button" onclick="setDecision('reject')">Reject</button>
                </div>
                <textarea id="reviewNotes" maxlength="1000" placeholder="Decision notes"></textarea>
                <div class="actions">
                  <button class="primary" type="button" onclick="saveDecision()">Save Decision</button>
                  <span class="review-status" id="reviewStatus">Writes to data/reviews/resonance_decisions.jsonl</span>
                </div>
              </div>

              <div class="panel stack">
                <div class="panel-title">
                  <h2>Decision History</h2>
                  <span class="pill" id="decisionPath">local JSONL</span>
                </div>
                <div class="evidence-list" id="recentDecisions"></div>
              </div>
            </section>
          </main>
        </div>
      </div>
    <script>
      let creators = [];
      let selectedCreatorId = '';
      let selectedDecision = 'revise';
      let currentAnalysis = null;
      const sampleIdea = 'A fast, visual explanation of why a familiar everyday object works differently than people think, ending with one practical takeaway.';
      const fmt = (v) => (v === null || v === undefined || Number.isNaN(Number(v))) ? 'N/A' : Number(v).toFixed(2);
      const CENSOR_WORDS = [
        'fuck', 'fucking', 'fucked', 'fucker',
        'shit', 'bullshit', 'bitch', 'asshole',
        'dick', 'pussy', 'cunt', 'bastard',
        'slut', 'whore', 'damn', 'mf'
      ];
      const ESCAPE_CHARS = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};

      function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, ch => ESCAPE_CHARS[ch]);
      }
      function censor(text) {
        if (!text) return '';
        let out = text;
        for (const w of CENSOR_WORDS) {
          const re = new RegExp(`\\b${w}\\b`, 'gi');
          out = out.replace(re, (m) => m[0] + '*'.repeat(Math.max(0, m.length - 1)));
        }
        return out;
      }
      function safeTikTokUrl(value) {
        try {
          const url = new URL(value, window.location.origin);
          const host = url.hostname.toLowerCase();
          if (url.protocol === 'https:' && (host === 'tiktok.com' || host === 'www.tiktok.com' || host.endsWith('.tiktok.com'))) {
            return url.href;
          }
        } catch (_) {}
        return '';
      }
      function currentCreator() {
        return creators.find(c => c.creator_id === selectedCreatorId) || null;
      }
      function confidenceClass(level) {
        return `confidence-${level || 'low'}`;
      }
      function setDecision(decision) {
        selectedDecision = decision;
        for (const value of ['approve', 'revise', 'reject']) {
          const el = document.getElementById(`decision-${value}`);
          if (el) el.classList.toggle('selected', value === decision);
        }
      }
      function selectCreator(creatorId) {
        selectedCreatorId = creatorId;
        renderCreators();
        renderCoverage();
        analyzeIdea();
      }
      function renderCreators() {
        const target = document.getElementById('creatorList');
        target.innerHTML = creators.map(c => {
          const selected = c.creator_id === selectedCreatorId ? ' selected' : '';
          const formats = (c.dominant_formats || []).slice(0, 2).join(', ') || 'formats pending';
          return `<button class="creator-option${selected}" type="button" data-creator="${escapeHtml(c.creator_id)}">
            <div class="creator-name">${escapeHtml(c.creator_id)}</div>
            <div class="sub">${escapeHtml(c.videos_analyzed)} videos · ${escapeHtml(c.human_label_count)} labels · ${escapeHtml(formats)}</div>
          </button>`;
        }).join('');
        target.querySelectorAll('[data-creator]').forEach(btn => {
          btn.addEventListener('click', () => selectCreator(btn.dataset.creator));
        });
      }
      function renderCoverage() {
        const creator = currentCreator();
        if (!creator) return;
        document.getElementById('selectedCreatorPill').innerText = creator.creator_id;
        const pill = document.getElementById('confidencePill');
        pill.className = `pill ${confidenceClass(creator.confidence_level)}`;
        pill.innerText = `${creator.confidence_level} confidence`;
        document.getElementById('confidenceReason').innerText = creator.confidence_reason || '';
        document.getElementById('coverageGrid').innerHTML = `
          <div class="mini"><strong>${escapeHtml(creator.videos_analyzed)}</strong>videos analyzed</div>
          <div class="mini"><strong>${escapeHtml(creator.caption_count)}</strong>captions</div>
          <div class="mini"><strong>${escapeHtml(creator.visual_signal_count)}</strong>visual signals</div>
          <div class="mini"><strong>${escapeHtml(creator.human_label_count)}</strong>human labels</div>
          <div class="mini"><strong>${escapeHtml(creator.hit_count)}/${escapeHtml(creator.ok_count)}/${escapeHtml(creator.miss_count)}</strong>hit / ok / miss</div>
          <div class="mini"><strong>${creator.embeddings_exist ? 'yes' : 'no'}</strong>embeddings</div>
        `;
      }
      function metricRow(label, value) {
        const numeric = Number(value);
        const pct = Number.isFinite(numeric) ? Math.max(0, Math.min(1, numeric)) * 100 : 0;
        return `<div class="metric-row">
          <span>${escapeHtml(label)}</span>
          <span class="bar"><div style="width:${pct.toFixed(1)}%"></div></span>
          <span>${escapeHtml(fmt(value))}</span>
        </div>`;
      }
      function evidenceItem(item) {
        const label = item.performance_label ? `${item.performance_label}` : 'unlabeled';
        const format = item.format_label ? ` · ${item.format_label}` : '';
        const url = safeTikTokUrl(item.tiktok_url);
        return `<div class="evidence-item">
          <div><strong>video ${escapeHtml(item.video_id || '-')}</strong> · ${escapeHtml(label)}${escapeHtml(format)} · similarity ${escapeHtml(fmt(item.similarity))}</div>
          <div>${escapeHtml(censor(item.text || ''))}</div>
          ${url ? `<a href="${url}" target="_blank" rel="noopener">Open video</a>` : ''}
        </div>`;
      }
      function renderEvidence(targetId, items, emptyText) {
        const target = document.getElementById(targetId);
        target.innerHTML = (items || []).map(evidenceItem).join('') || `<div class="sub">${escapeHtml(emptyText)}</div>`;
      }
      function renderAnalysis(data) {
        currentAnalysis = data;
        const resonance = data.resonance || {};
        document.getElementById('score').innerHTML = `${escapeHtml(fmt(resonance.resonance_score))}<small>fit score</small>`;
        document.getElementById('metrics').innerHTML = [
          metricRow('semantic evidence', resonance.semantic_alignment),
          metricRow('format', resonance.format_alignment),
          metricRow('motion', resonance.motion_alignment),
          metricRow('text density', resonance.text_density_alignment),
        ].join('');
        const interp = data.interpretation || {};
        document.getElementById('interpretation').innerText =
          `Semantic fit: ${interp.semantic_fit || '-'} · Format match: ${interp.format_match || '-'} · Coverage confidence: ${interp.confidence || data.coverage?.confidence_level || '-'}`;
        document.getElementById('analysisMode').innerText = data.analysis_mode || 'local';
        document.getElementById('analysisNote').innerText = data.analysis_note || '';
        renderEvidence('evidence', (data.evidence || []).slice(0, 5), 'No local evidence found for this idea.');
        renderEvidence('hitEvidence', data.hit_evidence || [], 'No hit-labeled matches in the retrieved evidence.');
        renderEvidence('missEvidence', data.miss_evidence || [], 'No miss-labeled matches in the retrieved evidence.');
        const suggestions = (data.suggestions || []).map(s => (
          `<div class="suggestion"><strong>${escapeHtml(s.title || '')}</strong><div>${escapeHtml(s.detail || '')}</div></div>`
        )).join('');
        document.getElementById('suggestions').innerHTML = suggestions || '<div class="sub">No revision suggestion from current evidence.</div>';
      }
      async function analyzeIdea() {
        const ideaText = document.getElementById('ideaInput').value.trim();
        if (!selectedCreatorId || !ideaText) return;
        document.getElementById('analysisMode').innerText = 'Analyzing...';
        const res = await fetch('/api/idea-review', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ creator_id: selectedCreatorId, idea_text: ideaText })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          document.getElementById('analysisMode').innerText = `Analysis failed: ${data.detail || data.error || res.status}`;
          return;
        }
        renderAnalysis(data);
      }
      function renderRecentDecisions(decisions) {
        const target = document.getElementById('recentDecisions');
        const html = (decisions || []).map(d => {
          const note = d.notes ? `<div>${escapeHtml(d.notes)}</div>` : '';
          return `<div class="recent-decision">
            <div><strong>${escapeHtml(d.decision || '-')}</strong> · ${escapeHtml(d.creator_id || '-')} · score ${escapeHtml(fmt(d.resonance_score))}</div>
            <div>${escapeHtml(d.idea_snippet || d.idea_text || '')}</div>
            <div class="sub">${escapeHtml(d.created_at || '')}</div>
            ${note}
          </div>`;
        }).join('');
        target.innerHTML = html || '<div class="sub">No review decisions saved yet.</div>';
      }
      async function loadRecentDecisions() {
        try {
          const res = await fetch('/api/review-decisions');
          const data = await res.json();
          document.getElementById('decisionPath').innerText = data.path || 'local JSONL';
          renderRecentDecisions(data.decisions || []);
        } catch (_) {
          renderRecentDecisions([]);
        }
      }
      async function saveDecision() {
        const notes = document.getElementById('reviewNotes').value || '';
        const ideaText = document.getElementById('ideaInput').value.trim();
        const status = document.getElementById('reviewStatus');
        status.innerText = 'Saving decision...';
        status.classList.remove('error');
        const res = await fetch('/api/review-decision', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            decision: selectedDecision,
            notes,
            creator_id: selectedCreatorId,
            idea_text: ideaText,
            reviewer_id: 'creator_strategist'
          })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          status.innerText = `Save failed: ${data.detail || data.error || res.status}`;
          status.classList.add('error');
          return;
        }
        status.innerText = `Saved ${data.decision.decision} decision to ${data.path}`;
        document.getElementById('reviewNotes').value = '';
        await loadRecentDecisions();
      }
      async function loadCreators() {
        const res = await fetch('/api/creators');
        const data = await res.json();
        creators = data.creators || [];
        selectedCreatorId = data.default_creator_id || creators[0]?.creator_id || '';
        document.getElementById('cohortLabel').innerText = data.cohort_label || 'Curated demo cohort';
        renderCreators();
        renderCoverage();
      }
      async function refreshWorkspace() {
        await loadCreators();
        await analyzeIdea();
        await loadRecentDecisions();
      }
      document.getElementById('ideaInput').value = sampleIdea;
      setDecision(selectedDecision);
      refreshWorkspace();
    </script>
    </body>
    </html>
    """
    return HTMLResponse(html)
