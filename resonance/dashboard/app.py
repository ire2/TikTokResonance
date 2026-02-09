from pathlib import Path
import json
import yaml

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from profiling.utils.creator_config import get_active_creator, get_default_model_name
from utils.warnings import silence_common_warnings
from profiling.embedding.embedder import TextEmbedder
from profiling.embedding.embedding_store import load_creator_embeddings
from resonance.idea_encoder import encode_idea
from resonance.resonance_score import compute_resonance
from resonance.resonance_report import build_resonance_report
from profiling.cv.visual_signals import extract_visual_signals
from profiling.cv.learned_classifier import LearnedFormatClassifier
from profiling.nlp.asr import ensure_captions
from profiling.nlp.transcript_loader import load_transcript


TEST_VIDEO_DIR = Path("data/test/video")
RAW_VISUAL_PATH = Path("data/raw_visual")
DRAFTS_DIR = Path("data/drafts")

app = FastAPI()
silence_common_warnings()


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

    return {
        "creator_id": creator_id,
        "video_path": str(video_path),
        "idea_text": report["idea_text"],
        "resonance": report["resonance"],
        "interpretation": report["interpretation"],
        "evidence": report.get("top_similar_moments", []),
        "suggestions": _suggestions(resonance, creator_profile, idea_visual),
    }


@app.get("/api/resonance")
def api_resonance():
    try:
        return JSONResponse(_compute_resonance_from_video())
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
      <title>Resonance Dashboard</title>
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Work+Sans:wght@400;500;600&display=swap" rel="stylesheet">
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      <style>
        :root {
          --bg: #0b0e12;
          --card: #121825;
          --card-2: #0f1520;
          --text: #eaf2ff;
          --muted: #9fb3c8;
          --accent: #21f0c9;
          --accent-2: #6ee7ff;
          --border: #243242;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: "Work Sans", sans-serif;
          color: var(--text);
          background:
            radial-gradient(900px 500px at 8% -10%, #1d2a40 0%, transparent 60%),
            radial-gradient(900px 600px at 100% 0%, #1b2b3a 0%, transparent 55%),
            linear-gradient(180deg, #0b0e12 0%, #090c10 100%);
        }
        .page { max-width: 1200px; margin: 0 auto; padding: 28px 20px 40px; }
        .header { display:flex; align-items:center; justify-content:space-between; gap:16px; }
        .title { font-family: "Space Grotesk", sans-serif; font-size: 30px; font-weight: 700; }
        .sub { color: var(--muted); font-size: 14px; }
        .grid { display:grid; grid-template-columns: 1.1fr 0.9fr; gap:16px; margin-top:16px; }
        .card {
          background: linear-gradient(180deg, var(--card) 0%, var(--card-2) 100%);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 16px;
        }
        .pill { padding:6px 10px; border:1px solid var(--border); border-radius: 999px; color: var(--muted); font-size:12px; display:inline-flex; gap:8px; }
        .metrics { display:grid; grid-template-columns: repeat(2,1fr); gap:8px; margin-top:10px; }
        .metric { background:#0f1522; border:1px solid #26364a; border-radius:10px; padding:8px 10px; font-size:13px; color:var(--muted); }
        .suggestion { margin-top:10px; padding:10px; border:1px solid #233246; border-radius:12px; }
        .suggestion h4 { margin:0 0 6px 0; font-size:14px; }
        .evidence { font-size:13px; color: var(--muted); line-height:1.4; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        button { background: linear-gradient(180deg, #21f0c9 0%, #0ec2a3 100%); border:0; padding:10px 14px; border-radius:10px; font-weight:700; cursor:pointer; }
      </style>
    </head>
    <body>
      <div class="page">
        <div class="header">
          <div>
            <div class="title">Resonance Dashboard</div>
            <div class="sub" id="subtitle">Loading...</div>
          </div>
          <button onclick="loadData()">Refresh</button>
        </div>

        <div class="grid">
          <div class="card">
            <canvas id="radar"></canvas>
            <div class="metrics" id="metrics"></div>
          </div>
          <div class="card">
            <div class="pill" id="score"></div>
            <div style="margin-top:10px;">
              <div class="sub">Idea (auto‑transcribed)</div>
              <div class="evidence" id="idea"></div>
            </div>
            <div style="margin-top:12px;">
              <div class="sub">Top Evidence</div>
              <div class="evidence" id="evidence"></div>
            </div>
            <div style="margin-top:12px;">
              <div class="sub">Idea Surgery</div>
              <div id="suggestions"></div>
            </div>
          </div>
        </div>
      </div>
    <script>
      let chart;
      async function loadData() {
        const res = await fetch('/api/resonance');
        const data = await res.json();
        if (data.error) {
          document.getElementById('subtitle').innerText = data.error;
          return;
        }
        document.getElementById('subtitle').innerText = `Creator: ${data.creator_id}`;
        document.getElementById('score').innerText = `Resonance Score: ${data.resonance.resonance_score}`;
        document.getElementById('idea').innerText = data.idea_text || '';

        const ev = (data.evidence || []).slice(0,3).map(e => `• ${e.text}`).join('\\n');
        document.getElementById('evidence').innerText = ev || 'No evidence available';

        const sugg = (data.suggestions || []).map(s => (
          `<div class="suggestion"><h4>${s.title}</h4><div class="sub">${s.detail}</div></div>`
        )).join('');
        document.getElementById('suggestions').innerHTML = sugg || '<div class="sub">No suggestions.</div>';

        const labels = ['semantic', 'format', 'motion', 'text_density'];
        const values = [
          data.resonance.semantic_alignment || 0,
          data.resonance.format_alignment || 0,
          data.resonance.motion_alignment || 0,
          data.resonance.text_density_alignment || 0,
        ];

        if (chart) chart.destroy();
        const ctx = document.getElementById('radar');
        chart = new Chart(ctx, {
          type: 'radar',
          data: {
            labels,
            datasets: [{
              label: 'Fit',
              data: values,
              fill: true,
              backgroundColor: 'rgba(33, 240, 201, 0.15)',
              borderColor: '#21f0c9',
              pointBackgroundColor: '#6ee7ff',
            }]
          },
          options: {
            scales: {
              r: {
                suggestedMin: 0,
                suggestedMax: 1,
                grid: { color: '#243242' },
                angleLines: { color: '#243242' },
                pointLabels: { color: '#9fb3c8' },
                ticks: { color: '#9fb3c8' }
              }
            },
            plugins: { legend: { labels: { color: '#eaf2ff' } } }
          }
        });

        const metrics = document.getElementById('metrics');
        metrics.innerHTML = `
          <div class="metric">semantic_gate: ${data.resonance.semantic_gate ?? '-'}</div>
          <div class="metric">dialogue_affinity: ${data.resonance.dialogue_affinity ?? '-'}</div>
          <div class="metric">solo_rant_affinity: ${data.resonance.solo_rant_affinity ?? '-'}</div>
          <div class="metric">talking_head_affinity: ${data.resonance.talking_head_affinity ?? '-'}</div>
        `;
      }
      loadData();
    </script>
    </body>
    </html>
    """
    return HTMLResponse(html)
