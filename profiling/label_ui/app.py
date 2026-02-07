from pathlib import Path
import csv
from typing import Dict, List

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse


BASE_DIR = Path("profiling/metadata/labels")
CSV_PATH = BASE_DIR / "format_labels.csv"

FORMAT_LABELS = [
    "talking_head",
    "voiceover",
    "text_heavy",
    "broll",
    "dance",
    "duo_or_group",
    "tutorial_or_demo",
    "skit_or_comedy",
    "product_or_ad",
    "food_or_cooking",
    "other",
]
PERFORMANCE_LABELS = ["hit", "ok", "miss"]


app = FastAPI()


def load_rows() -> List[Dict]:
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_rows(rows: List[Dict]):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "creator_id",
                "video_id",
                "format_label",
                "performance_label",
                "tiktok_url",
                "views",
                "likes",
                "comments",
                "duration_sec",
                "posted_at",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def first_unlabeled(rows: List[Dict]) -> Dict | None:
    for r in rows:
        if not r.get("format_label") or not r.get("performance_label"):
            return r
    return None


def count_labeled(rows: List[Dict]) -> int:
    return sum(
        1 for r in rows
        if r.get("format_label") and r.get("performance_label")
    )


@app.get("/", response_class=HTMLResponse)
def index():
    rows = load_rows()
    if not rows:
        return HTMLResponse(
            "<h3>No labels CSV found. Run generate_label_queue.py first.</h3>"
        )

    current = first_unlabeled(rows)
    labeled = count_labeled(rows)
    total = len(rows)

    if not current:
        return HTMLResponse(
            f"<h3>All done. Labeled {labeled}/{total}.</h3>"
        )

    url = current.get("tiktok_url", "")

    def options_html(name, options, selected=""):
        opts = []
        for o in options:
            sel = "selected" if o == selected else ""
            opts.append(f"<option value='{o}' {sel}>{o}</option>")
        return (
            f"<select name='{name}' required>"
            + "".join(opts)
            + "</select>"
        )

    html = f"""
    <html>
    <head>
      <title>Labeling UI</title>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Work+Sans:wght@400;500;600&display=swap" rel="stylesheet">
      <style>
        :root {{
          --bg: #0b0e12;
          --card: #121825;
          --card-2: #0f1520;
          --text: #eaf2ff;
          --muted: #9fb3c8;
          --accent: #21f0c9;
          --accent-2: #ffb703;
          --accent-3: #6ee7ff;
          --border: #243242;
          --shadow: 0 18px 40px rgba(0, 0, 0, 0.45);
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          font-family: "Work Sans", system-ui, sans-serif;
          color: var(--text);
          background:
            radial-gradient(900px 500px at 8% -10%, #1d2a40 0%, transparent 60%),
            radial-gradient(900px 600px at 100% 0%, #1b2b3a 0%, transparent 55%),
            radial-gradient(700px 500px at 70% 120%, #1b1f36 0%, transparent 60%),
            linear-gradient(180deg, #0b0e12 0%, #090c10 100%);
        }}
        .page {{
          max-width: 1100px;
          margin: 0 auto;
          padding: 28px 20px 40px;
        }}
        .header {{
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          margin-bottom: 16px;
        }}
        .title {{
          font-family: "Space Grotesk", sans-serif;
          font-size: 30px;
          font-weight: 700;
          letter-spacing: -0.02em;
        }}
        .sub {{
          color: var(--muted);
          font-size: 14px;
        }}
        .progress {{
          display: grid;
          gap: 8px;
          width: 260px;
        }}
        .bar {{
          height: 8px;
          background: #1b2734;
          border-radius: 999px;
          overflow: hidden;
          border: 1px solid var(--border);
        }}
        .bar > span {{
          display: block;
          height: 100%;
          width: {int((labeled / total) * 100)}%;
          background: linear-gradient(90deg, var(--accent), var(--accent-3));
          transition: width 200ms ease;
        }}
        .layout {{
          display: grid;
          gap: 16px;
          grid-template-columns: 1.15fr 0.85fr;
        }}
        .card {{
          background: linear-gradient(180deg, var(--card) 0%, var(--card-2) 100%);
          border: 1px solid var(--border);
          border-radius: 16px;
          box-shadow: var(--shadow);
          padding: 16px;
        }}
        .meta {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 8px 12px;
          color: var(--muted);
          font-size: 13px;
        }}
        .meta div {{
          background: linear-gradient(180deg, #0f1a2a 0%, #0c1522 100%);
          border: 1px solid #26364a;
          border-radius: 10px;
          padding: 8px 10px;
          box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
        }}
        .embed {{
          margin: 8px 0 6px;
          display: grid;
          place-items: center;
        }}
        .link {{
          color: #8ae9ff;
          text-decoration: none;
          font-size: 13px;
        }}
        .link:hover {{ text-decoration: underline; }}
        .form {{
          display: grid;
          gap: 12px;
        }}
        .section-title {{
          font-family: "Space Grotesk", sans-serif;
          font-weight: 600;
          letter-spacing: 0.01em;
          color: #cfe4ff;
          margin: 6px 0 2px;
        }}
        label {{
          font-size: 13px;
          color: var(--muted);
        }}
        select {{
          width: 100%;
          padding: 10px 12px;
          border-radius: 12px;
          border: 1px solid var(--border);
          background: #0f151d;
          color: var(--text);
          font-size: 14px;
        }}
        .btn {{
          padding: 12px 14px;
          border-radius: 12px;
          border: 1px solid #165f55;
          background: linear-gradient(180deg, #21f0c9 0%, #0ec2a3 100%);
          color: #08211d;
          font-weight: 700;
          cursor: pointer;
          letter-spacing: 0.01em;
        }}
        .btn:hover {{
          filter: brightness(1.05);
        }}
        .pill {{
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 6px 10px;
          border-radius: 999px;
          border: 1px solid var(--border);
          background: #0f1522;
          color: var(--muted);
          font-size: 12px;
        }}
        .note {{
          color: var(--muted);
          font-size: 12px;
        }}
        @media (max-width: 900px) {{
          .layout {{ grid-template-columns: 1fr; }}
          .progress {{ width: 100%; }}
        }}
      </style>
    </head>
    <body>
      <div class="page">
        <div class="header">
          <div>
            <div class="title">Resonance Labeling</div>
            <div class="sub">Creator: {current.get("creator_id")} • Video {labeled + 1} of {total}</div>
          </div>
          <div class="progress">
            <div class="pill">{labeled} labeled • {total - labeled} remaining</div>
            <div class="bar"><span></span></div>
          </div>
        </div>

        <div class="layout">
          <div class="card">
            <div class="embed">
              <blockquote class="tiktok-embed" cite="{url}" data-video-id="{current.get("video_id")}" style="max-width: 605px;min-width: 325px;" >
                <section></section>
              </blockquote>
              <script async src="https://www.tiktok.com/embed.js"></script>
            </div>
            <a class="link" href="{url}" target="_blank">Open in TikTok</a>
          </div>

          <div class="card">
            <div class="form">
              <div class="section-title">Video Metrics</div>
              <div class="meta">
                <div>views: {current.get("views")}</div>
                <div>likes: {current.get("likes")}</div>
                <div>comments: {current.get("comments")}</div>
                <div>duration: {current.get("duration_sec")}s</div>
                <div>posted: {current.get("posted_at")}</div>
                <div>video_id: {current.get("video_id")}</div>
              </div>
              <div class="section-title">Labels</div>
              <form action="/label" method="post">
                <input type="hidden" name="creator_id" value="{current.get("creator_id")}">
                <input type="hidden" name="video_id" value="{current.get("video_id")}">
                <label>Format</label>
                {options_html("format_label", FORMAT_LABELS, current.get("format_label",""))}
                <label>Performance</label>
                {options_html("performance_label", PERFORMANCE_LABELS, current.get("performance_label",""))}
                <div style="margin-top: 10px;">
                  <button class="btn" type="submit">Save & Next</button>
                </div>
              </form>
              <div class="note">Tip: use keyboard to quickly pick labels, then click Save.</div>
            </div>
          </div>
        </div>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.post("/label")
def label(
    creator_id: str = Form(...),
    video_id: str = Form(...),
    format_label: str = Form(...),
    performance_label: str = Form(...),
):
    rows = load_rows()
    for r in rows:
        if r.get("creator_id") == creator_id and r.get("video_id") == video_id:
            r["format_label"] = format_label
            r["performance_label"] = performance_label
            break
    save_rows(rows)
    return RedirectResponse("/", status_code=303)
