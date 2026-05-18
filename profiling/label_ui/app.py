from pathlib import Path
import csv
import html as html_lib
from typing import Dict, List
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse


BASE_DIR = Path("data/labels")
CSV_PATH = BASE_DIR / "format_labels.csv"

FORMAT_LABELS = [
    "talking_head",
    "voiceover",
    "text_heavy",
    "broll",
    "dance",
    "duo_or_group",
    "tutorial_or_demo",
    "educational",
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


def _parse_metric(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _format_compact_number(value: float | int | str | None) -> str:
    parsed = _parse_metric(value)
    if parsed is None:
        return "-"
    abs_value = abs(parsed)
    units = [
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ]
    for threshold, suffix in units:
        if abs_value >= threshold:
            scaled = parsed / threshold
            text = f"{scaled:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return f"{int(parsed):,}" if parsed.is_integer() else f"{parsed:,.1f}"


def _format_duration(value: float | int | str | None) -> str:
    parsed = _parse_metric(value)
    if parsed is None:
        return "-"
    seconds = int(round(parsed))
    if seconds < 60:
        return f"{seconds} sec"
    minutes, remainder = divmod(seconds, 60)
    if remainder == 0:
        return f"{minutes} min"
    return f"{minutes}m {remainder}s"


def _rank_summary(rank: int | None, total: int) -> tuple[str, str]:
    if not rank or total <= 0:
        return ("No rank", "Not enough creator data")
    if total == 1:
        return ("Only sample", "Needs more videos")

    percentile = max(1, round((rank / total) * 100))
    if percentile <= 10:
        band = "top 10%"
    elif percentile <= 25:
        band = "top 25%"
    elif percentile >= 90:
        band = "bottom 10%"
    elif percentile >= 75:
        band = "bottom 25%"
    else:
        band = "middle range"
    return (f"#{rank} of {total}", f"{band} for this creator")


def _metric_context(rows: List[Dict], current: Dict) -> str:
    metric_labels = {
        "views": ("Views", _format_compact_number),
        "likes": ("Likes", _format_compact_number),
        "comments": ("Comments", _format_compact_number),
        "duration_sec": ("Duration", _format_duration),
    }
    cards = []
    current_id = current.get("video_id")

    for metric, (label, formatter) in metric_labels.items():
        values = [
            (r, _parse_metric(r.get(metric)))
            for r in rows
        ]
        values = [(r, v) for r, v in values if v is not None]
        if not values:
            continue

        low_row, low_value = min(values, key=lambda item: item[1])
        high_row, high_value = max(values, key=lambda item: item[1])
        ranked = sorted(values, key=lambda item: item[1], reverse=True)
        rank = next(
            (
                idx
                for idx, (row, _) in enumerate(ranked, start=1)
                if row.get("video_id") == current_id
            ),
            None,
        )
        rank_text, band_text = _rank_summary(rank, len(ranked))
        current_value = next(
            (
                value
                for row, value in ranked
                if row.get("video_id") == current_id
            ),
            None,
        )

        label_text = html_lib.escape(label)
        current_text = html_lib.escape(formatter(current_value))
        high_text = html_lib.escape(formatter(high_value))
        low_text = html_lib.escape(formatter(low_value))
        rank_safe = html_lib.escape(rank_text)
        band_safe = html_lib.escape(band_text)
        cards.append(
            f"""
            <div class="metric-card">
              <div class="metric-top">
                <span>{label_text}</span>
                <strong>{current_text}</strong>
              </div>
              <div class="metric-rank">{rank_safe}</div>
              <small>{band_safe} · creator range {low_text} to {high_text}</small>
            </div>
            """
        )

    return "".join(cards) or "<div>No creator-level metrics available.</div>"


@app.get("/", response_class=HTMLResponse)
def index(creator: str | None = None):
    rows = load_rows()
    if not rows:
        return HTMLResponse(
            "<h3>No labels CSV found. Run generate_label_queue.py first.</h3>"
        )

    visible_rows = [
        r for r in rows
        if not creator or r.get("creator_id") == creator
    ]
    current = first_unlabeled(visible_rows)
    labeled = count_labeled(visible_rows)
    total = len(visible_rows)

    if creator and not visible_rows:
        safe_creator = html_lib.escape(creator)
        return HTMLResponse(
            f"<h3>No label rows found for creator: {safe_creator}</h3>"
        )

    if not current:
        html = f"""
        <html>
        <head>
          <title>Labeling Complete</title>
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <link rel="preconnect" href="https://fonts.googleapis.com">
          <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
          <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Work+Sans:wght@400;500;600&display=swap" rel="stylesheet">
          <style>
            :root {{
              --bg: #0b0e12;
              --card: #121825;
              --text: #eaf2ff;
              --muted: #9fb3c8;
              --accent: #21f0c9;
              --border: #243242;
              --shadow: 0 20px 60px rgba(0,0,0,0.45);
            }}
            body {{
              margin: 0;
              font-family: "Work Sans", system-ui, sans-serif;
              color: var(--text);
              background:
                radial-gradient(900px 500px at 8% -10%, #1d2a40 0%, transparent 60%),
                radial-gradient(900px 600px at 100% 0%, #1b2b3a 0%, transparent 55%),
                linear-gradient(180deg, #0b0e12 0%, #090c10 100%);
              display: grid;
              place-items: center;
              min-height: 100vh;
              padding: 20px;
            }}
            .card {{
              background: linear-gradient(180deg, var(--card) 0%, #0f1520 100%);
              border: 1px solid var(--border);
              border-radius: 18px;
              box-shadow: var(--shadow);
              padding: 28px 30px;
              max-width: 520px;
              text-align: center;
            }}
            .title {{
              font-family: "Space Grotesk", sans-serif;
              font-size: 30px;
              font-weight: 700;
              margin-bottom: 6px;
            }}
            .sub {{
              color: var(--muted);
              font-size: 14px;
              margin-bottom: 18px;
            }}
            .stat {{
              display: inline-flex;
              align-items: center;
              gap: 10px;
              padding: 8px 12px;
              border-radius: 999px;
              border: 1px solid var(--border);
              background: #0f1522;
              color: var(--muted);
              font-size: 13px;
            }}
            .accent {{
              color: var(--accent);
              font-weight: 700;
            }}
          </style>
        </head>
        <body>
          <div class="card">
            <div class="title">All Done</div>
            <div class="sub">Every video in the queue has been labeled.</div>
            <div class="stat">
              Labeled <span class="accent">{labeled}</span> / {total}
            </div>
          </div>
        </body>
        </html>
        """
        return HTMLResponse(html)

    url = current.get("tiktok_url", "")
    url_attr = html_lib.escape(url, quote=True)
    creator_text = html_lib.escape(str(current.get("creator_id", "")))
    creator_attr = html_lib.escape(str(current.get("creator_id", "")), quote=True)
    video_id_text = html_lib.escape(str(current.get("video_id", "")))
    video_id_attr = html_lib.escape(str(current.get("video_id", "")), quote=True)
    creator_query_attr = html_lib.escape(str(creator or ""), quote=True)

    def options_html(name, options, selected=""):
        opts = []
        name_attr = html_lib.escape(name, quote=True)
        for o in options:
            sel = "selected" if o == selected else ""
            value = html_lib.escape(o, quote=True)
            label = html_lib.escape(o)
            opts.append(f"<option value='{value}' {sel}>{label}</option>")
        return (
            f"<select name='{name_attr}' required>"
            + "".join(opts)
            + "</select>"
        )

    def fmt_date(v):
        if not v:
            return "-"
        s = str(v)
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:]}"
        return s

    creator_metric_rows = [
        r for r in rows
        if r.get("creator_id") == current.get("creator_id")
    ]
    views_text = html_lib.escape(_format_compact_number(current.get("views")))
    likes_text = html_lib.escape(_format_compact_number(current.get("likes")))
    comments_text = html_lib.escape(_format_compact_number(current.get("comments")))
    duration_text = html_lib.escape(_format_duration(current.get("duration_sec")))
    posted_text = html_lib.escape(fmt_date(current.get("posted_at")))
    metric_context = _metric_context(creator_metric_rows, current)

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
          letter-spacing: 0;
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
          border-radius: 8px;
          box-shadow: var(--shadow);
          padding: 16px;
        }}
        .meta {{
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 1px 14px;
          color: var(--muted);
          font-size: 13px;
          overflow: hidden;
          border: 1px solid #26364a;
          border-radius: 8px;
        }}
        .meta div {{
          display: grid;
          gap: 2px;
          background: rgba(15, 26, 42, 0.55);
          padding: 9px 10px;
        }}
        .meta span {{
          color: var(--muted);
          font-size: 11px;
          text-transform: uppercase;
        }}
        .meta strong {{
          color: var(--text);
          font-size: 15px;
          font-weight: 700;
          overflow-wrap: anywhere;
        }}
        .metric-context {{
          display: grid;
          gap: 8px;
          margin-top: 4px;
        }}
        .metric-card {{
          display: grid;
          gap: 5px;
          background: linear-gradient(180deg, #0f1a2a 0%, #0c1522 100%);
          border: 1px solid #26364a;
          border-radius: 8px;
          padding: 10px;
          color: var(--muted);
          font-size: 12px;
        }}
        .metric-top {{
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 12px;
        }}
        .metric-top span {{
          color: var(--muted);
          font-size: 11px;
          text-transform: uppercase;
        }}
        .metric-context strong {{
          color: var(--text);
          font-size: 17px;
        }}
        .metric-rank {{
          color: #cfe4ff;
          font-size: 13px;
          font-weight: 600;
        }}
        .metric-context small {{
          color: var(--muted);
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
          border-radius: 8px;
          border: 1px solid var(--border);
          background: #0f151d;
          color: var(--text);
          font-size: 14px;
        }}
        .btn {{
          padding: 12px 14px;
          border-radius: 8px;
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
            <div class="sub">Creator: {creator_text} • Video {labeled + 1} of {total}</div>
          </div>
          <div class="progress">
            <div class="pill">{labeled} labeled • {total - labeled} remaining</div>
            <div class="bar"><span></span></div>
          </div>
        </div>

        <div class="layout">
          <div class="card">
            <div class="embed">
              <blockquote class="tiktok-embed" cite="{url_attr}" data-video-id="{video_id_attr}" style="max-width: 605px;min-width: 325px;" >
                <section></section>
              </blockquote>
              <script async src="https://www.tiktok.com/embed.js"></script>
            </div>
            <a class="link" href="{url_attr}" target="_blank" rel="noopener">Open in TikTok</a>
          </div>

          <div class="card">
            <div class="form">
              <div class="section-title">Video Metrics</div>
              <div class="meta">
                <div><span>Views</span><strong>{views_text}</strong></div>
                <div><span>Likes</span><strong>{likes_text}</strong></div>
                <div><span>Comments</span><strong>{comments_text}</strong></div>
                <div><span>Duration</span><strong>{duration_text}</strong></div>
                <div><span>Posted</span><strong>{posted_text}</strong></div>
                <div><span>Video ID</span><strong>{video_id_text}</strong></div>
              </div>
              <div class="section-title">Creator-Relative Metrics</div>
              <div class="metric-context">
                {metric_context}
              </div>
              <div class="section-title">Labels</div>
              <form action="/label" method="post">
                <input type="hidden" name="creator_id" value="{creator_attr}">
                <input type="hidden" name="video_id" value="{video_id_attr}">
                <input type="hidden" name="creator_filter" value="{creator_query_attr}">
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
    creator_filter: str = Form(""),
):
    if format_label not in FORMAT_LABELS:
        raise HTTPException(status_code=400, detail="Invalid format_label")
    if performance_label not in PERFORMANCE_LABELS:
        raise HTTPException(status_code=400, detail="Invalid performance_label")

    rows = load_rows()
    updated = False
    for r in rows:
        if r.get("creator_id") == creator_id and r.get("video_id") == video_id:
            r["format_label"] = format_label
            r["performance_label"] = performance_label
            updated = True
            break
    if not updated:
        raise HTTPException(status_code=404, detail="Video not found in label queue")
    save_rows(rows)
    redirect = f"/?creator={quote(creator_filter)}" if creator_filter else "/"
    return RedirectResponse(redirect, status_code=303)
