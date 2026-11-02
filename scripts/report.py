"""Turn one run's log into a page you can open or send, and a summary picture for the video.

    python scripts/report.py runs/20261103-142500

Writes, inside that run folder:
  report.html  one self-contained file: the demo and mess photos, then every attempt with its photo,
               measured error, PASS/RETRY/FAIL, the correction chosen, Spark's explanation, cost and time.
               The photos are embedded (shrunk to 640 px), so the file works on its own.
  summary.png  your demo next to Mimic's result, with the run's numbers underneath.
Everything shown comes from that run's own files (summary.json, attempts.jsonl, the photos).
"""
import base64
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

THUMB_WIDTH = 640
PASS_CM = 1.5


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((run_dir / "summary.json").read_text())
    attempts_file = run_dir / "attempts.jsonl"
    attempts = [json.loads(line) for line in attempts_file.read_text().splitlines()] if attempts_file.exists() else []
    return summary, attempts


def _thumbnail(path: Path) -> str | None:
    image = cv2.imread(str(path))
    if image is None:
        return None
    scale = THUMB_WIDTH / image.shape[1]
    image = cv2.resize(image, (THUMB_WIDTH, round(image.shape[0] * scale)))
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode() if ok else None


def _img(run_dir: Path, name: str, alt: str) -> str:
    data = _thumbnail(run_dir / name)
    if data is None:
        return f'<div class="missing">{html.escape(name)} not found</div>'
    return f'<img src="{data}" alt="{html.escape(alt)}">'


def _cm(pair) -> str:
    return f"({pair[0]:.1f}, {pair[1]:.1f}) cm" if pair else "none"


def _attempt_card(run_dir: Path, a: dict) -> str:
    e = html.escape
    miss = a["miss_cm"]
    rows = [
        ("Measured miss", f"{miss['distance']:.2f} cm (dx {miss['dx']:+.2f}, dy {miss['dy']:+.2f})"),
        ("Aimed at", _cm(a.get("aim_cm"))),
        ("Picked from", _cm(a.get("from_cm"))),
        ("Planned grip hold", f"{a.get('planned_grip_hold_s', 0):.1f} s"),
        ("Cost so far", f"${a.get('cost_so_far_usd', 0):.4f}"),
        ("Time", f"{a.get('time_s', 0):.0f} s into the run"),
    ]
    if a.get("next_correction_cm"):
        source = "Spark's fix" if a.get("correction_source") == "spark" else "numeric correction"
        rows.append(("Next aim", f"{_cm(a['next_correction_cm'])} from the slot ({source})"))
    if a.get("spark_fix"):
        fix = a["spark_fix"]
        rows.append(("Spark's fix", f"({fix['dx_cm']:+.1f}, {fix['dy_cm']:+.1f}) cm: {a.get('spark_fix_note') or ''}"))
    if a.get("consistency"):
        rows.append(("Spark's position", a["consistency"]))
    elif a.get("fallback_reason"):
        rows.append(("Spark", a["fallback_reason"]))
    table = "".join(
        f"<tr{' class=\"warn\"' if str(v).startswith('contradicts') else ''}><th>{e(k)}</th><td>{e(v)}</td></tr>"
        for k, v in rows)
    spark = (f'<p class="spark"><b>Spark ({e(a.get("problem") or "")}):</b> {e(a["explanation"])}</p>'
             if a.get("explanation") else "")
    return f"""
<section class="card">
  {_img(run_dir, a["photo"], f"after try {a['attempt']} of {a['name']}")}
  <div class="body">
    <h3>{e(a['name'])} &rarr; slot {a['move']}, try {a['attempt']}
        <span class="badge {e(a['result'].lower())}">{e(a['result'])}</span></h3>
    <table>{table}</table>
    {spark}
  </div>
</section>"""


def build_html(run_dir: Path) -> str:
    summary, attempts = load_run(run_dir)
    e = html.escape
    placed = sum(1 for m in summary.get("moves", []) if m["result"] == "PASS")
    headline = (f"{placed} of {len(summary.get('moves', []))} blocks placed within {PASS_CM} cm"
                if summary.get("moves") else "No blocks placed")
    facts = [f"{summary.get('attempts', 0)} attempts",
             f"worst error {summary['max_error_cm']} cm" if summary.get("max_error_cm") is not None else None,
             f"cost ${summary.get('cost_usd', 0):.4f}", f"{summary.get('seconds', 0):.0f} s"]
    stopped = f'<p class="stopped">Stopped: {e(summary["error"])}</p>' if summary.get("error") else ""
    cards = "".join(_attempt_card(run_dir, a) for a in attempts)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mimic run {e(run_dir.name)}</title>
<style>
:root {{ --bg: #f7f7f5; --card: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --line: #e3e3de;
        --pass: #1f7a3a; --retry: #9a6a00; --fail: #b3261e; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #161615; --card: #20201e; --text: #ececea; --muted: #a3a39d; --line: #34342f;
          --pass: #5cc77d; --retry: #e0b040; --fail: #f2766c; }} }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 system-ui, sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin: 32px 0 12px; }}
h3 {{ font-size: 16px; margin: 0 0 8px; }}
.facts {{ color: var(--muted); margin: 0; }}
.stopped {{ color: var(--fail); font-weight: 600; }}
.pair {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
.pair figure {{ margin: 0; }} figcaption {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
img {{ width: 100%; height: auto; border-radius: 8px; display: block; }}
.card {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; background: var(--card);
        border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
@media (max-width: 720px) {{ .card {{ grid-template-columns: 1fr; }} }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th {{ text-align: left; color: var(--muted); font-weight: 500; padding: 3px 12px 3px 0; white-space: nowrap; vertical-align: top; }}
td {{ padding: 3px 0; overflow-wrap: anywhere; }}
.badge {{ font-size: 12px; padding: 2px 8px; border-radius: 999px; margin-left: 6px; color: var(--card); }}
.badge.pass {{ background: var(--pass); }} .badge.retry {{ background: var(--retry); }} .badge.fail {{ background: var(--fail); }}
.spark {{ margin: 12px 0 0; padding: 10px 12px; border-left: 3px solid var(--line); }}
tr.warn td, tr.warn th {{ color: var(--fail); font-weight: 600; }}
.missing {{ color: var(--muted); border: 1px dashed var(--line); border-radius: 8px; padding: 24px; }}
</style></head>
<body><main>
<h1>Mimic run {e(run_dir.name)}: {e(headline)}</h1>
<p class="facts">{e(" | ".join(f for f in facts if f))}</p>
{stopped}
<h2>The goal and the starting mess</h2>
<div class="pair">
  <figure>{_img(run_dir, "demo.jpg", "the demonstration")}<figcaption>Your demonstration</figcaption></figure>
  <figure>{_img(run_dir, "mess.jpg", "the mess")}<figcaption>The mess Mimic started from</figcaption></figure>
</div>
<h2>Every attempt</h2>
{cards or "<p>No attempts were made.</p>"}
</main></body></html>
"""


def _fit_height(image, height: int):
    return cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height))


def summary_image(run_dir: Path) -> Path:
    """Your demo and Mimic's result side by side, with the run's numbers underneath."""
    summary, attempts = load_run(run_dir)
    result_photo = attempts[-1]["photo"] if attempts else "mess.jpg"
    height = 480
    panels = []
    for name, label in (("demo.jpg", "Your demo"), (result_photo, "Mimic's result")):
        image = cv2.imread(str(run_dir / name))
        if image is None:
            image = np.full((height, 640, 3), 200, np.uint8)
        image = _fit_height(image, height)
        cv2.rectangle(image, (0, 0), (image.shape[1], 44), (30, 30, 30), -1)
        cv2.putText(image, label, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(image)
    gap = np.full((height, 12, 3), 255, np.uint8)
    top = np.hstack([panels[0], gap, panels[1]])
    footer = np.full((64, top.shape[1], 3), 255, np.uint8)
    moves = summary.get("moves", [])
    placed = sum(1 for m in moves if m["result"] == "PASS")
    if summary.get("error"):
        line = f"Stopped: {summary['error'][:80]}"
    else:
        worst = summary.get("max_error_cm")
        worst_text = f"worst {worst:.1f} cm" if worst is not None else "no placements"
        line = (f"{placed}/{len(moves)} blocks within {PASS_CM} cm  |  {summary.get('attempts', 0)} attempts  |  "
                f"{worst_text}  |  ${summary.get('cost_usd', 0):.2f}  |  {summary.get('seconds', 0) / 60:.1f} min")
    cv2.putText(footer, line, (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    out = run_dir / "summary.png"
    cv2.imwrite(str(out), np.vstack([top, footer]))
    return out


def main(argv=None, say=print) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        say("Usage: python scripts/report.py runs/<time>")
        return 1
    run_dir = Path(argv[0])
    if not (run_dir / "summary.json").exists():
        say(f"No summary.json in {run_dir}: is that a run folder?")
        return 1
    (run_dir / "report.html").write_text(build_html(run_dir))
    picture = summary_image(run_dir)
    say(f"Report: {run_dir / 'report.html'}")
    say(f"Summary picture: {picture}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
