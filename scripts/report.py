"""Turn one run's log into a page you can open or send, and a summary picture for the video.

    python scripts/report.py runs/20261103-142500

Writes, inside that run folder:
  report.html  one self-contained file: the demo and mess photos, then every attempt with its photo,
               measured error, PASS/RETRY/FAIL, the correction chosen, Spark's explanation, cost and time.
               The photos are embedded (shrunk to 640 px), so the file works on its own.
  summary.png  your demo next to Mimic's result, with the run's numbers underneath.
Everything shown comes from that run's own files (summary.json, attempts.jsonl, the photos).

Ring-mode attempts (run_mimic.py --ring) are shown with their miss along the ring and along the radius.
A simulation folder (run_mimic.py --ring --sim; its files start with "simulation_") gets simulation_report.html and
simulation_summary.png instead, both with a SIMULATED EXECUTION banner; the page shows the real photo and the
simulated world, and the picture carries "SIMULATION" in its PNG text too. run_mimic.py --ring --sim runs this itself.
"""
import base64
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from mimic import marks

THUMB_WIDTH = 640
PASS_CM = 1.5
SIM = "simulation_"
SIM_BANNER = "SIMULATED EXECUTION"
SIM_NOTE = ("The arm in this report is simulated on a made-up ring. Perception (SAM 3.1, Muse Spark 1.3) and planning "
            "ran live on the real photo below; simulation_moves.json is the move list the real arm would receive.")
SIM_MARK = "SIMULATION: an output of Mimic's simulation mode (run_mimic.py --ring --sim); the arm was simulated."


def prefix_of(run_dir: Path) -> str:
    """"simulation_" for a simulation's folder, "" for a real run's."""
    return SIM if (run_dir / f"{SIM}summary.json").exists() else ""


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    prefix = prefix_of(run_dir)
    summary = json.loads((run_dir / f"{prefix}summary.json").read_text())
    attempts_file = run_dir / f"{prefix}attempts.jsonl"
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


def _ring_rows(a: dict) -> list:
    """Ring mode (and the simulation): the miss is a distance, split along the ring and along the radius."""
    when = (f"{a['sim_time_s']:.0f} s into the simulated run" if "sim_time_s" in a
            else f"{a.get('time_s', 0):.0f} s into the run")
    rows = [
        ("Measured miss", f"{a['miss_cm']:.2f} cm ({a['angle_miss_deg']:+.1f} degrees along the ring, "
                          f"{a['radial_miss_cm']:+.1f} cm along the radius)"),
        ("Picked at", f"base {a['pick_deg']}"),
        ("Dropped at", f"base {a['drop_deg']} (the slot is at base {a['slot_deg']:g})"),
        ("Claw closed", f"{a.get('claw_closed_s', 0):.2f} s"),
        ("Cost so far", f"${a.get('cost_so_far_usd', 0):.4f}"),
        ("Time", when),
    ]
    if a.get("next_correction_deg") is not None:
        rows.append(("Next aim", f"{a['next_correction_deg']:+.1f} degrees from the slot (numeric correction)"))
    return rows


def _attempt_card(run_dir: Path, a: dict) -> str:
    e = html.escape
    miss = a["miss_cm"]
    if not isinstance(miss, dict):
        return _card(run_dir, a, _ring_rows(a))
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
    return _card(run_dir, a, rows)


def _card(run_dir: Path, a: dict, rows: list) -> str:
    e = html.escape
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
    sim = prefix_of(run_dir) == SIM
    things = "dice" if str(summary.get("mode", "")).startswith("ring") else "blocks"
    placed = sum(1 for m in summary.get("moves", []) if m["result"] == "PASS")
    where = (f"placed within {PASS_CM} cm" if not summary.get("stack") else
             f"stacked on the {summary['target'][0]}" if summary.get("stack_in_place") else "stacked on slot 1")
    headline = (f"{placed} of {len(summary.get('moves', []))} {things} {where}"
                if summary.get("moves") else f"No {things} placed")
    facts = [f"{summary.get('attempts', 0)} attempts",
             f"worst error {summary['max_error_cm']} cm" if summary.get("max_error_cm") is not None else None,
             f"cost ${summary.get('cost_usd', 0):.4f}", f"{summary.get('seconds', 0):.0f} s",
             f"{summary['sim_seconds']:.0f} s of simulated arm time" if sim and "sim_seconds" in summary else None]
    stopped = f'<p class="stopped">Stopped: {e(summary["error"])}</p>' if summary.get("error") else ""
    cards = "".join(_attempt_card(run_dir, a) for a in attempts)
    if sim:
        title = f"Mimic SIMULATION {run_dir.name}"
        banner = (f'<div class="sim-banner">{SIM_BANNER}</div>\n<p class="sim-note">{e(SIM_NOTE)}</p>' +
                  (f'\n<p class="stopped">{e("Stacking is simulation only. " + summary["sim_only"])}</p>'
                   if summary.get("sim_only") else ""))
        pair = [(summary.get("photo_copy", f"{SIM}photo.jpg"), "the real photo",
                 "The real photo: SAM and Spark ran on it (live calls)"),
                (f"{SIM}start.jpg", "the simulated world", "The made-up world before the first carry (simulated)")]
        heading, attempts_heading = "The real photo and the simulated start", "Every simulated camera check"
    else:
        title, banner = f"Mimic run {run_dir.name}", ""
        pair = [("demo.jpg", "the demonstration", "Your demonstration"),
                ("mess.jpg", "the mess", "The mess Mimic started from")]
        heading, attempts_heading = "The goal and the starting mess", "Every attempt"
    figures = "\n".join(f'  <figure>{_img(run_dir, name, alt)}<figcaption>{e(caption)}</figcaption></figure>'
                        for name, alt, caption in pair)
    description = f'<meta name="description" content="{e(SIM_MARK)}">' if sim else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>{description}
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
.sim-banner {{ position: sticky; top: 0; z-index: 1; background: #c62828; color: #ffffff; font-weight: 700;
              letter-spacing: 0.08em; text-align: center; padding: 10px 16px; }}
.sim-note {{ max-width: 1100px; margin: 12px auto 0; padding: 0 16px; color: var(--muted); }}
</style></head>
<body>{banner}<main>
<h1>{e(title)}: {e(headline)}</h1>
<p class="facts">{e(" | ".join(f for f in facts if f))}</p>
{stopped}
<h2>{e(heading)}</h2>
<div class="pair">
{figures}
</div>
<h2>{e(attempts_heading)}</h2>
{cards or "<p>No attempts were made.</p>"}
</main></body></html>
"""


def _fit_height(image, height: int):
    return cv2.resize(image, (round(image.shape[1] * height / image.shape[0]), height))


def summary_image(run_dir: Path) -> Path:
    """Your demo and Mimic's result side by side, with the run's numbers underneath. For a simulation: the real
    photo and the simulated result, under a SIMULATED EXECUTION banner."""
    summary, attempts = load_run(run_dir)
    sim = prefix_of(run_dir) == SIM
    if sim:
        pair = ((summary.get("photo_copy", f"{SIM}photo.jpg"), "The real photo"), (f"{SIM}end.jpg", "Simulated result"))
    else:
        pair = (("demo.jpg", "Your demo"), (attempts[-1]["photo"] if attempts else "mess.jpg", "Mimic's result"))
    height = 480
    panels = []
    for name, label in pair:
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
        things = "dice" if str(summary.get("mode", "")).startswith("ring") else "blocks"
        line = (f"{placed}/{len(moves)} {things} within {PASS_CM} cm  |  {summary.get('attempts', 0)} attempts  |  "
                f"{worst_text}  |  ${summary.get('cost_usd', 0):.2f}  |  " +
                ("simulated" if sim else f"{summary.get('seconds', 0) / 60:.1f} min"))
    cv2.putText(footer, line, (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    picture = np.vstack([top, footer])
    if sim:
        strip = np.full((56, picture.shape[1], 3), (40, 40, 200), np.uint8)
        (tw, _), _ = cv2.getTextSize(SIM_BANNER, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
        cv2.putText(strip, SIM_BANNER, ((strip.shape[1] - tw) // 2, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                    (255, 255, 255), 2, cv2.LINE_AA)
        picture = np.vstack([strip, picture])
    out = run_dir / f"{prefix_of(run_dir)}summary.png"
    cv2.imwrite(str(out), picture)
    if sim:
        marks.mark_png(out, "Comment", SIM_MARK)
    return out


def main(argv=None, say=print) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        say("Usage: python scripts/report.py runs/<time>")
        return 1
    run_dir = Path(argv[0])
    prefix = prefix_of(run_dir)
    if not (run_dir / f"{prefix}summary.json").exists():
        say(f"No summary.json in {run_dir}: is that a run folder?")
        return 1
    page = run_dir / f"{prefix}report.html"
    page.write_text(build_html(run_dir))
    picture = summary_image(run_dir)
    say(f"Report: {page}")
    say(f"Summary picture: {picture}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
