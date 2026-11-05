"""Mimic's demo page: run `python web/app.py`, then open http://127.0.0.1:8000 (this computer only).

The page (web/index.html) sends a photo and a goal. The server runs the simulation (mimic/sim.py: real SAM and
Spark calls on the photo, ring mode's real planner, a SIMULATED arm on a made-up ring) in a background thread, and
streams every line the run prints to the page as it happens (Server-Sent Events). Then the page plays the
"Expected simulation" video. Nothing here talks to the arm: no serial port, no camera.

    GET  /                           the page
    GET  /fonts/<file>               the bundled fonts (Archivo, JetBrains Mono; OFL licences beside them)
    GET  /sample.jpg                 tests/fixtures/photos/dice_sheet.jpg, if it is there
    POST /api/run                    {"photo": "sample" or a data URL, "goal": "...", "use_cache": false}
    GET  /api/events/<job>           the run's lines as they happen, then its result
    GET  /api/runs/<folder>/<file>   a file from a simulation folder (the video is served in byte ranges)
    GET  /api/runs/<folder>/program.zip   the arm program (mimic/job.py): mimic_job_<time>.py, moves.json, README

One run at a time. A photo must be a JPEG or PNG (checked by its first bytes and by decoding it), at most 15 MB.
The goal is free text: Spark reads it with the photo (brain.read_goal) and code checks the reply; before any API
call only its length is checked (sim.check_goal). Only the Python
standard library serves the page; nothing is loaded from the internet.
"""
import argparse
import base64
import binascii
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from mimic import config, job, meta_api, program, ring_loop, sim

WEB = Path(__file__).resolve().parent
PAGE = WEB / "index.html"
FONTS = WEB / "fonts"
SAMPLE = config.ROOT / "tests" / "fixtures" / "photos" / "dice_sheet.jpg"
UPLOADS = config.RUNS_DIR / "web_uploads"
MAX_PHOTO_BYTES = 15 * 1024 * 1024
MAX_BODY_BYTES = MAX_PHOTO_BYTES * 4 // 3 + 64 * 1024     # base64 is a third bigger
HOST, PORT = "127.0.0.1", 8000
RUN_FOLDER = re.compile(r"^\d{8}-\d{6}_simulation$")
RUN_FILE = re.compile(r"^simulation_[a-z0-9_]+\.(mp4|jpg|png|json|html)$")
FONT_FILE = re.compile(r"^[A-Za-z0-9_.-]+\.(woff2|txt)$")
TYPES = {".mp4": "video/mp4", ".jpg": "image/jpeg", ".png": "image/png", ".json": "application/json",
         ".html": "text/html; charset=utf-8", ".woff2": "font/woff2", ".txt": "text/plain; charset=utf-8"}


class Refused(ValueError):
    """A request the page should show as a message: (field, text)."""

    def __init__(self, field: str, text: str):
        super().__init__(text)
        self.field = field


# ---------- checking what the page sends ----------

def photo_bytes(data_url: str) -> tuple[bytes, str]:
    """A data URL from the page -> (the image bytes, ".jpg" or ".png"). Anything that is not a JPEG or PNG is
    refused, whatever its name or stated type says."""
    found = re.match(r"^data:([\w/+.-]*);base64,(.*)$", data_url or "", re.S)
    if not found:
        raise Refused("photo", "Send a photo file (JPEG or PNG).")
    try:
        data = base64.b64decode(found.group(2), validate=True)
    except (binascii.Error, ValueError):
        raise Refused("photo", "The photo could not be read (its data is damaged).") from None
    if len(data) > MAX_PHOTO_BYTES:
        raise Refused("photo", f"The photo is {len(data) / 1e6:.1f} MB; the limit is {MAX_PHOTO_BYTES / 1e6:.0f} MB.")
    if data.startswith(b"\xff\xd8\xff"):
        suffix = ".jpg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        suffix = ".png"
    else:
        stated = found.group(1) or "unknown"
        raise Refused("photo", f"Only JPEG or PNG photos can be used; this file is not one (it says {stated}). "
                               "A phone's HEIC photo can be exported as JPEG.")
    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
        raise Refused("photo", "The photo starts like an image but cannot be decoded.")
    return data, suffix


def check_goal(goal: str) -> str:
    try:
        return sim.check_goal(goal or "")   # free text: Spark reads it with the photo; only its length is checked
    except sim.SimError as error:
        raise Refused("goal", str(error)) from None


# ---------- one run at a time ----------

class Job:
    """One simulation run: the lines it printed so far, and its result once it is done."""

    def __init__(self, job_id: str):
        self.id, self.lines, self.done, self.result = job_id, [], False, None
        self.changed = threading.Condition()

    def say(self, text: str) -> None:
        print(text, flush=True)
        line = {"text": text, "simulated": text.startswith(sim.SIM_TAG.strip()),
                "cost": round(meta_api.cost_so_far(), 4)}
        with self.changed:
            self.lines.append(line)
            self.changed.notify_all()

    def finish(self, result: dict) -> None:
        with self.changed:
            self.result, self.done = result, True
            self.changed.notify_all()


JOBS: dict[str, Job] = {}
RUNNING = threading.Lock()


def result_of(summary: dict) -> dict:
    folder = Path(summary["out_dir"]).name
    return {"ok": True, "folder": folder, "video": f"/api/runs/{folder}/simulation_video.mp4",
            "summary": {k: summary.get(k) for k in ("success", "stack", "target", "moves", "attempts",
                                                    "max_error_cm", "cost_usd", "sim_seconds")},
            "cap_usd": ring_loop.RUN_BUDGET_USD}


def failed_folder(lines: list) -> str | None:
    """The simulation folder a stopped run had made (sim.run prints "Simulation log: <folder>"), so the page can show
    its labelled photo."""
    for line in lines:
        if line["text"].startswith("Simulation log: "):
            name = Path(line["text"].split()[2]).name
            return name if RUN_FOLDER.match(name) else None
    return None


def start_job(photo: Path, goal: str, use_cache: bool, run=None) -> Job:
    """Starts the simulation in a background thread. Refused (Refused("busy")) while another run is going."""
    if not RUNNING.acquire(blocking=False):
        raise Refused("busy", "A run is already going: wait for it to finish.")
    job = Job(time.strftime("%Y%m%d-%H%M%S"))
    JOBS[job.id] = job

    def work():
        cache_before = meta_api.cache_enabled
        meta_api.cache_enabled = use_cache
        try:
            job.say(f"Run started: goal {goal!r}" if goal else "Run started: no goal typed, so the photo's "
                    "left-to-right order is the target")
            job.say("Replies already paid for are reused (use cache is on)" if use_cache
                    else "Every SAM and Spark call is a real, paid call (use cache is off)")
            job.finish(result_of((run or sim.run)(photo, goal or None, 0.0, say=job.say)))
        except Exception as error:
            job.say(f"Stopped: {error}")
            job.finish({"ok": False, "error": str(error), "folder": failed_folder(job.lines),
                        "cost_usd": round(meta_api.cost_so_far(), 4), "cap_usd": ring_loop.RUN_BUDGET_USD})
        finally:
            meta_api.cache_enabled = cache_before
            RUNNING.release()

    threading.Thread(target=work, daemon=True).start()
    return job


# ---------- the web server ----------

class Handler(BaseHTTPRequestHandler):
    server_version = "Mimic/1"

    def log_message(self, format, *args):  # the run's own lines are the log; keep the terminal readable
        pass

    def _send(self, code: int, body: bytes, kind: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, data: dict) -> None:
        self._send(code, json.dumps(data).encode(), "application/json")

    def _file(self, path: Path) -> None:
        """A file, or the byte range the browser asks for (so the video can be played and seeked)."""
        data, kind = path.read_bytes(), TYPES.get(path.suffix.lower(), "application/octet-stream")
        wanted = re.match(r"^bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
        if not wanted or not (wanted.group(1) or wanted.group(2)):
            return self._send(200, data, kind, {"Accept-Ranges": "bytes"})
        if wanted.group(1):
            start = int(wanted.group(1))
            end = min(int(wanted.group(2)) if wanted.group(2) else len(data) - 1, len(data) - 1)
        else:  # the last N bytes
            start, end = max(0, len(data) - int(wanted.group(2))), len(data) - 1
        if start > end or start >= len(data):
            return self._send(416, b"", kind, {"Content-Range": f"bytes */{len(data)}"})
        self._send(206, data[start:end + 1], kind, {"Accept-Ranges": "bytes",
                                                    "Content-Range": f"bytes {start}-{end}/{len(data)}"})

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self._send(200, PAGE.read_bytes(), TYPES[".html"])
        if path == "/sample.jpg":
            return self._file(SAMPLE) if SAMPLE.exists() else self._json(404, {"error": "No sample photo here."})
        parts = path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "fonts" and FONT_FILE.match(parts[1]) and (FONTS / parts[1]).exists():
            return self._file(FONTS / parts[1])
        if len(parts) == 3 and parts[:2] == ["api", "events"] and parts[2] in JOBS:
            return self._events(JOBS[parts[2]])
        if (len(parts) == 4 and parts[:2] == ["api", "runs"] and RUN_FOLDER.match(parts[2])
                and RUN_FILE.match(parts[3]) and (config.RUNS_DIR / parts[2] / parts[3]).exists()):
            return self._file(config.RUNS_DIR / parts[2] / parts[3])
        if (len(parts) == 4 and parts[:2] == ["api", "runs"] and RUN_FOLDER.match(parts[2])
                and parts[3] in ("program.zip", "program.json") and (config.RUNS_DIR / parts[2]).is_dir()):
            try:
                if parts[3] == "program.json":   # the zip's file names and sizes, for the page's card
                    files = program.program_files(config.RUNS_DIR / parts[2])
                    return self._json(200, {"files": [{"name": n, "bytes": len(b)} for n, b in files.items()],
                                            "simulation_angles": b"SIMULATION_ANGLES = True" in next(iter(files.values()))})
                name, data = program.program_zip(config.RUNS_DIR / parts[2])
            except (job.JobError, OSError, KeyError, ValueError) as error:
                return self._json(404, {"error": f"No arm program for this run: {error}"})
            return self._send(200, data, "application/zip",
                              {"Content-Disposition": f'attachment; filename="{name}"'})
        self._json(404, {"error": "Not found."})

    def do_POST(self):
        if self.path != "/api/run":
            return self._json(404, {"error": "Not found."})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                raise Refused("photo", f"The photo is over {MAX_PHOTO_BYTES / 1e6:.0f} MB.")
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                raise Refused("request", "The request could not be read.") from None
            goal = check_goal(str(request.get("goal", "")))
            if request.get("photo") == "sample":
                if not SAMPLE.exists():
                    raise Refused("photo", f"The sample photo is not in this copy of the project ({SAMPLE}).")
                photo = SAMPLE
            else:
                data, suffix = photo_bytes(str(request.get("photo", "")))
                UPLOADS.mkdir(parents=True, exist_ok=True)
                photo = UPLOADS / f"{time.strftime('%Y%m%d-%H%M%S')}_upload{suffix}"
                photo.write_bytes(data)
            job = start_job(photo, goal, bool(request.get("use_cache", False)))
        except Refused as refused:
            return self._json(409 if refused.field == "busy" else 400, {"field": refused.field, "error": str(refused)})
        self._json(200, {"job": job.id, "cap_usd": ring_loop.RUN_BUDGET_USD})

    def _events(self, job: Job) -> None:
        """Every line of the run as it is printed, then "done" with the result."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        sent = 0
        try:
            while True:
                with job.changed:
                    job.changed.wait_for(lambda: len(job.lines) > sent or job.done, timeout=15)
                    new, done = job.lines[sent:], job.done
                for line in new:
                    self.wfile.write(f"data: {json.dumps(line)}\n\n".encode())
                sent += len(new)
                if done and sent >= len(job.lines):
                    self.wfile.write(f"event: done\ndata: {json.dumps(job.result)}\n\n".encode())
                    self.wfile.flush()
                    return
                if not new:
                    self.wfile.write(b": still running\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return  # the page was closed or reloaded; the run goes on


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mimic's demo page (this computer only).")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((HOST, args.port), Handler)
    server.daemon_threads = True
    print(f"Mimic demo page: http://{HOST}:{args.port}  (Ctrl+C to stop). Runs are simulations: no arm is moved.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
