"""web/app.py, the demo page's server: photo and goal refusals, and a run streamed to the page (fake simulation).

A real local server on a free port; no network beyond 127.0.0.1, no API calls, no arm.
"""
import base64
import importlib.util
import json
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from mimic import config

spec = importlib.util.spec_from_file_location("web_app", config.ROOT / "web/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def data_url(data: bytes, kind="image/jpeg") -> str:
    return f"data:{kind};base64," + base64.b64encode(data).decode()


def jpeg() -> bytes:
    return cv2.imencode(".jpg", np.full((20, 30, 3), 120, np.uint8))[1].tobytes()


# ---------- photo refusals ----------

def test_a_jpeg_and_a_png_are_accepted():
    assert app.photo_bytes(data_url(jpeg()))[1] == ".jpg"
    png = cv2.imencode(".png", np.zeros((5, 5, 3), np.uint8))[1].tobytes()
    assert app.photo_bytes(data_url(png, "image/png"))[1] == ".png"


@pytest.mark.parametrize("url, problem", [
    (data_url(b"hello, I am a text file", "text/plain"), "Only JPEG or PNG photos.*it says text/plain"),
    (data_url(b"\x00\x00\x00\x18ftypheic" + b"\x00" * 40, "image/heic"), "Only JPEG or PNG.*HEIC"),
    (data_url(b"hello", "image/jpeg"), "Only JPEG or PNG"),                    # a stated type is not trusted
    (data_url(b"\xff\xd8\xff" + b"\x00" * 50), "cannot be decoded"),
    ("data:image/jpeg;base64,@@@@", "could not be read"),
    ("not a data url", "Send a photo file"),
])
def test_anything_but_a_real_jpeg_or_png_is_refused(url, problem):
    with pytest.raises(app.Refused, match=problem):
        app.photo_bytes(url)


def test_an_oversized_photo_is_refused(monkeypatch):
    monkeypatch.setattr(app, "MAX_PHOTO_BYTES", 100)
    with pytest.raises(app.Refused, match="the limit is"):
        app.photo_bytes(data_url(jpeg()))


# ---------- the server ----------

@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "UPLOADS", tmp_path / "uploads")
    httpd = app.ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def post(base, body):
    request = urllib.request.Request(base + "/api/run", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_the_page_loads_nothing_from_the_internet(server):
    page = urllib.request.urlopen(server + "/", timeout=5).read().decode()
    assert "Expected simulation" in page and "https://" not in page and "http://" not in page
    font = urllib.request.urlopen(server + "/fonts/archivo-latin-wdth-normal.woff2", timeout=5)
    assert font.headers["Content-Type"] == "font/woff2"
    for bad in ("/fonts/../app.py", "/api/runs/../../.env", "/api/runs/20261104-212832_simulation/../x"):
        with pytest.raises(urllib.error.HTTPError, match="404"):
            urllib.request.urlopen(server + bad, timeout=5)


def test_a_text_file_or_a_bad_goal_is_refused_and_nothing_runs(server, monkeypatch):
    monkeypatch.setattr(app.sim, "run", lambda *a, **k: pytest.fail("the simulation ran"))
    assert post(server, {"photo": data_url(b"not an image", "text/plain"), "goal": ""}) == (
        400, {"field": "photo", "error": "Only JPEG or PNG photos can be used; this file is not one (it says "
                                         "text/plain). A phone's HEIC photo can be exported as JPEG."})
    code, reply = post(server, {"photo": data_url(jpeg()), "goal": "make a tower"})
    assert code == 400 and reply["field"] == "goal" and "Goals that work" in reply["error"]


def test_a_run_streams_its_lines_then_the_result(server, monkeypatch, tmp_path):
    def fake_run(photo, goal, miss, say):
        say("Perception on the photo")
        say("[simulated]   lower to 178: base 72, shoulder 178, grip 90 (50 ms)")
        return {"out_dir": str(tmp_path / "20261104-230000_simulation"), "success": True, "stack": False,
                "target": ["red die"], "moves": [{"result": "PASS"}], "attempts": 1, "max_error_cm": 0.0,
                "cost_usd": 0.0156, "sim_seconds": 12.0}
    monkeypatch.setattr(app.sim, "run", fake_run)
    code, reply = post(server, {"photo": data_url(jpeg()), "goal": "row red", "use_cache": False})
    assert code == 200
    stream = urllib.request.urlopen(f"{server}/api/events/{reply['job']}", timeout=10).read().decode()
    events = [block for block in stream.split("\n\n") if block.startswith(("data:", "event:"))]
    lines = [json.loads(e[6:]) for e in events if e.startswith("data:")]
    assert [line["simulated"] for line in lines][-2:] == [False, True]
    assert "Every SAM and Spark call is a real, paid call (use cache is off)" in [line["text"] for line in lines]
    done = json.loads(events[-1].split("data: ", 1)[1])
    assert done["ok"] is True and done["video"] == "/api/runs/20261104-230000_simulation/simulation_video.mp4"
    assert list(app.UPLOADS.iterdir())[0].suffix == ".jpg"
