"""report.py on a run made by the fake-world loop (see test_loop.py)."""
import importlib.util
import json

import cv2
from test_loop import Rig, diagnosis

from mimic import config

spec = importlib.util.spec_from_file_location("report", config.ROOT / "scripts/report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def a_run_with_one_retry(tmp_path, monkeypatch, explanation="It landed right of the slot."):
    rig = Rig(tmp_path, monkeypatch)
    rig.world.release_misses = [(2.5, 0.0), (2.5, 0.0)]
    rig.diagnoses = [diagnosis(-1.8, 0.0, explanation=explanation)]
    rig.run()
    return next(rig.out_root.iterdir())


def test_report_shows_every_attempt_with_its_photo_and_decision(tmp_path, monkeypatch):
    run_dir = a_run_with_one_retry(tmp_path, monkeypatch)
    said = []
    assert report.main([str(run_dir)], say=said.append) == 0
    page = (run_dir / "report.html").read_text()
    attempts = [json.loads(line) for line in (run_dir / "attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 4
    assert page.count('<section class="card">') == 4
    assert page.count("data:image/jpeg;base64,") == 2 + 4          # demo, mess and every attempt, embedded
    assert "RETRY" in page and page.count(">PASS<") == 3
    assert "It landed right of the slot." in page and "Spark&#x27;s fix" in page
    assert "3 of 3 blocks placed within 1.5 cm" in page
    assert "src=\"http" not in page and "<script" not in page        # nothing loaded from outside, no scripts
    assert said == [f"Report: {run_dir / 'report.html'}", f"Summary picture: {run_dir / 'summary.png'}"]


def test_sparks_text_cannot_inject_html(tmp_path, monkeypatch):
    run_dir = a_run_with_one_retry(tmp_path, monkeypatch, explanation='<script>alert("x")</script> slid')
    report.main([str(run_dir)], say=lambda s: None)
    page = (run_dir / "report.html").read_text()
    assert "<script>" not in page
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; slid" in page


def test_summary_picture_puts_the_demo_next_to_the_result(tmp_path, monkeypatch):
    run_dir = a_run_with_one_retry(tmp_path, monkeypatch)
    picture = cv2.imread(str(report.summary_image(run_dir)))
    assert picture.shape[0] == 480 + 64                               # two photos 480 px high, one line of numbers
    assert picture.shape[1] > 2 * 480


def test_not_a_run_folder(tmp_path):
    said = []
    assert report.main([str(tmp_path)], say=said.append) == 1
    assert "is that a run folder" in said[0]
