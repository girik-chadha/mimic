"""report.py on a run made by the fake-world loop (see test_loop.py)."""
import importlib.util
import json

import cv2
from test_loop import Rig, diagnosis

from mimic import config, marks

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
    assert "It landed right of the slot." in page
    assert "logged only (set MIMIC_USE_SPARK_FIX=1 to use it)" in page
    assert "agrees with the measurement" in page
    assert "3 of 3 blocks placed within 1.5 cm" in page
    assert "src=\"http" not in page and "<script" not in page        # nothing loaded from outside, no scripts
    assert said == [f"Report: {run_dir / 'report.html'}", f"Summary picture: {run_dir / 'summary.png'}"]


def test_sparks_text_cannot_inject_html(tmp_path, monkeypatch):
    run_dir = a_run_with_one_retry(tmp_path, monkeypatch, explanation='<script>alert("x")</script> slid')
    report.main([str(run_dir)], say=lambda s: None)
    page = (run_dir / "report.html").read_text()
    assert "<script>" not in page
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; slid" in page


def test_a_contradiction_is_highlighted_in_the_report(tmp_path, monkeypatch):
    rig = Rig(tmp_path, monkeypatch)
    rig.world.release_misses = [(2.5, 0.0), (2.5, 0.0)]
    rig.diagnoses = [diagnosis(-1.0, 0.0, seen=(1.0, 0.0))]
    rig.run()
    run_dir = next(rig.out_root.iterdir())
    report.main([str(run_dir)], say=lambda s: None)
    page = (run_dir / "report.html").read_text()
    assert '<tr class="warn"><th>Spark&#x27;s position</th><td>contradicts measurement' in page


def test_summary_picture_puts_the_demo_next_to_the_result(tmp_path, monkeypatch):
    run_dir = a_run_with_one_retry(tmp_path, monkeypatch)
    picture = cv2.imread(str(report.summary_image(run_dir)))
    assert picture.shape[0] == 480 + 64                               # two photos 480 px high, one line of numbers
    assert picture.shape[1] > 2 * 480


def test_not_a_run_folder(tmp_path):
    said = []
    assert report.main([str(tmp_path)], say=said.append) == 1
    assert "is that a run folder" in said[0]


# ---------- ring mode, and the simulation (run_mimic.py --ring --sim) ----------

def test_a_ring_run_is_reported_with_its_miss_along_the_ring(tmp_path, monkeypatch):
    from test_ring_loop import Rig as RingRig
    rig = RingRig(tmp_path, monkeypatch)
    rig.world.miss_deg = {"blue die": 8}
    rig.run()
    run_dir = next(rig.out_root.iterdir())
    assert report.main([str(run_dir)], say=lambda s: None) == 0
    page = (run_dir / "report.html").read_text()
    assert "3 of 3 dice placed within 1.5 cm" in page and page.count('<section class="card">') == 4
    assert "degrees along the ring" in page and "-2.0 degrees from the slot (numeric correction)" in page
    assert "SIMULATED" not in page and (run_dir / "summary.png").exists()


def a_simulation(tmp_path, monkeypatch, instruction=None, miss_deg=0.0):
    from test_sim import Fakes
    fakes = Fakes(tmp_path, monkeypatch)
    fakes.run(instruction, miss_deg)
    return fakes.out_dir()


def test_a_simulation_gets_its_own_report_with_the_banner(tmp_path, monkeypatch):
    run_dir = a_simulation(tmp_path, monkeypatch, miss_deg=6)
    said = []
    assert report.main([str(run_dir)], say=said.append) == 0
    assert not (run_dir / "report.html").exists() and not (run_dir / "summary.png").exists()
    page = (run_dir / "simulation_report.html").read_text()
    assert '<div class="sim-banner">SIMULATED EXECUTION</div>' in page
    assert f"<title>Mimic SIMULATION {run_dir.name}</title>" in page and 'name="description" content="SIMULATION' in page
    assert "3 of 3 dice placed within 1.5 cm" in page and page.count('<section class="card">') == 4
    assert page.count("data:image/jpeg;base64,") == 2 + 4    # the real photo, the simulated start, every check
    assert "The real photo: SAM and Spark ran on it" in page and "RETRY" in page
    assert "into the simulated run" in page and "src=\"http" not in page and "<script" not in page
    picture = run_dir / "simulation_summary.png"
    assert marks.png_texts(picture)["Comment"].startswith("SIMULATION")
    assert (cv2.imread(str(picture))[10, 10] == (40, 40, 200)).all()          # the red banner strip
    assert said == [f"Report: {run_dir / 'simulation_report.html'}", f"Summary picture: {picture}"]


def test_a_stack_report_says_stacking_is_simulation_only(tmp_path, monkeypatch):
    run_dir = a_simulation(tmp_path, monkeypatch, "stack black, blue, red")
    report.main([str(run_dir)], say=lambda s: None)
    page = (run_dir / "simulation_report.html").read_text()
    assert "2 of 2 dice stacked on the black die" in page and "Stacking is simulation only. The real arm cannot stack" in page
