"""The arm program (mimic/job.py, the demo page's download): the zip, --dry-run, and every refusal on the real arm.

Fake SAM and Spark (test_sim.Fakes); the real arm is never opened (the bridge is replaced by a failing stand-in).
"""
import io
import json
import subprocess
import sys
import zipfile

import pytest
from test_ring import fitted, made_up
from test_sim import Fakes

from mimic import arm, job, pickdrop, program
from mimic import ring as ring_module


def a_job(tmp_path, monkeypatch, goal=None):
    fakes = Fakes(tmp_path, monkeypatch)   # also makes sure no serial bridge or camera is opened
    fakes.run(goal)
    return fakes.out_dir()


def test_the_zip_holds_a_standalone_program_a_sketch_the_move_list_and_a_readme(tmp_path, monkeypatch):
    run_dir = a_job(tmp_path, monkeypatch)
    name, data = program.program_zip(run_dir)
    stamp = run_dir.name.removesuffix("_simulation")
    assert name == f"mimic_arm_program_{stamp}.zip"
    z = zipfile.ZipFile(io.BytesIO(data))
    script, sketch = f"mimic_job_{stamp}.py", f"mimic_arm_{stamp}/mimic_arm_{stamp}.ino"
    assert sorted(z.namelist()) == sorted([script, sketch, "moves.json", "README.md"])
    assert z.read("moves.json") == (run_dir / "simulation_moves.json").read_bytes()
    code = z.read(script).decode()
    assert "import mimic" not in code and "from mimic" not in code and "SIMULATION_ANGLES = True" in code
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "job.py").write_text(code)
    dry = subprocess.run([sys.executable, "-I", "job.py", "--dry-run"], cwd=tmp_path / "out", capture_output=True,
                         text=True, timeout=30)
    assert dry.returncode == 0 and "steps checked, nothing was sent" in dry.stdout and "step  1: J " in dry.stdout
    real = subprocess.run([sys.executable, "-I", "job.py", "--port", "COM7"], cwd=tmp_path / "out", capture_output=True,
                          text=True, timeout=30)
    assert real.returncode == 1 and "REFUSED: these are the simulation's angles" in real.stdout
    ino = z.read(sketch).decode()
    assert '#error "Simulation angles: calibrate the ring before uploading"' in ino and "#include <ESP32Servo.h>" in ino
    assert "const Step STEPS[] = {" in ino and "waitForGo();" in ino and "SIMULATION ANGLES" in z.read("README.md").decode()


def test_a_dry_run_replays_the_simulation_on_the_fake_arm(tmp_path, monkeypatch):
    the_job = job.job_from_run(a_job(tmp_path, monkeypatch))
    said = []
    assert job.main(the_job, ["--dry-run"], read=lambda q: "yes", say=said.append) == 0
    sent = [line.split("<-")[0].strip()[3:] for line in said if line.startswith("  -> ")]
    assert sent[:2] == ["?", "E 90 150 90 90"] and sent[-1] == "D"     # status, enable at the start pose ... off
    bases = [int(line.split()[1]) for line in sent if line.startswith("J")]
    assert {72, 22, 88, 38, 106, 54} <= set(bases)                  # the simulation's picks and slots
    assert any("Dry run: the expected simulation's angles" in line for line in said)


def test_the_real_arm_is_refused_while_the_ring_is_a_placeholder(tmp_path, monkeypatch):
    the_job = job.job_from_run(a_job(tmp_path, monkeypatch))
    said = []
    assert job.main(the_job, [], read=lambda q: pytest.fail("asked to enable"), say=said.append) == 1
    assert "REFUSED before opening anything: calib/ring.json is still a placeholder" in said[-1]


def test_real_angles_come_from_the_real_ring_and_outside_80_120_is_refused(tmp_path, monkeypatch):
    the_job = job.job_from_run(a_job(tmp_path, monkeypatch))
    # a real ring through where SAM measured the dice: centre (18, 27), radius 20.3; slots at 100, 105, 110
    real = fitted(made_up([80, 90, 100, 110, 120]), slots_deg=[100, 105, 110], camera_clear={"base": 90, "shoulder": 90})
    real = ring_module.Ring(**{**real.__dict__, "centre_cm": (18.0, 27.0), "radius_cm": 20.3, "offset_deg": -180.0,
                               "scale": 1.0})
    monkeypatch.setattr(ring_module, "load_ring", lambda path=None: real)
    planned = job.carries(the_job, real, dry_run=False)
    assert [(c["name"], c["drop"]) for c in planned] == [("red die", 100), ("black die", 105), ("blue die", 110)]
    assert planned[0]["pick"] == round(real.base_angle_for(*the_job["seen_cm"]["red die"]))
    said = []
    assert job.main(the_job, [], read=lambda q: pytest.fail("asked to enable"), say=said.append) == 1
    assert "Base angles outside the approved 80-120: red die pick at base 73" in said[-1]


@pytest.mark.parametrize("stack_json, problem", [
    (None, "stack.json is missing. Real stacking needs calib/stack.json"),
    ({"placeholder": True}, "is still a placeholder. Real stacking needs"),
    ({"placeholder": False, "pick_base_deg": {"blue die": 100}, "drop_base_deg": 90,
      "drop_shoulder_by_level": [180, 176, 172]}, "is incomplete \\(no pick angle for black die, red die\\)"),
])
def test_a_real_stack_needs_a_measured_stack_json(tmp_path, monkeypatch, stack_json, problem):
    the_job = job.job_from_run(a_job(tmp_path, monkeypatch, "put black on blue, then red on black"))
    path = tmp_path / "stack.json"
    if stack_json is not None:
        path.write_text(json.dumps(stack_json))
    monkeypatch.setattr(job, "STACK_PATH", path)
    said = []
    assert job.main(the_job, [], read=lambda q: pytest.fail("asked to enable"), say=said.append) == 1
    assert "REFUSED before opening anything" in said[-1]
    assert __import__("re").search(problem, said[-1]) and "drop_shoulder_by_level" in said[-1]


def test_a_measured_stack_lets_go_one_level_higher_each_time(tmp_path, monkeypatch):
    the_job = job.job_from_run(a_job(tmp_path, monkeypatch, "stack blue, black, red"))
    path = tmp_path / "stack.json"
    path.write_text(json.dumps({"placeholder": False, "drop_base_deg": 100, "drop_shoulder_by_level": [180, 176, 172],
                                "pick_base_deg": {"blue die": 84, "black die": 110, "red die": 116}}))
    monkeypatch.setattr(job, "STACK_PATH", path)
    planned = job.carries(the_job, ring_module.load_ring(), dry_run=False)
    assert [c["shoulder_drop"] for c in planned] == [176, 172]           # blue stays put: levels 2 and 3
    steps = job.plan_all(arm.load_limits(), arm.load_start_pose(), ring_module.load_ring(), planned)
    opens = [s.angles["shoulder"] for _, carry_steps, _ in steps for s in carry_steps if s.label == "open"]
    assert opens == [176, 172]


def test_a_drop_shoulder_above_the_lift_is_refused():
    limits = arm.load_limits()
    with pytest.raises(pickdrop.PickError, match="drop shoulder 150 must be lower down than the lifted angle 160"):
        pickdrop.plan(limits, arm.load_start_pose(), 100, 110, shoulder_pick=180, grip_hold=20, lift=20, approach=10,
                      stop_at=183, shoulder_drop=150)


def test_a_pile_built_in_place_carries_only_the_dice_on_top(tmp_path, monkeypatch):
    the_job = job.job_from_run(a_job(tmp_path, monkeypatch, "put black on blue, then red on black"))
    assert the_job["stack_in_place"] is True
    said = []
    assert job.main(the_job, ["--dry-run"], read=lambda q: "yes", say=said.append) == 0
    assert [line.strip() for line in said if line.endswith("drop at base 106")] == [
        "black die: pick at base 88, drop at base 106", "red die: pick at base 72, drop at base 106"]
    path = tmp_path / "stack.json"
    path.write_text(json.dumps({"placeholder": False, "drop_base_deg": 100, "drop_shoulder_by_level": [180, 176, 172],
                                "pick_base_deg": {"black die": 110, "red die": 116}}))   # no angle needed for blue
    monkeypatch.setattr(job, "STACK_PATH", path)
    planned = job.carries(the_job, ring_module.load_ring(), dry_run=False)
    assert [(c["name"], c["shoulder_drop"]) for c in planned] == [("black die", 176), ("red die", 172)]
