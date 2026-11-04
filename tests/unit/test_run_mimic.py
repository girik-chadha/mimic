"""run_mimic.py refuses, before opening the serial port or the camera, while anything is not ready."""
import dataclasses
import importlib.util

import pytest
from test_kinematics import linear_grid, wide_limits

from mimic import arm, camera, config, kinematics, scene

spec = importlib.util.spec_from_file_location("run_mimic", config.ROOT / "scripts/run_mimic.py")
run_mimic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_mimic)


def test_refuses_before_opening_anything(monkeypatch, capsys):
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("serial bridge opened"))
    monkeypatch.setattr(camera, "capture", lambda *a, **k: pytest.fail("camera used"))
    assert run_mimic.main([]) == 1  # the committed layout and pose grid are still placeholders
    assert "STOPPED before opening anything" in capsys.readouterr().out
    assert run_mimic.main(["--push"]) == 1
    assert "STOPPED before opening anything" in capsys.readouterr().out


def test_push_mode_without_grip_push_refuses_before_opening_anything(monkeypatch, capsys):
    ready_layout = dataclasses.replace(scene.load_layout(config.LAYOUT_PATH), placeholder=False)
    monkeypatch.setattr(scene, "load_layout", lambda path=config.LAYOUT_PATH: ready_layout)
    monkeypatch.setattr(kinematics, "load_grid", lambda path=kinematics.POSE_GRID_PATH: linear_grid())  # no grip_push
    limits = wide_limits()  # built before load_limits is replaced (wide_limits itself calls it)
    monkeypatch.setattr(arm, "load_limits", lambda path=config.ARM_LIMITS_PATH: limits)
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("serial bridge opened"))
    assert run_mimic.main(["--push"]) == 1
    assert "STOPPED before opening anything: Push mode needs grip_push" in capsys.readouterr().out


def ready(monkeypatch, typed):
    """Everything ready, a fake (disarmed) arm in place of the bridge, and loop.run replaced by a stand-in."""
    from mimic import loop, session
    from mimic.arm import FakeTransport

    limits = wide_limits()
    transport = FakeTransport(limits)  # disarmed, like the firmware at power-on
    monkeypatch.setattr(loop, "load_and_check", lambda push=False: (None, None, limits))
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: transport)
    monkeypatch.setattr(arm.time, "sleep", lambda s: None)
    monkeypatch.setattr(session.time, "sleep", lambda s: None)
    runs = []

    def fake_run(a, capture, layout, grid, limits, push=False):
        runs.append(a.status()["armed"])
        return {"success": True, "attempts": 3, "max_error_cm": 0.4, "cost_usd": 0.05}

    monkeypatch.setattr(loop, "run", fake_run)
    return transport, runs, (lambda q: typed)


def test_run_mimic_enables_after_a_yes_and_disables_at_the_end(monkeypatch, capsys):
    transport, runs, read = ready(monkeypatch, "yes")
    assert run_mimic.main([], read=read) == 0
    assert runs == [True]                                  # the run started with the servos on
    assert transport.sent[:2] == ["?", "E 90 150 90 90"] and transport.sent[-1] == "D"
    assert "Hold the arm" in capsys.readouterr().out


def test_run_mimic_does_not_run_without_a_yes(monkeypatch, capsys):
    transport, runs, read = ready(monkeypatch, "no")
    assert run_mimic.main([], read=read) == 1
    assert runs == [] and transport.sent == ["?"]
    assert "Not enabled: nothing was moved." in capsys.readouterr().out


# ---------- ring mode ----------

def test_ring_mode_with_the_committed_placeholder_ring_refuses_before_opening_anything(monkeypatch, capsys):
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("serial bridge opened"))
    assert run_mimic.main(["--ring"]) == 1
    assert "STOPPED before opening anything: calib/ring.json is still a placeholder" in capsys.readouterr().out


def test_ring_and_push_together_are_refused(capsys):
    assert run_mimic.main(["--ring", "--push"]) == 1
    assert "--ring and --push cannot be used together" in capsys.readouterr().out


def test_ring_mode_asks_before_enabling_runs_the_ring_loop_and_disables(monkeypatch, capsys):
    from mimic import ring_loop, session
    from mimic.arm import FakeTransport

    limits = wide_limits()
    transport = FakeTransport(limits)   # disarmed, like the firmware at power-on
    from mimic import camera, ring
    the_ring = ring.Ring(camera_frames=25)
    monkeypatch.setattr(ring_loop, "load_and_check", lambda: ("layout", the_ring, limits))
    grabs = []
    monkeypatch.setattr(camera, "capture",
                        lambda path, frames=15, name=None, **k: grabs.append((frames, name)) or str(path))
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: transport)
    monkeypatch.setattr(arm.time, "sleep", lambda s: None)
    monkeypatch.setattr(session.time, "sleep", lambda s: None)
    runs, prompts = [], []

    def fake_run(a, capture, layout, ring_used, limits):
        runs.append((layout, ring_used is the_ring, a.status()["armed"]))
        capture("photo.jpg")
        return {"success": True, "attempts": 3, "max_error_cm": 0.4, "cost_usd": 0.05}

    monkeypatch.setattr(ring_loop, "run", fake_run)
    assert run_mimic.main(["--ring"], read=lambda q: (prompts.append(q), "yes")[1]) == 0
    assert runs == [("layout", True, True)]
    assert grabs == [(25, "e2eSoft iVCam")]               # camera_frames and camera_name from ring.json
    assert run_mimic.main(["--ring", "--camera-frames", "40", "--camera-name", "Camo"], read=lambda q: "yes") == 0
    assert grabs[-1] == (40, "Camo")                      # the options win
    assert "pose the arm like the start pose: upper arm vertical, forearm horizontal, claw open" in prompts[0]
    assert transport.sent[-1] == "D" and "Hold the arm" in capsys.readouterr().out


# ---------- simulation mode (run_mimic.py --ring --sim) ----------

@pytest.mark.parametrize("argv, problem", [
    (["--sim", "--photo", "p.jpg"], "--sim runs ring mode: use --ring --sim"),
    (["--ring", "--sim"], "--sim needs --photo"),
    (["--ring", "--photo", "p.jpg"], "--photo only work with --ring --sim"),
    (["--ring", "--sim-miss-deg", "6"], "--sim-miss-deg only work with --ring --sim"),
    (["--ring", "--instruction", "row red, blue"], "--instruction only work with --ring --sim"),
    (["--ring", "--instruction", "stack black, blue, red"], "stacking is simulation-only"),
    (["--instruction", "Stack red"], "stacking is simulation-only"),
])
def test_simulation_options_are_checked_before_anything_runs(monkeypatch, capsys, argv, problem):
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("serial bridge opened"))
    monkeypatch.setattr(run_mimic.sim, "run", lambda *a, **k: pytest.fail("simulation ran"))
    assert run_mimic.main(argv) == 1
    assert problem in capsys.readouterr().out


def test_ring_sim_runs_the_simulation_and_opens_no_serial_port(monkeypatch, capsys):
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("serial bridge opened"))
    monkeypatch.setattr(camera, "capture", lambda *a, **k: pytest.fail("camera used"))
    calls = []
    monkeypatch.setattr(run_mimic.sim, "run", lambda *a: calls.append(a) or {
        "success": True, "moves": [{"result": "PASS"}] * 3, "attempts": 3, "max_error_cm": 0.0, "cost_usd": 0.013,
        "out_dir": "runs/x_simulation"})
    assert run_mimic.main(["--ring", "--sim", "--photo", "p.jpg", "--instruction", "row red, blue",
                           "--sim-miss-deg", "6"]) == 0
    assert calls == [("p.jpg", "row red, blue", 6.0)]
    assert "SIMULATION SUCCESS: 3 of 3 dice placed" in capsys.readouterr().out


def test_a_refused_simulation_says_so(monkeypatch, capsys):
    def refuse(*a):
        raise run_mimic.sim.SimError("'purple' matches 0 of the dice Spark named")
    monkeypatch.setattr(run_mimic.sim, "run", refuse)
    assert run_mimic.main(["--ring", "--sim", "--photo", "p.jpg", "--instruction", "row purple"]) == 1
    assert "STOPPED (simulation): 'purple' matches 0" in capsys.readouterr().out
