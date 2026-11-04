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
    assert transport.sent[:2] == ["?", "E 90 90 90 90"] and transport.sent[-1] == "D"
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
    monkeypatch.setattr(ring_loop, "load_and_check", lambda: ("layout", "ring", limits))
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: transport)
    monkeypatch.setattr(arm.time, "sleep", lambda s: None)
    monkeypatch.setattr(session.time, "sleep", lambda s: None)
    runs, prompts = [], []

    def fake_run(a, capture, layout, the_ring, limits):
        runs.append((layout, the_ring, a.status()["armed"]))
        return {"success": True, "attempts": 3, "max_error_cm": 0.4, "cost_usd": 0.05}

    monkeypatch.setattr(ring_loop, "run", fake_run)
    assert run_mimic.main(["--ring"], read=lambda q: (prompts.append(q), "yes")[1]) == 0
    assert runs == [("layout", "ring", True)]
    assert "pose the arm like the start pose: upper arm vertical, forearm horizontal, claw open" in prompts[0]
    assert transport.sent[-1] == "D" and "Hold the arm" in capsys.readouterr().out
