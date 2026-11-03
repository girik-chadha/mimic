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
