"""run_mimic.py refuses, before opening the serial port or the camera, while anything is not ready."""
import importlib.util

import pytest

from mimic import arm, camera, config

spec = importlib.util.spec_from_file_location("run_mimic", config.ROOT / "scripts/run_mimic.py")
run_mimic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_mimic)


def test_refuses_before_opening_anything(monkeypatch, capsys):
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("serial bridge opened"))
    monkeypatch.setattr(camera, "capture", lambda *a, **k: pytest.fail("camera used"))
    assert run_mimic.main() == 1  # the committed layout, pose grid and arm limits are all still placeholders
    assert "STOPPED before opening anything" in capsys.readouterr().out
