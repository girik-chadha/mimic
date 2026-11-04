"""teach.py with a fake arm: jog, set, save poses, finish. The real pose grid is never touched."""
import dataclasses
import importlib.util
import json
import shutil

import pytest
from test_kinematics import wide_limits

from mimic import arm, config, kinematics
from mimic.arm import Arm, ArmError, FakeTransport

spec = importlib.util.spec_from_file_location("teach", config.ROOT / "scripts/teach.py")
teach = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teach)


@pytest.fixture
def teacher(tmp_path):
    grid_path = tmp_path / "pose_grid.json"
    shutil.copy(kinematics.POSE_GRID_PATH, grid_path)
    limits = wide_limits()
    t = FakeTransport(limits, enabled=True)
    return teach.Teacher(Arm(t, limits), grid_path, ask=lambda q: "yes"), t, grid_path


def test_jog_steps_are_1_2_5_or_10(teacher):
    tt, t, _ = teacher
    assert tt.handle("base +5") == "base now 95"
    assert tt.handle("elbow -10") == "elbow now 80"
    assert tt.handle("base +3") == "refused: jog steps are 1, 2, 5, 10 degrees"
    assert [line for line in t.sent if line.startswith("J")] == ["J 95 90 90 90 400", "J 95 90 80 90 400"]


def test_enable_and_disable_commands(tmp_path, monkeypatch):
    limits = arm.load_limits(config.ARM_LIMITS_PATH)
    t = FakeTransport(limits)                      # disarmed, like the firmware at power-on
    grid_path = tmp_path / "grid.json"
    shutil.copy(kinematics.POSE_GRID_PATH, grid_path)
    monkeypatch.setattr(arm.time, "sleep", lambda s: None)
    tt = teach.Teacher(Arm(t, limits), grid_path, ask=lambda q: "no")
    assert tt.handle("set base 100").startswith("refused: The arm is not enabled")
    assert tt.handle("enable") == "enabled, servos on at base 90, shoulder 150, elbow 90, grip 90"
    assert tt.handle("set base 100") == "base now 100"
    assert tt.handle("disable") == "disabled: the servos are off (limp)"
    assert tt.handle("enable 100 90 90 90") == "enabled, servos on at base 100, shoulder 90, elbow 90, grip 90"
    assert [line for line in t.sent if line[0] in "EDJ"] == ["E 90 150 90 90", "J 100 150 90 90 400", "D",
                                                             "E 100 90 90 90"]


def test_set_goes_through_the_range_and_speed_limits(teacher):
    tt, t, _ = teacher
    assert tt.handle("set base 150") == "base now 150"
    assert t.sent[-1] == "J 150 90 90 90 600"            # 60 degrees * 10 ms
    assert tt.handle("set grip 200").startswith("refused: grip 200 degrees is outside its range 0-140")


def test_saving_poses_and_finishing(teacher):
    tt, _, grid_path = teacher
    for i, (x, y) in enumerate([(0, 0), (30, 0), (30, 24), (0, 24)]):
        tt.handle(f"set base {60 + 10 * i}")
        assert tt.handle(f"save hover {x} {y}").startswith(f"saved hover over ({float(x)}, {float(y)}) cm")
        tt.handle("elbow -10")
        tt.handle(f"save grab {x} {y}")
        tt.handle("elbow +10")
    assert tt.handle("finish").startswith("refused: The pose grid needs grip_open, grip_closed and camera_clear")
    tt.handle("save grip_open")
    tt.handle("set grip 130")
    tt.handle("save grip_closed")
    tt.handle("set grip 90")
    tt.handle("save clear")
    assert tt.handle("finish") == f"finished: {grid_path} is marked as taught"
    data = json.loads(grid_path.read_text())
    assert data["placeholder"] is False
    assert data["grip_open"] == 90 and data["grip_closed"] == 130
    assert data["points"][1] == {"x_cm": 30.0, "y_cm": 0.0, "hover": {"base": 70, "shoulder": 90, "elbow": 90},
                                 "grab": {"base": 70, "shoulder": 90, "elbow": 80}}


def test_saving_the_same_spot_again_replaces_it(teacher):
    tt, _, grid_path = teacher
    tt.handle("save hover 10 5")
    tt.handle("set base 100")
    tt.handle("save hover 10 5")
    points = json.loads(grid_path.read_text())["points"]
    assert len(points) == 1 and points[0]["hover"]["base"] == 100


def test_finish_needs_yes(tmp_path):
    grid_path = tmp_path / "g.json"
    shutil.copy(kinematics.POSE_GRID_PATH, grid_path)
    limits = wide_limits()
    tt = teach.Teacher(Arm(FakeTransport(limits, enabled=True), limits), grid_path, ask=lambda q: "no")
    for x, y in [(0, 0), (30, 0), (30, 24), (0, 24)]:
        tt.handle(f"save hover {x} {y}")
        tt.handle(f"save grab {x} {y}")
    for cmd in ["save grip_open", "save grip_closed", "save clear"]:
        tt.handle(cmd)
    assert tt.handle("finish") == "not finished: the grid is still a placeholder"
    assert json.loads(grid_path.read_text())["placeholder"] is True


def test_dry_run_never_writes_the_real_grid():
    assert teach.DRY_RUN_GRID != kinematics.POSE_GRID_PATH
    assert teach.DRY_RUN_GRID.parent == config.RUNS_DIR


def test_real_arm_refused_while_locked_before_the_bridge_opens(monkeypatch):
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("bridge opened while locked"))
    with pytest.raises(ArmError, match="placeholder"):
        teach.make_arm(dry_run=False, limits=dataclasses.replace(arm.load_limits(config.ARM_LIMITS_PATH),
                                                                 placeholder=True))


def test_setting_the_grip_onto_the_closed_limit_warns(tmp_path):
    grid_path = tmp_path / "pose_grid.json"
    shutil.copy(kinematics.POSE_GRID_PATH, grid_path)
    limits = arm.load_limits(config.ARM_LIMITS_PATH)          # the committed grip: closed limit 3
    tt = teach.Teacher(Arm(FakeTransport(limits, enabled=True), limits), grid_path, ask=lambda q: "no")
    assert tt.handle("set grip 3").startswith("grip now 3\nWARNING: the grip is at its closed limit (3, PROVISIONAL)")
    assert tt.handle("set grip 10") == "grip now 10"
    assert tt.handle("set grip 2").startswith("refused: grip 2 degrees is outside its range 3-140")
