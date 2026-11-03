"""grip_test.py with the fake arm and typed-in answers. calib/arm_limits.json is never changed."""
import dataclasses
import importlib.util
import json
import shutil

import pytest

from mimic import arm, config
from mimic.arm import Arm, FakeTransport

spec = importlib.util.spec_from_file_location("grip_test", config.ROOT / "scripts/grip_test.py")
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)


def proposed_limits(**changes):
    limits = dataclasses.replace(arm.load_limits(config.ARM_LIMITS_PATH),
                                 ranges_deg={"base": (45, 135), "shoulder": (80, 120), "elbow": (60, 120)},
                                 placeholder=False, limits_checked_by_hand=True)
    return dataclasses.replace(limits, **changes)


def answers(record=None, *typed):
    queue = list(typed)

    def read(question):
        if record is not None:
            record.append(question)
        return queue.pop(0)
    return read


def test_lift_can_use_several_joints_up_to_10_degrees():
    assert gt.parse_lift("shoulder -4") == {"shoulder": -4}
    assert gt.parse_lift("shoulder -4 elbow +3") == {"shoulder": -4, "elbow": 3}
    for bad, problem in [("shoulder -6 elbow +5", "11 degrees"), ("grip +2", "not base"), ("shoulder", "pairs")]:
        with pytest.raises(gt.GripTestError, match=problem):
            gt.parse_lift(bad)


def test_closes_in_2_degree_steps_opening_between_and_lifts_at_the_first_yes():
    limits = proposed_limits()
    t = FakeTransport(limits, enabled=True)
    a = Arm(t, limits)
    sent_when_placing = []

    def read(question):
        if question.startswith("Place the cube"):
            sent_when_placing.append(len(t.sent))
        return typed.pop(0)

    typed = ["", "n", "n", "y", "y"]
    angle = gt.run_test(a, {"shoulder": -4}, read, say=lambda s: None, sleep=lambda s: None)
    assert angle == 96
    grips = [int(line.split()[4]) for line in t.sent if line.startswith("J")]
    assert grips[:7] == [90, 92, 90, 94, 90, 96, 90]          # close a step, open, ask, next step
    assert "J 90 86 90 96 400" in t.sent                       # the lift, grip closed at 96
    n = sent_when_placing[0]                                   # while you place the cube:
    assert t.sent[n - 1] == "J 90 90 90 90 400"                # the last thing sent was "open, at home"
    assert t.sent[n] == "J 90 90 90 92 400"                    # and the next is the first step, after Enter


def test_not_held_through_the_lift_gives_no_angle():
    limits = proposed_limits()
    a = Arm(FakeTransport(limits, enabled=True), limits)
    assert gt.run_test(a, {"shoulder": -4}, answers(None, "", "y", "n"), say=lambda s: None,
                       sleep=lambda s: None) is None


def test_lift_test_refused_if_the_grip_would_be_closed_too_long():
    limits = proposed_limits(max_grip_hold_s=2.5)   # allowed 2.0 s; close + lift + 2 s hold + lower is more
    t = FakeTransport(limits, enabled=True)
    with pytest.raises(gt.GripTestError, match="would hold the grip closed for .* Nothing was moved"):
        gt.run_test(Arm(t, limits), {"shoulder": -4}, answers(None, "", "y"), say=lambda s: None,
                    sleep=lambda s: None)
    assert not any(line.startswith("J 90 86") for line in t.sent)


def test_main_records_the_angle_in_the_proposal_only(monkeypatch, tmp_path):
    proposed = tmp_path / "arm_limits_proposed.json"
    data = json.loads(config.ARM_LIMITS_PATH.read_text())
    data.update(placeholder=False, limits_checked_by_hand=True,
                ranges_deg={"base": [45, 135], "shoulder": [80, 120], "elbow": [60, 120]})
    proposed.write_text(json.dumps(data))
    before = config.ARM_LIMITS_PATH.read_bytes()
    code = gt.main(["--lift", "shoulder -4", "--dry-run"], read=answers(None, "yes", "", "n", "y", "y"),
                   sleep=lambda s: None, say=lambda s: None, proposed_path=proposed)
    assert code == 0
    assert json.loads(proposed.read_text())["grip_hold_deg"] == 94
    assert config.ARM_LIMITS_PATH.read_bytes() == before


def test_main_needs_the_proposal_first(tmp_path):
    said = []
    assert gt.main(["--lift", "shoulder -4", "--dry-run"], read=answers(), say=said.append,
                   proposed_path=tmp_path / "missing.json") == 1
    assert "Run scripts/first_moves.py first" in said[0]


def test_real_arm_refused_while_locked(monkeypatch, tmp_path):
    real_load = arm.load_limits
    locked = dataclasses.replace(real_load(config.ARM_LIMITS_PATH), placeholder=True)
    monkeypatch.setattr(arm, "load_limits",
                        lambda path=config.ARM_LIMITS_PATH: locked if path == config.ARM_LIMITS_PATH else real_load(path))
    monkeypatch.setattr(arm, "SerialBridgeTransport", lambda *a, **k: pytest.fail("bridge opened while locked"))
    proposed = tmp_path / "p.json"
    shutil.copy(config.ARM_LIMITS_PATH, proposed)
    said = []
    assert gt.main(["--lift", "shoulder -4"], read=answers(), say=said.append, proposed_path=proposed) == 1
    assert "placeholder" in said[0]
