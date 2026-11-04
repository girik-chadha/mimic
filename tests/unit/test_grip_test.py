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


def test_closes_down_from_start_in_2_degree_steps_and_holds_3_degrees_further_closed():
    limits = proposed_limits()               # the committed grip: a lower angle closes, limit 20
    t = FakeTransport(limits, enabled=True)
    a = Arm(t, limits)
    sent_when_placing = []

    def read(question):
        if question.startswith("Place the die"):
            sent_when_placing.append(len(t.sent))
        return typed.pop(0)

    typed = ["", "n", "n", "y", "y"]
    angle = gt.run_test(a, {"shoulder": -4}, read, say=lambda s: None, sleep=lambda s: None, start=50)
    assert angle == 43                                         # held first at 46, then 3 degrees further closed
    grips = [int(line.split()[4]) for line in t.sent if line.startswith("J")]
    assert grips[:7] == [90, 50, 90, 48, 90, 46, 90]          # close a step, open, ask, next step
    assert "J 90 86 90 43 400" in t.sent                       # the lift check, at grip_hold 43
    n = sent_when_placing[0]                                   # while you place the die:
    assert t.sent[n - 1] == "J 90 90 90 90 400"                # the last thing sent was "open, at home"
    assert t.sent[n] == "J 90 90 90 50 1200"                   # then --start, after Enter: 40 degrees * 30 ms


def test_grip_hold_is_never_past_the_closed_limit():
    limits = proposed_limits()
    said = []
    assert gt.hold_angle(limits, 30) == 27
    assert gt.hold_angle(limits, 22, said.append) == 20        # 19 would pass the limit (20)
    assert "would pass the closed limit 20: using 20" in said[0]


def test_start_must_close_the_claw_and_not_be_past_the_limit():
    limits = proposed_limits()
    gt.check_start(limits, 50)
    gt.check_start(limits, 20)
    for bad in (90, 95, 19):
        with pytest.raises(gt.GripTestError, match=f"--start {bad} must be between open \\(90\\) and the closed limit"):
            gt.check_start(limits, bad)


def test_the_same_test_closes_upwards_on_a_claw_whose_limit_is_above_open():
    limits = proposed_limits(grip_closed_limit_deg=140)   # the one setting, on the other side of open
    t = FakeTransport(limits, enabled=True)
    angle = gt.run_test(Arm(t, limits), {"shoulder": -4}, answers(None, "", "n", "y", "y"), say=lambda s: None,
                        sleep=lambda s: None, start=100)
    grips = [int(line.split()[4]) for line in t.sent if line.startswith("J")]
    assert grips[:5] == [90, 100, 90, 102, 90] and angle == 105


def test_not_held_through_the_lift_gives_no_angle():
    limits = proposed_limits()
    a = Arm(FakeTransport(limits, enabled=True), limits)
    assert gt.run_test(a, {"shoulder": -4}, answers(None, "", "y", "n"), say=lambda s: None,
                       sleep=lambda s: None) is None


def test_lift_test_refused_if_the_grip_would_be_closed_too_long():
    # allowed 3.0 s: a closing step (1.2 s + 1.5 s hold) fits, but close + lift + 2 s hold + lower (3.2 s) does not
    limits = proposed_limits(max_grip_hold_s=3.5)
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
    assert json.loads(proposed.read_text())["grip_hold_deg"] == 45   # held at 48 (from --start 50), then 3 closer
    assert config.ARM_LIMITS_PATH.read_bytes() == before


def test_main_uses_the_grip_settings_in_force_not_an_older_proposals(monkeypatch, tmp_path):
    proposed = tmp_path / "arm_limits_proposed.json"
    data = json.loads(config.ARM_LIMITS_PATH.read_text())
    data.update(placeholder=False, limits_checked_by_hand=True, grip_closed_limit_deg=140, grip_range_deg=[90, 140],
                ranges_deg={"base": [45, 135], "shoulder": [80, 120], "elbow": [60, 120]})   # made before 4 Nov
    proposed.write_text(json.dumps(data))
    said = []
    code = gt.main(["--lift", "shoulder -4", "--dry-run", "--start", "60"], read=answers(None, "yes", "", "y", "y"),
                   sleep=lambda s: None, say=said.append, proposed_path=proposed)
    assert code == 0 and json.loads(proposed.read_text())["grip_hold_deg"] == 57   # closed downwards from 60
    assert "  grip 60" in said


def test_a_bad_start_is_refused_before_anything_moves(tmp_path):
    proposed = tmp_path / "arm_limits_proposed.json"
    shutil.copy(config.ARM_LIMITS_PATH, proposed)
    said = []
    assert gt.main(["--lift", "shoulder -4", "--dry-run", "--start", "100"], read=answers(), say=said.append,
                   proposed_path=proposed) == 1
    assert said[0].startswith("STOPPED: --start 100 must be between open (90) and the closed limit (20)")


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
