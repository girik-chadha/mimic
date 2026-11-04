"""arm.py tests with a fake transport. No arm, no USB, no network."""
import dataclasses
import json
import re

import pytest

from mimic import arm, config
from mimic.arm import Arm, ArmError, FakeTransport

HEADER = config.ROOT / "firmware" / "mimic_arm" / "arm_logic.h"


def committed_limits():
    return arm.load_limits(config.ARM_LIMITS_PATH)


def ready_limits():
    """The committed numbers, as if measured and checked by hand."""
    return dataclasses.replace(committed_limits(), placeholder=False, limits_checked_by_hand=True)


class RealLookingTransport(FakeTransport):
    """Answers like the fake, but says it is a real arm, so the placeholder guard applies."""
    is_real = True


def fake_arm(limits=None, transport_class=FakeTransport):
    limits = limits or committed_limits()
    transport = transport_class(limits, enabled=True)  # already switched on (E)
    return Arm(transport, limits), transport


# ---------- arm_limits.json and the firmware agree ----------

def header_values():
    text = HEADER.read_text()
    scalars = {name: int(value) for name, value in re.findall(r"#define (\w+) (-?\d+)", text)}
    arrays = {name: [int(v) for v in values.split(",")]
              for name, values in re.findall(r"const int (\w+)\[\d+\] = \{([^}]*)\};", text)}
    return scalars, arrays


def test_firmware_limits_match_arm_limits_json():
    limits = committed_limits()
    scalars, arrays = header_values()
    assert scalars["LIMITS_ARE_PLACEHOLDER"] == int(limits.placeholder)
    assert scalars["LIMITS_CHECKED_BY_HAND"] == int(limits.limits_checked_by_hand)
    assert arrays["PINS"] == [limits.pins[j] for j in arm.JOINTS]
    assert arrays["MIN_ANGLE"] == [limits.ranges_deg[j][0] for j in arm.JOINTS[:3]]
    assert arrays["MAX_ANGLE"] == [limits.ranges_deg[j][1] for j in arm.JOINTS[:3]]
    assert arrays["HOME"] == [limits.home_deg[j] for j in arm.JOINTS]
    assert scalars["GRIP_OPEN"] == limits.grip_open_deg
    assert scalars["GRIP_CLOSED_LIMIT"] == limits.grip_closed_limit_deg
    assert arrays["GRIP_RANGE"] == list(limits.grip_range_deg)
    assert scalars["MIN_MS_PER_DEGREE"] == limits.min_ms_per_degree
    assert scalars["MAX_GRIP_HOLD_MS"] == limits.max_grip_hold_s * 1000
    assert scalars["MAX_LINE"] == arm.MAX_LINE_CHARS


def locked_limits():
    """The committed numbers with the lock on: the guard tests below must not depend on the committed state."""
    return dataclasses.replace(committed_limits(), placeholder=True, limits_checked_by_hand=False)


def test_committed_limits_are_unlocked_with_the_kit_ranges_for_the_first_powered_session():
    # Changed on purpose (3 Nov, needs.md C7 step 2): unlocked with the kit's ranges, untested on this arm.
    limits = committed_limits()
    assert limits.placeholder is False and limits.limits_checked_by_hand is True
    assert limits.ranges_deg == {"base": (0, 180), "shoulder": (80, 180), "elbow": (30, 120)}
    # 4 Nov (Girik): a LOWER angle closes this claw; open 90; closed limit 3 PROVISIONAL (not 0: the servo's end stop)
    assert (limits.grip_open_deg, limits.grip_closed_limit_deg) == (90, 3)
    assert limits.grip_range_deg == (0, 140) and limits.grip_close_dir == -1
    assert limits.range_of("grip") == (3, 140)                    # never past the closed limit, so never 0
    assert limits.grip_hold_deg is None                           # unknown until grip_test.py
    assert limits.home_deg == {"base": 90, "shoulder": 90, "elbow": 90, "grip": 90}


@pytest.mark.parametrize("change, problem", [
    ({"home_deg": {"base": 90, "shoulder": 70, "elbow": 90, "grip": 90}}, "home shoulder 70 is outside"),
    ({"grip_closed_limit_deg": 90}, "cannot be the same angle"),
    ({"pins": {"base": 4, "shoulder": 4, "elbow": 18, "grip": 19}}, "4 different pins"),
    ({"ranges_deg": {"base": [110, 70], "shoulder": [70, 110], "elbow": [70, 110]}}, "base range 110-70"),
])
def test_bad_limits_file_is_refused(tmp_path, change, problem):
    data = json.loads(config.ARM_LIMITS_PATH.read_text())
    data.update(change)
    (tmp_path / "arm_limits.json").write_text(json.dumps(data))
    with pytest.raises(ArmError, match=problem):
        arm.load_limits(tmp_path / "arm_limits.json")


# ---------- the exact lines sent ----------

def test_home_sends_H():
    a, t = fake_arm()
    a.home()
    assert t.sent == ["H"]
    assert a.angles == {"base": 90, "shoulder": 90, "elbow": 90, "grip": 90}


def test_status_sends_question_mark_and_reads_the_reply():
    a, t = fake_arm()
    assert a.status() == {"base": 90, "shoulder": 90, "elbow": 90, "grip": 90, "locked": False, "armed": True}
    assert t.sent == ["?"]
    assert a.angles == {"base": 90, "shoulder": 90, "elbow": 90, "grip": 90}


# ---------- disarmed at power-on: E (enable) and D (disable) ----------

def disarmed_arm(limits=None):
    limits = limits or committed_limits()
    t = FakeTransport(limits)   # like the firmware: disarmed at power-on
    return Arm(t, limits), t


def test_a_disarmed_arm_reports_D_and_refuses_moves():
    a, t = disarmed_arm()
    assert a.status() == {"base": 90, "shoulder": 90, "elbow": 90, "grip": 90, "locked": False, "armed": False}
    assert a.angles is None   # the firmware's own numbers: no servo is driven
    with pytest.raises(ArmError, match="not enabled .* ends with D.*Enable it first"):
        a.move({"base": 95, "shoulder": 90, "elbow": 90, "grip": 90}, ms=800)
    assert t.sent == ["?", "?"]


def test_enable_sends_E_waits_for_all_four_servos_and_knows_the_pose():
    a, t = disarmed_arm()
    waits = []
    assert a.enable(sleep=waits.append) == {"base": 90, "shoulder": 90, "elbow": 90, "grip": 90}
    assert t.sent == ["E 90 90 90 90", "?"] and waits == [arm.ENABLE_WAIT_S] and arm.ENABLE_WAIT_S > 1.2
    a.move({"base": 95, "shoulder": 90, "elbow": 90, "grip": 90}, ms=800)
    assert t.sent[-1] == "J 95 90 90 90 800"


def start_pose_file(tmp_path, monkeypatch, **angles):
    pose = dict({"base": 90, "shoulder": 120, "elbow": 90, "grip": 90}, **angles)
    path = tmp_path / "start_pose.json"
    path.write_text(json.dumps(dict(pose, note="test")))
    monkeypatch.setattr(arm, "START_POSE_PATH", path)
    return pose


def test_enable_with_no_angles_starts_at_the_start_pose(tmp_path, monkeypatch):
    pose = start_pose_file(tmp_path, monkeypatch)
    a, t = disarmed_arm()
    assert a.enable(sleep=lambda s: None) == pose and t.sent[0] == "E 90 120 90 90"


def test_a_missing_start_pose_falls_back_to_90_with_a_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(arm, "START_POSE_PATH", tmp_path / "missing.json")
    said = []
    assert arm.load_start_pose(say=said.append) == {"base": 90, "shoulder": 90, "elbow": 90, "grip": 90}
    assert said[0].startswith("WARNING:") and "falls back to 90 for every joint" in said[0]


def test_a_wrong_start_pose_file_is_refused_not_guessed(tmp_path, monkeypatch):
    path = tmp_path / "start_pose.json"
    path.write_text(json.dumps({"base": 90, "shoulder": "S0", "elbow": 90, "grip": 90}))
    with pytest.raises(ArmError, match="needs a whole-number angle for shoulder"):
        arm.load_start_pose(path)


def test_the_committed_start_pose_is_90_everywhere_until_s0_is_in():
    assert arm.load_start_pose(say=None) == {"base": 90, "shoulder": 90, "elbow": 90, "grip": 90}
    assert "PROVISIONAL shoulder" in json.loads(arm.START_POSE_PATH.read_text())["note"]


def test_enable_with_angles_sends_them_and_refuses_out_of_range_ones():
    a, t = disarmed_arm()
    pose = {"base": 100, "shoulder": 85, "elbow": 70, "grip": 85}
    assert a.enable(pose, sleep=lambda s: None) == pose
    assert t.sent[0] == "E 100 85 70 85"
    b, u = disarmed_arm()
    with pytest.raises(ArmError, match="enable: shoulder 70 degrees is outside its range 80-180"):
        b.enable({"base": 90, "shoulder": 70, "elbow": 90, "grip": 90}, sleep=lambda s: None)
    assert u.sent == []


def test_enable_twice_and_old_firmware_are_clear_errors():
    a, t = disarmed_arm()
    a.enable(sleep=lambda s: None)
    with pytest.raises(ArmError, match="replied 'ERR already enabled'"):
        a.enable(sleep=lambda s: None)
    b, u = disarmed_arm()
    u.send = lambda line: "ERR bad command"   # the build on the board before 3 Nov
    with pytest.raises(ArmError, match="no E command .* upload the new build"):
        b.enable(sleep=lambda s: None)
    with pytest.raises(ArmError, match="no D command .* upload the new build"):
        b.disable()


def test_disable_sends_D_and_forgets_the_pose():
    a, t = fake_arm()
    a.home()
    a.disable()
    assert t.sent == ["H", "D"] and a.angles is None and not t.enabled
    with pytest.raises(ArmError, match="not enabled"):
        a.home()


def test_a_restart_while_enabled_says_so_with_the_boot_reason():
    a, t = fake_arm()
    a.home()
    t.enabled, t.boot_reason = False, "brownout"    # the board restarted: it comes back disarmed
    with pytest.raises(ArmError, match=r"ERR not enabled.*board restarted: it last started \(brownout\) 12 s ago"):
        a.move({"base": 95, "shoulder": 90, "elbow": 90, "grip": 90}, ms=800)
    assert t.sent[-1] == "B" and a.angles is None


# ---------- a locked firmware ("?" ends with L) ----------

def test_locked_status_is_reported_and_not_kept_as_a_pose():
    limits = committed_limits()
    a = Arm(FakeTransport(limits, locked=True), limits)
    assert a.status() == {"base": 90, "shoulder": 90, "elbow": 90, "grip": 90, "locked": True, "armed": False}
    assert a.angles is None  # the firmware's own numbers, not where the servos really are


def test_move_refuses_when_the_firmware_says_it_is_locked():
    # e.g. arm_limits.json was unlocked but the firmware was not uploaded again
    limits = ready_limits()
    t = FakeTransport(limits, locked=True)
    a = Arm(t, limits)
    with pytest.raises(ArmError, match="firmware is locked"):
        a.move({"base": 95, "shoulder": 90, "elbow": 90, "grip": 90}, ms=800)
    with pytest.raises(ArmError, match="firmware is locked"):
        a.grip(closed=True)
    assert t.sent == ["?", "?"]  # it asked, and sent no move


def test_grip_can_go_5_degrees_below_open():
    a, t = fake_arm()
    a.home()
    a.move({"base": 90, "shoulder": 90, "elbow": 90, "grip": 85}, ms=500)   # "grip -5" from the open stop
    assert t.sent[-1] == "J 90 90 90 85 500"


def test_the_closed_limit_alone_sets_the_closing_direction():
    down = committed_limits()                                                   # limit 20, below open
    up = dataclasses.replace(down, grip_closed_limit_deg=140)                   # limit on the other side of open
    assert (down.grip_close_dir, down.range_of("grip")) == (-1, (3, 140))
    assert (up.grip_close_dir, up.range_of("grip")) == (1, (0, 140))


def write_limits(tmp_path, **changes):
    data = json.loads(config.ARM_LIMITS_PATH.read_text())
    data.update(changes)
    path = tmp_path / "limits.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("changes, problem", [
    ({"grip_closed_limit_deg": 150}, "closed limit 150 is outside the grip range 0-140"),
    ({"grip_range_deg": [95, 140]}, "include open"),
    ({"grip_hold_deg": 95}, "grip_hold_deg 95 must be on the closing side of open"),    # that would open it
    ({"grip_hold_deg": 1}, "grip_hold_deg 1 must be .* short of the closed limit"),      # past the limit
    ({"grip_hold_deg": 3}, r"grip_hold_deg 3 must be .* short of the closed limit \(3\): never hold at the limit"),
    ({"grip_hold_deg": 90}, "grip_hold_deg 90 must be on the closing side"),
])
def test_bad_grip_settings_are_refused(tmp_path, changes, problem):
    with pytest.raises(ArmError, match=problem):
        arm.load_limits(write_limits(tmp_path, **changes))


def test_good_grip_hold_and_an_older_file_without_a_range(tmp_path):
    assert arm.load_limits(write_limits(tmp_path, grip_hold_deg=45)).grip_hold_deg == 45
    assert arm.load_limits(write_limits(tmp_path, grip_hold_deg=4)).grip_hold_deg == 4     # just short of the limit
    data = json.loads(config.ARM_LIMITS_PATH.read_text())
    del data["grip_range_deg"]             # an older file: the range is just open .. closed limit
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data))
    assert arm.load_limits(path).grip_range_deg == (3, 90)


def test_the_limit_warning_is_only_for_the_closed_limit_itself():
    limits = committed_limits()
    assert "closed limit (3, PROVISIONAL)" in arm.grip_limit_warning(limits, 3)
    assert arm.grip_limit_warning(limits, 4) is None and arm.grip_limit_warning(limits, 90) is None


@pytest.mark.parametrize("reply", ["A 90 90 90 90 X", "A 90 90 90 90 L extra", "B 90 90 90 90"])
def test_odd_status_replies_raise(reply):
    a, t = fake_arm()
    t.send = lambda line: reply
    with pytest.raises(ArmError, match="replied"):
        a.status()


# ---------- B: why the board last started ----------

def test_boot_reason_reads_the_reason_and_seconds():
    a, t = fake_arm()
    assert a.boot_reason() == {"reason": "power-on", "seconds_since_boot": 12}
    t.send = lambda line: "B software reset 42"
    assert a.boot_reason() == {"reason": "software reset", "seconds_since_boot": 42}
    assert t.sent == ["B"]


def test_boot_reason_works_on_a_locked_real_arm():
    limits = dataclasses.replace(committed_limits(), placeholder=True)  # read-only, so the lock does not matter
    t = RealLookingTransport(limits, locked=True, boot_reason="brownout")
    assert Arm(t, limits).boot_reason()["reason"] == "brownout"


def test_old_firmware_without_B_says_upload_the_new_build():
    a, t = fake_arm()
    t.send = lambda line: "ERR bad command"   # what the build on the board before 3 Nov replies
    with pytest.raises(ArmError, match="no B command .* upload the new build"):
        a.boot_reason()


@pytest.mark.parametrize("reply", ["B brownout", "B sleepy 4", "B power-on four", "A 90 90 90 90", "B"])
def test_odd_boot_replies_raise(reply):
    a, t = fake_arm()
    t.send = lambda line: reply
    with pytest.raises(ArmError, match="replied"):
        a.boot_reason()


def test_move_sends_one_J_line():
    a, t = fake_arm()
    a.home()
    a.move({"base": 95, "shoulder": 85, "elbow": 100, "grip": 90}, ms=800)
    assert t.sent == ["H", "J 95 85 100 90 800"]


def test_move_rounds_angles_to_whole_degrees():
    a, t = fake_arm()
    a.home()
    a.move({"base": 94.6, "shoulder": 85.2, "elbow": 99.5, "grip": 90.0}, ms=800)
    assert t.sent[-1] == "J 95 85 100 90 800"


def test_move_with_unknown_pose_asks_first():
    a, t = fake_arm()
    a.move({"base": 95, "shoulder": 90, "elbow": 90, "grip": 90}, ms=800)
    assert t.sent == ["?", "J 95 90 90 90 800"]


def test_grip_close_goes_to_grip_hold_and_open_to_90_keeping_the_rest_of_the_pose():
    a, t = fake_arm(dataclasses.replace(committed_limits(), grip_hold_deg=45))
    a.home()
    a.move({"base": 100, "shoulder": 80, "elbow": 95, "grip": 90}, ms=800)
    a.grip(closed=True)
    a.grip(closed=False)
    assert t.sent[-2:] == ["J 100 80 95 45 500", "J 100 80 95 90 500"]   # 45 degrees * 10 ms = 450, at least 500


def test_grip_close_is_refused_until_grip_hold_is_known():
    a, t = fake_arm()
    a.home()
    with pytest.raises(ArmError, match="grip_hold_deg is not known yet: .* never goes to the closed limit"):
        a.grip(closed=True)
    assert t.sent == ["H"]


def test_stop_sends_S_and_forgets_the_pose():
    a, t = fake_arm()
    a.home()
    a.stop()
    assert t.sent == ["H", "S"]
    a.move({"base": 95, "shoulder": 90, "elbow": 90, "grip": 90}, ms=800)
    assert t.sent[-2:] == ["?", "J 95 90 90 90 800"]  # asks where it stopped before moving again


# ---------- refused before anything is sent ----------

@pytest.mark.parametrize("pose, problem", [
    ({"base": 181, "shoulder": 90, "elbow": 90, "grip": 90}, "base 181 degrees is outside its range 0-180"),
    ({"base": 90, "shoulder": 79, "elbow": 90, "grip": 90}, "shoulder 79 degrees is outside its range 80-180"),
    ({"base": 90, "shoulder": 90, "elbow": 121, "grip": 90}, "elbow 121 degrees is outside its range 30-120"),
    ({"base": 90, "shoulder": 90, "elbow": 90, "grip": 141}, "grip 141 degrees is outside its range 3-140"),
    ({"base": 90, "shoulder": 90, "elbow": 90, "grip": 2}, "grip 2 degrees is outside its range 3-140"),
    ({"base": 90, "shoulder": 90, "elbow": 90, "grip": 0}, "grip 0 degrees is outside its range 3-140"),
    ({"base": 90, "shoulder": 90, "elbow": 90}, "needs exactly these joints"),
    ({"base": 90, "shoulder": 90, "elbow": 90, "grip": 90, "wrist": 90}, "needs exactly these joints"),
])
def test_bad_moves_are_refused_and_nothing_is_sent(pose, problem):
    a, t = fake_arm()
    with pytest.raises(ArmError, match=problem):
        a.move(pose, ms=800)
    assert t.sent == []


def test_too_fast_is_refused():
    a, t = fake_arm()
    a.home()
    with pytest.raises(ArmError, match="Too fast: 20 degrees in 100 ms; the speed limit needs at least 200 ms"):
        a.move({"base": 110, "shoulder": 90, "elbow": 90, "grip": 90}, ms=100)
    assert t.sent == ["H"]


def test_line_over_64_characters_is_never_sent():
    a, t = fake_arm()
    a.home()
    with pytest.raises(ArmError, match="over the 64 the firmware accepts"):
        a.move({"base": 95, "shoulder": 90, "elbow": 90, "grip": 90}, ms=10 ** 60)
    assert t.sent == ["H"]


def test_error_reply_raises():
    a, t = fake_arm()
    t.send = lambda line: "ERR limits not measured"
    with pytest.raises(ArmError, match="replied 'ERR limits not measured' to 'H'"):
        a.home()


def test_garbled_status_reply_raises():
    a, t = fake_arm()
    t.send = lambda line: "A 90 90"
    with pytest.raises(ArmError, match="replied 'A 90 90'"):
        a.status()


# ---------- the placeholder guard on a real arm ----------

def test_real_arm_with_placeholder_limits_refuses_moves():
    a, t = fake_arm(locked_limits(), RealLookingTransport)
    with pytest.raises(ArmError, match="placeholder numbers: measure the arm and set placeholder to false"):
        a.home()
    with pytest.raises(ArmError, match="placeholder"):
        a.move({"base": 95, "shoulder": 90, "elbow": 90, "grip": 90}, ms=800)
    with pytest.raises(ArmError, match="placeholder"):
        a.grip(closed=True)
    assert "H" not in t.sent and not any(line.startswith("J") for line in t.sent)


def test_real_arm_still_obeys_stop_and_status_with_placeholder_limits():
    a, t = fake_arm(locked_limits(), RealLookingTransport)
    a.stop()
    a.status()
    assert t.sent == ["S", "?"]


def test_real_arm_needs_limits_checked_by_hand_too():
    measured_only = dataclasses.replace(locked_limits(), placeholder=False)
    a, t = fake_arm(measured_only, RealLookingTransport)
    with pytest.raises(ArmError, match="not been checked by hand"):
        a.home()
    assert t.sent == []


def test_real_arm_moves_when_both_flags_are_right():
    a, t = fake_arm(ready_limits(), RealLookingTransport)
    a.home()
    a.move({"base": 95, "shoulder": 90, "elbow": 90, "grip": 90}, ms=800)
    assert t.sent == ["H", "J 95 90 90 90 800"]


def test_fake_arm_works_with_placeholder_limits():
    a, t = fake_arm(committed_limits(), FakeTransport)  # tests and dry runs
    a.home()
    assert t.sent == ["H"]
