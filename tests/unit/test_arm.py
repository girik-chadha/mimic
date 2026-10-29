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
    transport = transport_class(limits)
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
    assert scalars["MIN_MS_PER_DEGREE"] == limits.min_ms_per_degree
    assert scalars["MAX_GRIP_HOLD_MS"] == limits.max_grip_hold_s * 1000
    assert scalars["MAX_LINE"] == arm.MAX_LINE_CHARS


def test_committed_limits_are_locked():
    limits = committed_limits()
    assert limits.placeholder is True
    assert limits.limits_checked_by_hand is False


@pytest.mark.parametrize("change, problem", [
    ({"home_deg": {"base": 150, "shoulder": 90, "elbow": 90, "grip": 90}}, "home base 150 is outside"),
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
    assert a.status() == {"base": 90, "shoulder": 90, "elbow": 90, "grip": 90}
    assert t.sent == ["?"]


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


def test_grip_close_and_open_keep_the_rest_of_the_pose():
    a, t = fake_arm()
    a.home()
    a.move({"base": 100, "shoulder": 80, "elbow": 95, "grip": 90}, ms=800)
    a.grip(closed=True)
    a.grip(closed=False)
    assert t.sent[-2:] == ["J 100 80 95 120 500", "J 100 80 95 90 500"]


def test_stop_sends_S_and_forgets_the_pose():
    a, t = fake_arm()
    a.home()
    a.stop()
    assert t.sent == ["H", "S"]
    a.move({"base": 95, "shoulder": 90, "elbow": 90, "grip": 90}, ms=800)
    assert t.sent[-2:] == ["?", "J 95 90 90 90 800"]  # asks where it stopped before moving again


# ---------- refused before anything is sent ----------

@pytest.mark.parametrize("pose, problem", [
    ({"base": 111, "shoulder": 90, "elbow": 90, "grip": 90}, "base 111 degrees is outside its range 70-110"),
    ({"base": 90, "shoulder": 69, "elbow": 90, "grip": 90}, "shoulder 69 degrees is outside"),
    ({"base": 90, "shoulder": 90, "elbow": 90, "grip": 121}, "grip 121 degrees is outside its range 90-120"),
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
    a, t = fake_arm(committed_limits(), RealLookingTransport)
    with pytest.raises(ArmError, match="placeholder numbers: measure the arm and set placeholder to false"):
        a.home()
    with pytest.raises(ArmError, match="placeholder"):
        a.move({"base": 95, "shoulder": 90, "elbow": 90, "grip": 90}, ms=800)
    with pytest.raises(ArmError, match="placeholder"):
        a.grip(closed=True)
    assert "H" not in t.sent and not any(line.startswith("J") for line in t.sent)


def test_real_arm_still_obeys_stop_and_status_with_placeholder_limits():
    a, t = fake_arm(committed_limits(), RealLookingTransport)
    a.stop()
    a.status()
    assert t.sent == ["S", "?"]


def test_real_arm_needs_limits_checked_by_hand_too():
    measured_only = dataclasses.replace(committed_limits(), placeholder=False)
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
