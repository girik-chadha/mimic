"""motion.py: pick and place as planned moves, with the grip hold-time limit. Fake arm, no waiting."""
import dataclasses

import pytest
from test_kinematics import linear_grid, wide_limits

from mimic import motion
from mimic.arm import Arm, FakeTransport
from mimic.motion import MotionError


def fake_arm(limits):
    t = FakeTransport(limits)
    a = Arm(t, limits)
    a.home()
    return a, t


def test_plan_goes_hover_lower_close_lift_travel_lower_release_lift():
    grid, limits = linear_grid(), wide_limits()
    steps, hold_s = motion.plan_pick_and_place(grid, limits, dict(limits.home_deg), (5, 4), (28, 20.7))
    assert [s.label for s in steps] == ["hover over the block", "lower", "close grip", "lift", "travel",
                                        "lower", "release", "lift"]
    assert [s.angles["grip"] for s in steps] == [90, 90, 130, 130, 130, 130, 90, 90]
    assert [round(s.angles["elbow"]) for s in steps] == [120, 100, 100, 120, 120, 100, 100, 120]
    assert hold_s == sum(s.ms for s in steps[2:6]) / 1000   # from "close grip" to "release"
    assert all(s.ms >= motion.MIN_STEP_MS for s in steps)


def test_pick_and_place_prints_the_hold_time_and_sends_every_step():
    grid, limits = linear_grid(), wide_limits()
    a, t = fake_arm(limits)
    said, waits = [], []
    hold_s = motion.pick_and_place(a, grid, limits, (5, 4), (28, 20.7), sleep=waits.append, say=said.append)
    assert f"planned grip hold {hold_s:.1f} s (limit 4.5 s)" in said[0]
    assert len([line for line in t.sent if line.startswith("J")]) == 8
    assert len(waits) == 8 and all(w >= motion.MIN_STEP_MS / 1000 for w in waits)


def test_too_long_a_grip_hold_is_refused_before_anything_moves():
    grid = linear_grid()
    slow = dataclasses.replace(wide_limits(), min_ms_per_degree=40)  # the long travel now takes about 4.4 s
    a, t = fake_arm(slow)
    with pytest.raises(MotionError, match=r"hold the grip closed for .* s, over the 4.5 s allowed"):
        motion.pick_and_place(a, grid, slow, (-1, -1), (37, 25), sleep=lambda s: None, say=lambda s: None)
    assert not any(line.startswith("J") for line in t.sent)


def test_camera_clear_is_a_normal_move_with_the_grip_open():
    grid, limits = linear_grid(), wide_limits()
    a, t = fake_arm(limits)
    motion.go_camera_clear(a, grid, limits, sleep=lambda s: None)
    assert t.sent[-1].startswith("J 20 150 150 90 ")
    assert t.sent.count("H") == 1  # only the home() in fake_arm, not for the photo
