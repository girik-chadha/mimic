"""motion.py: pick and place as planned moves, with the grip hold-time limit. Fake arm, no waiting."""
import dataclasses

import pytest
from test_kinematics import linear_grid, wide_limits

from mimic import motion
from mimic.arm import Arm, FakeTransport
from mimic.motion import MotionError


def fake_arm(limits):
    t = FakeTransport(limits, enabled=True)
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


# ---------- push mode (opt-in) ----------

def push_grid(grip_push=126):
    return dataclasses.replace(linear_grid(), grip_push=grip_push)


def test_push_comes_down_behind_the_block_and_slides_it_to_the_slot_with_the_claw_at_grip_push():
    grid, limits = push_grid(), wide_limits()
    plan = motion.plan_push(grid, limits, dict(limits.home_deg), (17, 18), (24, 18), {})  # along a taught row
    labels = [s.label for s in plan.steps]
    assert labels[:3] == ["hover behind the block", "close to grip_push", "lower behind the block"]
    assert labels[-3:] == ["back off", "lift", "open"]
    slides = [s for s in plan.steps if s.label.startswith("push ")]
    assert [s.angles["grip"] for s in plan.steps] == [90] + [126] * (len(plan.steps) - 2) + [90]  # never 140
    assert all(round(s.angles["elbow"]) == 100 for s in slides)            # at grab height the whole slide
    # base = 40 + 3x at the taught points (every 4 cm): the claw starts 3 cm behind the block (x = 14) and ends
    # 2 cm short of the slot (x = 22), both taught points, so their angles are exact
    assert plan.steps[0].angles["base"] == pytest.approx(40 + 3 * 14)
    assert slides[-1].angles["base"] == pytest.approx(40 + 3 * 22)
    assert len(slides) == 6                                                  # 8 cm in steps of at most 1.5 cm
    assert 40 + 3 * 20 < plan.steps[-3].angles["base"] < 40 + 3 * 22        # backed off (to x = 21) before lifting
    assert plan.reaches_slot and plan.block_end_cm == pytest.approx((24, 18))
    assert all(s.ms >= motion.MIN_STEP_MS for s in plan.steps)


def test_a_long_push_is_cut_short_by_the_grip_timer_and_says_where_the_block_will_be():
    grid, limits = push_grid(), wide_limits()
    plan = motion.plan_push(grid, limits, dict(limits.home_deg), (9, 20.5), (32, 20), {})
    assert plan.hold_s <= limits.max_grip_hold_s - motion.HOLD_MARGIN_S
    assert not plan.reaches_slot
    moved = plan.block_end_cm[0] - 9
    assert 7 < moved < 11                                                    # about 9 cm per push
    assert motion.distance_to_segment(plan.block_end_cm, (9, 20.5), (32, 20)) < 1e-9  # still on the line


def test_grip_push_must_be_known_and_never_the_closed_limit():
    limits = wide_limits()
    with pytest.raises(MotionError, match="grip_push is not known yet: find it in the grip test"):
        motion.plan_push(push_grid(None), limits, dict(limits.home_deg), (16, 20), (20, 20), {})
    with pytest.raises(MotionError, match="grip_push 140 is the closed limit"):
        motion.plan_push(push_grid(140), limits, dict(limits.home_deg), (16, 20), (20, 20), {})


def test_a_block_near_the_path_is_never_knocked():
    grid, limits = push_grid(), wide_limits()
    start = dict(limits.home_deg)
    with pytest.raises(MotionError, match=r"blue block at \(18.0, 22.0\) cm is 2.0 cm from the push path"):
        motion.plan_push(grid, limits, start, (12, 20), (26, 20), {"blue block": (18, 22)})
    with pytest.raises(MotionError, match="green block .* 1.4 cm from the push path"):   # behind, where the claw lands
        motion.plan_push(grid, limits, start, (12, 20), (26, 20), {"green block": (8, 21)})
    motion.plan_push(grid, limits, start, (12, 20), (26, 20), {"blue block": (18, 24.5), "green block": (30.5, 20)})


def test_nothing_to_push_and_outside_the_taught_area_are_refused():
    grid, limits = push_grid(), wide_limits()
    start = dict(limits.home_deg)
    with pytest.raises(MotionError, match="0.30 cm from its slot: nothing to push"):
        motion.plan_push(grid, limits, start, (19.7, 20), (20, 20), {})
    with pytest.raises(MotionError, match="leaves the taught area: .*teach_plan.py --push"):
        motion.plan_push(grid, limits, start, (-1, 20), (10, 20), {})  # the claw would land at x = -4: 2 cm out


def test_push_sends_every_step_and_says_the_claw_closed_time():
    grid, limits = push_grid(), wide_limits()
    a, t = fake_arm(limits)
    said, waits = [], []
    plan = motion.push(a, grid, limits, (16, 20), (20, 20), {}, sleep=waits.append, say=said.append)
    assert "planned claw-closed time" in said[0] and "cut short" not in said[0]
    assert len([line for line in t.sent if line.startswith("J")]) == len(plan.steps) == len(waits)
    assert max(int(line.split()[4]) for line in t.sent if line.startswith("J")) == 126   # never the 140 limit
