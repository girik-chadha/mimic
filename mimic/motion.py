"""Pick a block up and put it down somewhere else, as a short list of planned moves.

    hover over it (grip open) -> lower -> close grip -> lift -> travel -> lower -> release -> lift

Everything is planned before anything moves. The grip servo overheats, and the firmware lets go of a
grip held closed for max_grip_hold_s (5 s), so the planned time from "close grip" to "release" must stay
under that, with 0.5 s to spare for serial delays; otherwise MotionError is raised and nothing is sent.
Pick and place are one call on purpose: a pick on its own would start the grip timer with nothing to end it.
Each move is waited out before the next, because the firmware replies OK as soon as a move starts and a
new move would cut the old one short.
"""
import math
import time
from dataclasses import dataclass

from mimic.arm import JOINTS
from mimic.kinematics import KinematicsError

MIN_STEP_MS = 400        # even a tiny move takes this long, so nothing is jerky
SETTLE_S = 0.1           # extra wait after each move
HOLD_MARGIN_S = 0.5      # spare time under max_grip_hold_s for serial delays


class MotionError(RuntimeError):
    pass


@dataclass
class Step:
    label: str
    angles: dict   # base, shoulder, elbow, grip
    ms: int


def _step_ms(previous: dict, angles: dict, limits) -> int:
    """Long enough for the speed limit (the same rounding as Arm.move), and never shorter than MIN_STEP_MS."""
    biggest = max(abs(round(angles[j]) - round(previous[j])) for j in JOINTS)
    return max(MIN_STEP_MS, biggest * limits.min_ms_per_degree)


def _pose(grid, cm, height: str, grip: int) -> dict:
    return dict(grid.to_angles(cm[0], cm[1], height), grip=grip)


def plan_pick_and_place(grid, limits, start: dict, from_cm, to_cm) -> tuple[list[Step], float]:
    """Returns (steps, planned grip hold in seconds). Raises MotionError if the hold would be too long."""
    wanted = [
        ("hover over the block", _pose(grid, from_cm, "hover", grid.grip_open)),
        ("lower", _pose(grid, from_cm, "grab", grid.grip_open)),
        ("close grip", _pose(grid, from_cm, "grab", grid.grip_closed)),        # the grip timer starts here
        ("lift", _pose(grid, from_cm, "hover", grid.grip_closed)),
        ("travel", _pose(grid, to_cm, "hover", grid.grip_closed)),
        ("lower", _pose(grid, to_cm, "grab", grid.grip_closed)),
        ("release", _pose(grid, to_cm, "grab", grid.grip_open)),               # ...and ends here
        ("lift", _pose(grid, to_cm, "hover", grid.grip_open)),
    ]
    steps, previous = [], start
    for label, angles in wanted:
        steps.append(Step(label, angles, _step_ms(previous, angles, limits)))
        previous = angles
    hold_s = sum(s.ms for s in steps[2:6]) / 1000
    allowed = limits.max_grip_hold_s - HOLD_MARGIN_S
    if hold_s > allowed:
        raise MotionError(f"This pick and place would hold the grip closed for {hold_s:.1f} s, over the "
                          f"{allowed:.1f} s allowed (max_grip_hold_s {limits.max_grip_hold_s} s minus "
                          f"{HOLD_MARGIN_S} s). Nothing was moved.")
    return steps, hold_s


def run_steps(arm, steps: list[Step], sleep=time.sleep) -> None:
    for step in steps:
        arm.move(step.angles, step.ms)
        sleep(step.ms / 1000 + SETTLE_S)


def pick_and_place(arm, grid, limits, from_cm, to_cm, sleep=time.sleep, say=print) -> float:
    """Plan, check the grip hold time, then move. Returns the planned hold time in seconds."""
    steps, hold_s = plan_pick_and_place(grid, limits, arm.known_pose(), from_cm, to_cm)
    say(f"  pick ({from_cm[0]:.1f}, {from_cm[1]:.1f}) -> place ({to_cm[0]:.1f}, {to_cm[1]:.1f}) cm: "
        f"planned grip hold {hold_s:.1f} s (limit {limits.max_grip_hold_s - HOLD_MARGIN_S:.1f} s)")
    run_steps(arm, steps, sleep)
    return hold_s


def go_camera_clear(arm, grid, limits, sleep=time.sleep) -> None:
    """Move the arm out of the camera's view (grip open) before a photo. A normal move, not the firmware's H."""
    angles = dict(grid.camera_clear, grip=grid.grip_open)
    run_steps(arm, [Step("camera clear", angles, _step_ms(arm.known_pose(), angles, limits))], sleep)


# ---------- push mode (opt-in: run_mimic.py --push), for when the claw cannot hold a block ----------
#
# The claw, closed to grip_push (2 degrees more open than where its empty tips touch: never the closed limit, because a
# stalled claw overheats), comes down behind the block and slides it in a straight line towards its slot:
#
#     hover behind (open) -> close to grip_push -> lower to grab height -> slide in short steps -> back off -> lift -> open
#
# The block sizes below are provisional (a 2 cm block, a claw about 1-2 cm across) until measured.

PUSH_BEHIND_CM = 3.0     # the claw comes down this far behind the block's centre (half a block + half the claw
                         # + 1 cm for camera error), so it never lands on the block
PUSH_CONTACT_CM = 2.0    # claw centre to block centre while pushing (half a block + half the claw)
PUSH_STEP_CM = 1.5       # the slide is cut into steps this long: one long move turns all the joints together, so
                         # the claw would curve away from the straight line and rise or dip between two grab poses
PUSH_BACK_OFF_CM = 1.0   # after the slide the claw backs off this far before lifting, so it does not drag the block
PUSH_CLEAR_CM = 4.0      # every other block's centre must be at least this far from the push path
MIN_PUSH_CM = 0.5        # nearer than this to the slot: nothing to push


@dataclass
class PushPlan:
    steps: list[Step]
    hold_s: float                 # planned time with the claw closed (grip timer)
    block_end_cm: tuple           # where the block's centre should be when this push ends
    reaches_slot: bool            # False when the grip timer cut the push short: push again after the photo


def distance_to_segment(point, a, b) -> float:
    """Shortest distance (cm) from a point to the straight line from a to b."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / length_sq))
    return math.dist(point, (ax + t * dx, ay + t * dy))


def _along(start, direction, cm) -> tuple[float, float]:
    return (start[0] + direction[0] * cm, start[1] + direction[1] * cm)


def _push_steps(grid, limits, start: dict, claw_start, slide: list, direction, grip: int) -> list[Step]:
    back_off = _along(slide[-1], direction, -PUSH_BACK_OFF_CM)
    wanted = [("hover behind the block", _pose(grid, claw_start, "hover", grid.grip_open)),
              ("close to grip_push", _pose(grid, claw_start, "hover", grip)),           # the grip timer starts
              ("lower behind the block", _pose(grid, claw_start, "grab", grip))]
    wanted += [(f"push {n}/{len(slide)}", _pose(grid, pt, "grab", grip)) for n, pt in enumerate(slide, start=1)]
    wanted += [("back off", _pose(grid, back_off, "grab", grip)),
               ("lift", _pose(grid, back_off, "hover", grip)),
               ("open", _pose(grid, back_off, "hover", grid.grip_open))]               # ...and ends here
    steps, previous = [], start
    for label, angles in wanted:
        steps.append(Step(label, angles, _step_ms(previous, angles, limits)))
        previous = angles
    return steps


def plan_push(grid, limits, start: dict, from_cm, to_cm, others_cm: dict) -> PushPlan:
    """Plan one straight push of the block at from_cm towards to_cm. others_cm: {name: (x, y)} of every other block.

    Raises MotionError, before anything moves, if grip_push is unknown, another block is near the path, or the
    path leaves the taught area. A push longer than the grip timer allows is cut short (reaches_slot False).
    """
    grip = grid.grip_push
    if grip is None:
        raise MotionError("grip_push is not known yet: find it in the grip test (needs.md C7 step 11) and write it "
                          "into calib/pose_grid.json. Push mode cannot run without it.")
    if grip == limits.grip_closed_limit_deg:
        raise MotionError(f"grip_push {grip} is the closed limit: a stalled claw overheats. Nothing was moved.")
    distance = math.dist(from_cm, to_cm)
    if distance < MIN_PUSH_CM:
        raise MotionError(f"The block is {distance:.2f} cm from its slot: nothing to push.")
    direction = ((to_cm[0] - from_cm[0]) / distance, (to_cm[1] - from_cm[1]) / distance)
    claw_start = _along(from_cm, direction, -PUSH_BEHIND_CM)
    claw_end = _along(to_cm, direction, -PUSH_CONTACT_CM)

    for name, pt in others_cm.items():  # the claw and the block sweep from the claw's start to the slot
        gap = distance_to_segment(pt, claw_start, to_cm)
        if gap < PUSH_CLEAR_CM:
            raise MotionError(f"{name} at ({pt[0]:.1f}, {pt[1]:.1f}) cm is {gap:.1f} cm from the push path (it must "
                              f"be at least {PUSH_CLEAR_CM} cm away): pushing would knock it. Nothing was moved.")

    slide_cm = math.dist(claw_start, claw_end)
    count = max(1, math.ceil(slide_cm / PUSH_STEP_CM))
    slide = [_along(claw_start, direction, slide_cm * n / count) for n in range(1, count + 1)]
    allowed = limits.max_grip_hold_s - HOLD_MARGIN_S
    try:
        for used in range(count, 0, -1):  # as far as the grip timer allows
            steps = _push_steps(grid, limits, start, claw_start, slide[:used], direction, grip)
            hold_s = sum(s.ms for s in steps[1:-1]) / 1000   # from "close to grip_push" to "open"
            if hold_s <= allowed:
                block_end = _along(slide[used - 1], direction, PUSH_CONTACT_CM)
                return PushPlan(steps, hold_s, block_end, reaches_slot=used == count)
    except KinematicsError as error:
        raise MotionError(f"The push path leaves the taught area: {error} (scripts/teach_plan.py --push lists the "
                          "extra points to teach.) Nothing was moved.") from None
    raise MotionError(f"Even one {PUSH_STEP_CM} cm push step would hold the claw closed for {hold_s:.1f} s, over the "
                      f"{allowed:.1f} s allowed. Nothing was moved.")


def push(arm, grid, limits, from_cm, to_cm, others_cm: dict, sleep=time.sleep, say=print) -> PushPlan:
    """Plan one push, check it, then move. Returns the plan (where the block should end up)."""
    plan = plan_push(grid, limits, arm.known_pose(), from_cm, to_cm, others_cm)
    say(f"  push ({from_cm[0]:.1f}, {from_cm[1]:.1f}) -> ({plan.block_end_cm[0]:.1f}, {plan.block_end_cm[1]:.1f}) cm"
        f"{'' if plan.reaches_slot else ' (cut short by the grip timer)'}: planned claw-closed time "
        f"{plan.hold_s:.1f} s (limit {limits.max_grip_hold_s - HOLD_MARGIN_S:.1f} s)")
    run_steps(arm, plan.steps, sleep)
    return plan
