"""Pick a block up and put it down somewhere else, as a short list of planned moves.

    hover over it (grip open) -> lower -> close grip -> lift -> travel -> lower -> release -> lift

Everything is planned before anything moves. The grip servo overheats, and the firmware lets go of a
grip held closed for max_grip_hold_s (5 s), so the planned time from "close grip" to "release" must stay
under that, with 0.5 s to spare for serial delays; otherwise MotionError is raised and nothing is sent.
Pick and place are one call on purpose: a pick on its own would start the grip timer with nothing to end it.
Each move is waited out before the next, because the firmware replies OK as soon as a move starts and a
new move would cut the old one short.
"""
import time
from dataclasses import dataclass

from mimic.arm import JOINTS

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
