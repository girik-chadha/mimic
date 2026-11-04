"""Pick and drop on the ring, with the elbow fixed (needs.md A10d): the moves, planned and checked before any is sent.

Used by scripts/pick_once.py, scripts/ring_calibrate.py and the ring-mode loop (mimic/ring_loop.py), so all three
move the same way: the shoulder at 25 ms per degree, the base and claw at the limits' speed, onto the table in
2-degree steps, never past stop_at, every angle inside calib/arm_limits.json, and (while a die is held) the claw
closed for at most max_grip_hold_s minus 0.5 s, counting every move, its wait and the bridge time per command.
A SMALLER shoulder angle lifts the claw. The elbow is always sent as its home number (an empty socket on IO2).
"""
from dataclasses import dataclass

from mimic.arm import JOINTS

SHOULDER_MS_PER_DEGREE = 25   # the shoulder moves gently: at least this slow
STEP_DEG = 2                  # lowering onto the table in steps this small
SETTLE_S = 0.1                # wait after each move, on top of its time
COMMAND_S = 0.05              # per command for the serial bridge: a guess until --measure-latency
HOLD_MARGIN_S = 0.5
PICK_ONLY_HOLD_S = 2.0
BASE_APPROVED = (80, 120)     # the base range proposed from Girik's first_moves (4 Nov, needs.md B4); warn outside


class PickError(RuntimeError):
    pass


@dataclass
class Step:
    label: str
    angles: dict      # base, shoulder, elbow, grip
    ms: int
    holding: bool     # the claw is closed on the die during this step (it counts towards the grip timer)
    wait_s: float = 0.0  # an extra wait after the move (--pick-only's 2 s hold)


def move_ms(previous: dict, angles: dict, limits) -> int:
    """The shoulder at SHOULDER_MS_PER_DEGREE, everything else at the limits' speed; the slowest joint decides."""
    rates = {j: max(SHOULDER_MS_PER_DEGREE if j == "shoulder" else 0, limits.min_ms_per_degree) for j in JOINTS}
    return max(abs(round(angles[j]) - round(previous[j])) * rates[j] for j in JOINTS)


def lowering(pose: dict, start: int, end: int) -> list[dict]:
    """Shoulder poses from start to end in STEP_DEG steps (the last one smaller if needed), end included."""
    direction = 1 if end > start else -1
    angles = list(range(start + direction * STEP_DEG, end, direction * STEP_DEG)) + [end]
    return [dict(pose, shoulder=a) for a in angles]


def plan(limits, start: dict, base_pick: int, base_drop: int, shoulder_pick: int, grip_hold: int, lift: int,
         approach: int, stop_at: int, pick_only: bool = False, command_s: float = COMMAND_S):
    """Returns (steps, hold_s, warnings). Raises PickError, before anything moves, if anything is wrong."""
    if lift <= 0 or approach <= 0:
        raise PickError("--lift and --approach must be above 0 (a smaller shoulder angle lifts the claw).")
    if shoulder_pick > stop_at:
        raise PickError(f"--shoulder-pick {shoulder_pick} is past --stop-at {stop_at}: the shoulder is never sent "
                        "past it.")
    d, open_deg, limit = limits.grip_close_dir, limits.grip_open_deg, limits.grip_closed_limit_deg
    if not ((grip_hold - open_deg) * d > 0 and (grip_hold - limit) * d < 0):
        raise PickError(f"--grip-hold {grip_hold} must close the claw (be on the closing side of open, {open_deg}) and "
                        f"stay short of the closed limit ({limit}).")
    elbow = limits.home_deg["elbow"]       # the dead elbow: always this number (an empty socket on IO2)
    above, lifted = shoulder_pick - approach, shoulder_pick - lift
    p = {"base": base_pick, "shoulder": shoulder_pick, "elbow": elbow, "grip": open_deg}

    wanted = [("open, base to pick", dict(start, base=base_pick, grip=open_deg, elbow=elbow), False),
              ("shoulder above the pick", dict(p, shoulder=above), False)]
    wanted += [(f"lower to {s['shoulder']}", s, False) for s in lowering(p, above, shoulder_pick)]
    wanted += [("close", dict(p, grip=grip_hold), True),
               ("lift", dict(p, shoulder=lifted, grip=grip_hold), True)]
    if pick_only:
        wanted += [("open in the air", dict(p, shoulder=lifted), False)]
    else:
        held = dict(p, base=base_drop, grip=grip_hold)
        wanted += [("swing to drop", dict(held, shoulder=lifted), True)]
        wanted += [(f"lower to {s['shoulder']}", s, True) for s in lowering(held, lifted, shoulder_pick)]
        wanted += [("open", dict(held, grip=open_deg), False),
                   ("lift away", dict(held, grip=open_deg, shoulder=lifted), False)]

    steps = _checked(limits, start, wanted, stop_at)
    if pick_only:
        steps[[s.label for s in steps].index("lift")].wait_s = PICK_ONLY_HOLD_S

    # From the start of closing until the open command has reached the arm.
    hold_s = sum(s.ms / 1000 + SETTLE_S + s.wait_s + command_s for s in steps if s.holding) + command_s
    allowed = limits.max_grip_hold_s - HOLD_MARGIN_S
    if hold_s > allowed:
        swing = abs(base_drop - base_pick)
        raise PickError(f"The claw would be closed for {hold_s:.2f} s, over the {allowed:.1f} s allowed "
                        f"(max_grip_hold_s {limits.max_grip_hold_s} s minus {HOLD_MARGIN_S} s): "
                        f"a {swing}-degree swing is too far. Nothing was moved.")
    warnings = [f"WARNING: base {b} is outside the range you approved ({BASE_APPROVED[0]}-{BASE_APPROVED[1]}, "
                f"needs.md B4); calib/arm_limits.json still allows 0-180."
                for b in sorted({base_pick, base_drop}) if not BASE_APPROVED[0] <= b <= BASE_APPROVED[1]]
    return steps, hold_s, warnings


def run_steps(arm, steps: list[Step], say, sleep) -> None:
    for step in steps:
        say(f"  {step.label}: base {step.angles['base']}, shoulder {step.angles['shoulder']}, "
            f"grip {step.angles['grip']} ({step.ms} ms)")
        arm.move(step.angles, step.ms)
        sleep(step.ms / 1000 + SETTLE_S)
        if step.wait_s:
            say(f"  holding for {step.wait_s:g} s")
            sleep(step.wait_s)


def _checked(limits, start: dict, wanted: list, stop_at: int) -> list[Step]:
    """(label, angles, holding) -> Steps with their move times, refusing any angle past stop_at or the limits."""
    steps, previous = [], start
    for label, angles, holding in wanted:
        if angles["shoulder"] > stop_at:
            raise PickError(f"{label}: shoulder {angles['shoulder']} is past --stop-at {stop_at}.")
        for joint in JOINTS:
            lo, hi = limits.range_of(joint)
            if not lo <= angles[joint] <= hi:
                raise PickError(f"{label}: {joint} {angles[joint]} is outside its range {lo}-{hi} "
                                "(calib/arm_limits.json). Nothing was moved.")
        steps.append(Step(label, angles, move_ms(previous, angles, limits), holding))
        previous = angles
    return steps


def plan_touch(limits, start: dict, base: int, shoulder_table: int, approach: int, stop_at: int) -> list[Step]:
    """Claw open: lift first if it is low, turn the base, then down onto the table in 2-degree steps (ring_calibrate)."""
    elbow, open_deg = limits.home_deg["elbow"], limits.grip_open_deg
    above = shoulder_table - approach
    here = dict(start, elbow=elbow, grip=open_deg)
    wanted = []
    if here["shoulder"] > above:  # nearer the table than "above": lift before turning the base
        wanted.append(("lift before turning", dict(here, shoulder=above), False))
    pose = {"base": base, "shoulder": above, "elbow": elbow, "grip": open_deg}
    wanted.append((f"base to {base}, shoulder above the table", pose, False))
    wanted += [(f"lower to {s['shoulder']}", s, False) for s in lowering(pose, above, shoulder_table)]
    return _checked(limits, start, wanted, stop_at)


def plan_lift(limits, start: dict, lift_to: int, stop_at: int) -> list[Step]:
    return _checked(limits, start, [(f"lift to {lift_to}", dict(start, shoulder=lift_to), False)], stop_at)


def plan_park(limits, start: dict, pose: dict, stop_at: int) -> list[Step]:
    """Claw open, to a parked pose ({"base", "shoulder"}: camera_clear), lifting first if the claw is low."""
    target = dict(start, base=pose["base"], shoulder=pose["shoulder"], elbow=limits.home_deg["elbow"],
                  grip=limits.grip_open_deg)
    wanted = []
    if start["shoulder"] > target["shoulder"] and start["base"] != target["base"]:
        wanted.append(("lift before turning", dict(start, shoulder=target["shoulder"]), False))
    wanted.append(("park (camera clear)", target, False))
    return _checked(limits, start, wanted, stop_at)
