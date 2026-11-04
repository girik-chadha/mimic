"""Pick one die up on the ring and put it down at another base angle, with the elbow fixed (4 Nov).

    python scripts/pick_once.py --base-pick 100 --base-drop 120 --dry-run     practise with the fake arm
    python scripts/pick_once.py --base-pick 100 --base-drop 120               the real arm
    python scripts/pick_once.py --base-pick 100 --base-drop 120 --pick-only   pick, lift, hold 2 s, open in the air
    python scripts/pick_once.py --measure-latency                             time 20 status calls (no move)

The elbow servo is dead and unplugged and the forearm is taped, so the claw tip meets the table only on a ring
around the base axis, at one shoulder angle (--shoulder-pick, default 173; a SMALLER shoulder angle lifts it).
The elbow is sent as 90 every time: the firmware puts that on IO2, an empty socket, which is harmless.

    1. open the claw (90) and turn the base to --base-pick
    2. shoulder to pick - approach, then down to the pick angle in 2-degree steps
    3. close to --grip-hold, then lift the shoulder by --lift
    4. swing the base to --base-drop
    5. shoulder back down to the pick angle in 2-degree steps, then open the claw
    6. lift the shoulder away by --lift, and stop there (no H)
    7. "Hold the arm", 2 s, then disable (session.guarded); s or Ctrl+C sends S then D at any time

Everything is planned and checked before you are asked to enable: every angle against calib/arm_limits.json (no
proposal file), the shoulder never past --stop-at, and the time from the start of closing until the open command
reaches the arm must fit max_grip_hold_s minus 0.5 s (the firmware opens a claw held closed for 5 s). That time
counts each move (shoulder at 25 ms per degree, base and claw at the limits' speed), the 0.1 s wait after it, and
--command-s per command for the serial bridge (a guess, 0.05 s, until --measure-latency gives a real number).
"""
import argparse
import math
import sys
import time
from dataclasses import dataclass

from mimic import arm as arm_module
from mimic import session
from mimic.arm import JOINTS, ArmError

SHOULDER_MS_PER_DEGREE = 25   # the shoulder moves gently: at least this slow
STEP_DEG = 2                  # lowering onto the table in steps this small
SETTLE_S = 0.1                # wait after each move, on top of its time
COMMAND_S = 0.05              # per command for the serial bridge: a guess until --measure-latency
HOLD_MARGIN_S = 0.5
PICK_ONLY_HOLD_S = 2.0
BASE_APPROVED = (80, 120)     # the base range proposed from Girik's first_moves (4 Nov, needs.md B4); warn outside
ENABLE_NOTE = "Pose the shoulder near 90 by hand first: enable jumps every joint to 90."
LATENCY_CALLS = 20


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


def measure_latency(arm, say, clock=time.perf_counter, calls: int = LATENCY_CALLS) -> float:
    """Times `calls` status calls (read-only, nothing moves). Returns the slowest, in seconds."""
    times = []
    for _ in range(calls):
        started = clock()
        arm.status()
        times.append(clock() - started)
    worst = max(times)
    suggestion = math.ceil(worst * 100) / 100   # rounded up to the next 0.01 s
    say(f"{calls} status calls: mean {sum(times) / calls * 1000:.1f} ms, slowest {worst * 1000:.1f} ms. "
        f"Use --command-s {suggestion:.2f} instead of the {COMMAND_S} s guess.")
    return worst


def main(argv=None, read=input, sleep=time.sleep, say=print, clock=time.perf_counter) -> int:
    parser = argparse.ArgumentParser(description="Pick one die up on the ring and put it down at another angle.")
    parser.add_argument("--base-pick", type=int, help="base angle of the die (required unless --measure-latency)")
    parser.add_argument("--base-drop", type=int, help="base angle to put it down (required unless --pick-only)")
    parser.add_argument("--shoulder-pick", type=int, default=173, help="shoulder angle at table height (173)")
    parser.add_argument("--grip-hold", type=int, default=23, help="grip angle that holds the die (23)")
    parser.add_argument("--lift", type=int, default=20, help="degrees to lift the shoulder (20)")
    parser.add_argument("--approach", type=int, default=10, help="degrees above the pick to start lowering from (10)")
    parser.add_argument("--stop-at", type=int, default=175, help="the shoulder is never sent past this (175)")
    parser.add_argument("--command-s", type=float, default=COMMAND_S,
                        help=f"seconds per command for the bridge ({COMMAND_S}; measure it with --measure-latency)")
    parser.add_argument("--pick-only", action="store_true", help="pick, lift, hold 2 s, open in the air")
    parser.add_argument("--measure-latency", action="store_true", help="time 20 status calls; nothing moves")
    parser.add_argument("--dry-run", action="store_true", help="the fake arm")
    args = parser.parse_args(argv)
    limits = arm_module.load_limits()   # the active limits: never a proposal file

    if args.measure_latency:
        try:
            arm = session.connect(args.dry_run, limits, echo=None)
        except ArmError as error:
            say(f"STOPPED: {error}")
            return 1
        def timing():
            measure_latency(arm, say, clock)
        return session.guarded(arm, timing, say, sleep)

    if args.base_pick is None or (args.base_drop is None and not args.pick_only):
        say("--base-pick and --base-drop are required (only --base-pick with --pick-only).")
        return 1
    base_drop = args.base_pick if args.pick_only else args.base_drop
    options = dict(base_pick=args.base_pick, base_drop=base_drop, shoulder_pick=args.shoulder_pick,
                   grip_hold=args.grip_hold, lift=args.lift, approach=args.approach, stop_at=args.stop_at,
                   pick_only=args.pick_only, command_s=args.command_s)
    enable_pose = {j: arm_module.ENABLE_DEFAULT_DEG for j in JOINTS}
    try:
        steps, hold_s, warnings = plan(limits, enable_pose, **options)
    except PickError as error:
        say(f"STOPPED: {error}")
        return 1
    for warning in warnings:
        say(warning)
    say(f"Planned: {len(steps)} moves. Claw closed for {hold_s:.2f} s from the start of closing until the open "
        f"command (limit {limits.max_grip_hold_s - HOLD_MARGIN_S:.1f} s; {args.command_s} s per command counted).")
    try:
        arm = session.connect(args.dry_run, limits, echo=say)  # a locked arm is refused here
    except ArmError as error:
        say(f"STOPPED: {error}")
        return 1

    def body():
        pose = session.confirm_and_enable(arm, read, say, sleep, note=ENABLE_NOTE)
        live_steps, live_hold, _ = plan(limits, pose, **options)  # from where the arm really is
        say(f"Claw closed for {live_hold:.2f} s (limit {limits.max_grip_hold_s - HOLD_MARGIN_S:.1f} s).")
        run_steps(arm, live_steps, say, sleep)

    return session.guarded(arm, body, say, sleep)


if __name__ == "__main__":
    sys.exit(main())
