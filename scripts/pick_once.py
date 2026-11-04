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

from mimic import arm as arm_module
from mimic import session
from mimic.arm import JOINTS, ArmError
from mimic.pickdrop import COMMAND_S, ENABLE_NOTE, HOLD_MARGIN_S, PickError, plan, run_steps

LATENCY_CALLS = 20

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
