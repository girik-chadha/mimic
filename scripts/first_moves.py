"""First powered session, step 1: find each joint's safe range, in small slow steps, with you watching.

    python scripts/first_moves.py --dry-run          practise with the fake arm
    python scripts/first_moves.py                    the real arm (needs.md C7: unlocked with the kit's range)
    python scripts/first_moves.py --max-delta 60     allow bigger steps away from home (default 45)

For base, then shoulder, then elbow (the others stay at home), it tries +5, -5, +10, -10, ... degrees from
home: never past the range in calib/arm_limits.json, and never more than --max-delta from home (so by
default the base stays within 45-135). Ranges can be lopsided: the shoulder may only have 10 degrees below
home. Every move is slow (at least 60 ms per degree and 1.5 s), goes through Arm.move, and returns home
afterwards. After each move you are asked "smooth, no buzzing, no strain? y/n". A direction stops at its
first n. s at any prompt, or Ctrl+C, sends S (stop) and exits without writing anything.

At the end it writes calib/arm_limits_proposed.json: the largest step you approved in each direction,
minus 5 degrees. It never edits calib/arm_limits.json. A completed session is the "checked by hand" step
(needs.md C7), so the proposal has limits_checked_by_hand true.
"""
import argparse
import json
import sys
import time

from mimic import arm as arm_module
from mimic import config, session
from mimic.arm import ArmError

ARM_JOINTS = ("base", "shoulder", "elbow")   # the grip is checked by scripts/grip_test.py
STEP_DEG = 5
SHRINK_DEG = 5
DEFAULT_MAX_DELTA = 45
SLOW_MS_PER_DEGREE = 60
MIN_MOVE_MS = 1500
PROPOSED_PATH = config.ROOT / "calib" / "arm_limits_proposed.json"
DRY_RUN_PROPOSED_PATH = config.RUNS_DIR / "arm_limits_proposed_dryrun.json"
QUESTION = "smooth, no buzzing, no strain? y/n (s = stop): "


def slow_move(arm, pose: dict, sleep) -> None:
    current = arm.known_pose()
    change = max(abs(round(pose[j]) - round(current[j])) for j in pose)
    ms = max(MIN_MOVE_MS, change * SLOW_MS_PER_DEGREE)
    arm.move(pose, ms)
    sleep(ms / 1000 + 0.2)


def deltas(home: int, edge: int, max_delta: int) -> list[int]:
    """The steps to try in one direction: 5, 10, 15 ... up to the range edge or max_delta, whichever is nearer."""
    room = min(abs(edge - home), max_delta)
    steps = list(range(STEP_DEG, room + 1, STEP_DEG))
    if room % STEP_DEG and room > 0:
        steps.append(room)  # the last, smaller step right up to the edge
    return steps


def yes_or_no(ask) -> bool:
    while True:
        answer = ask(QUESTION)
        if answer in ("y", "n"):
            return answer == "y"


def explore(arm, limits, max_delta: int, ask, say=print, sleep=time.sleep) -> dict:
    """Returns {joint: {"plus": approved degrees, "minus": approved degrees}}."""
    home = dict(limits.home_deg)
    slow_move(arm, home, sleep)
    approved = {}
    for joint in ARM_JOINTS:
        lo, hi = limits.range_of(joint)
        todo = {"plus": deltas(home[joint], hi, max_delta), "minus": deltas(home[joint], lo, max_delta)}
        approved[joint] = {"plus": 0, "minus": 0}
        say(f"\n{joint}: home {home[joint]}, trying up to +{max(todo['plus'], default=0)} / "
            f"-{max(todo['minus'], default=0)} degrees")
        while todo["plus"] or todo["minus"]:
            for direction, sign in (("plus", 1), ("minus", -1)):
                if not todo[direction]:
                    continue
                delta = todo[direction].pop(0)
                pose = dict(home, **{joint: home[joint] + sign * delta})
                say(f"  {joint} {'+' if sign > 0 else '-'}{delta} (to {pose[joint]})")
                slow_move(arm, pose, sleep)
                fine = yes_or_no(ask)
                slow_move(arm, home, sleep)
                if fine:
                    approved[joint][direction] = delta
                else:
                    todo[direction] = []  # this direction stops at its first n
                    say(f"  {joint} {direction}: stopped at {delta}; approved {approved[joint][direction]}")
    return approved


def proposal(limits_path, limits, approved: dict) -> dict:
    with open(limits_path) as f:
        data = json.load(f)
    home = limits.home_deg
    data["ranges_deg"] = {j: [home[j] - max(0, approved[j]["minus"] - SHRINK_DEG),
                              home[j] + max(0, approved[j]["plus"] - SHRINK_DEG)] for j in ARM_JOINTS}
    data["placeholder"] = False
    data["limits_checked_by_hand"] = True  # a completed first_moves session is the check (needs.md C7)
    data["approved_deg"] = approved
    data["note"] = (f"PROPOSED by scripts/first_moves.py on {time.strftime('%Y-%m-%d %H:%M')}: the largest step "
                    f"approved in each direction, minus {SHRINK_DEG} degrees. NOT ACTIVE: copy the ranges into "
                    "calib/arm_limits.json AND firmware/mimic_arm/arm_logic.h, then upload (needs.md C7).")
    return data


def main(argv=None, read=input, sleep=time.sleep, say=print, proposed_path=None) -> int:
    parser = argparse.ArgumentParser(description="Find each joint's safe range, in small slow steps.")
    parser.add_argument("--dry-run", action="store_true", help="the fake arm; writes runs/arm_limits_proposed_dryrun.json")
    parser.add_argument("--max-delta", type=int, default=DEFAULT_MAX_DELTA, help="most degrees from home (default 45)")
    args = parser.parse_args(argv)
    if not STEP_DEG <= args.max_delta <= 90:
        say("--max-delta must be between 5 and 90.")
        return 1
    out = proposed_path or (DRY_RUN_PROPOSED_PATH if args.dry_run else PROPOSED_PATH)
    if out == config.ARM_LIMITS_PATH:
        say("Refusing to write calib/arm_limits.json.")
        return 1
    limits = arm_module.load_limits()
    try:
        arm = session.connect(args.dry_run, limits, echo=say)
    except ArmError as error:
        say(f"STOPPED: {error}")
        return 1

    def body():
        approved = explore(arm, limits, args.max_delta, lambda q: session.ask(q, read), say, sleep)
        data = proposal(config.ARM_LIMITS_PATH, limits, approved)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2) + "\n")
        say("\nProposed ranges (approved minus 5 degrees):")
        for joint in ARM_JOINTS:
            lo, hi = data["ranges_deg"][joint]
            warn = "  <- no safe range: check this joint" if lo == hi else ""
            say(f"  {joint}: {lo}-{hi}{warn}")
        say(f"Written to {out}. calib/arm_limits.json was not changed.")

    return session.guarded(arm, body, say)


if __name__ == "__main__":
    sys.exit(main())
