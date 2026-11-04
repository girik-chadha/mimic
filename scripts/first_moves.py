"""First powered session, step 1: find each joint's safe range, in small slow steps, with you watching.

    python scripts/first_moves.py --dry-run          practise with the fake arm
    python scripts/first_moves.py                    the real arm (needs.md C7: unlocked with the kit's range)
    python scripts/first_moves.py --max-delta 60     allow bigger steps away from home (default 45)
    python scripts/first_moves.py --joints shoulder elbow --base-minus 15 --base-plus 35
                                                     only these joints; the base's result typed in, not moved

For base, then shoulder, then elbow (the others stay at home), it tries +5, -5, +10, -10, ... degrees from
home: never past the range in calib/arm_limits.json, and never more than --max-delta from home (so by
default the base stays within 45-135). Ranges can be lopsided: the shoulder may only have 10 degrees below
home. Every move is slow (at least 60 ms per degree and 1.5 s), goes through Arm.move, and returns home
afterwards. After each move you are asked "smooth, no buzzing, no strain? y/n". A direction stops at its
first n. Anything typed while the arm moves is thrown away (session.ask), so answer after each move.
s at any prompt, or Ctrl+C, sends S (stop) then D, and keeps every joint already finished.

After EVERY finished joint it saves calib/arm_limits_proposed.json: the largest step you approved in each
direction, minus 5 degrees. It always merges into the proposal already there: a joint it runs replaces that
joint's entry, the others are kept (and listed at the start, with when they were measured). --base-minus and
--base-plus record a base result you watched earlier, without moving the base. Until all three joints are in,
the proposal is INCOMPLETE (limits_checked_by_hand false, so nothing real can move with it); once they are, it is
the "checked by hand" step (needs.md C7). It never edits calib/arm_limits.json.
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


def room(limits, joint: str, direction: str, max_delta: int) -> int:
    """How far a joint may be tried from home in one direction: the range edge or max_delta, whichever is nearer."""
    lo, hi = limits.range_of(joint)
    home = limits.home_deg[joint]
    return min(abs((hi if direction == "plus" else lo) - home), max_delta)


def explore(arm, limits, max_delta: int, ask, say=print, sleep=time.sleep, joints=ARM_JOINTS,
            on_joint_done=None) -> dict:
    """Returns {joint: {"plus": approved degrees, "minus": approved degrees}} for the joints tried.
    on_joint_done(joint, result) is called as soon as each joint is finished (first_moves saves it then)."""
    home = dict(limits.home_deg)
    slow_move(arm, home, sleep)
    say("Answer each question only after the move has finished: anything typed while the arm moves is ignored.")
    approved = {}
    for joint in joints:
        lo, hi = limits.range_of(joint)
        todo = {"plus": deltas(home[joint], hi, max_delta), "minus": deltas(home[joint], lo, max_delta)}
        result = {"plus": 0, "minus": 0}
        say(f"\n{joint}: home {home[joint]}, trying up to +{max(todo['plus'], default=0)} / "
            f"-{max(todo['minus'], default=0)} degrees")
        try:
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
                        result[direction] = delta
                    else:
                        todo[direction] = []  # this direction stops at its first n
                        say(f"  {joint} {direction}: stopped at {delta}; approved {result[direction]}")
        except (session.StopRequested, KeyboardInterrupt):
            say(f"\n{joint} not finished: approved +{result['plus']} / -{result['minus']} so far (NOT saved). "
                f"Run it again with --joints {joint}.")
            raise
        approved[joint] = result
        if on_joint_done:
            on_joint_done(joint, result)
    return approved


def proposal(limits_path, limits, approved: dict, sources: dict | None = None) -> dict:
    """The proposed limits file. approved may hold only some joints: then it is INCOMPLETE, the missing joints keep
    the active ranges (only so the file still loads), and limits_checked_by_hand stays false."""
    with open(limits_path) as f:
        data = json.load(f)
    home = limits.home_deg
    done = [j for j in ARM_JOINTS if j in approved]
    missing = [j for j in ARM_JOINTS if j not in approved]
    data["ranges_deg"] = {j: ([home[j] - max(0, approved[j]["minus"] - SHRINK_DEG),
                               home[j] + max(0, approved[j]["plus"] - SHRINK_DEG)] if j in approved
                              else list(limits.ranges_deg[j])) for j in ARM_JOINTS}
    data["placeholder"] = False
    data["limits_checked_by_hand"] = not missing  # only a complete set of joints is the check (needs.md C7)
    data["approved_deg"] = {j: approved[j] for j in done}
    data["approved_from"] = {j: (sources or {}).get(j, "") for j in done}
    data["joints_done"] = done
    status = "COMPLETE" if not missing else f"INCOMPLETE: missing {', '.join(missing)} (kit ranges kept for them)"
    data["note"] = (f"PROPOSED by scripts/first_moves.py, saved {time.strftime('%Y-%m-%d %H:%M')}. {status}. "
                    f"Ranges: the largest step approved in each direction, minus {SHRINK_DEG} degrees. NOT ACTIVE: "
                    "when complete, copy the ranges into calib/arm_limits.json AND firmware/mimic_arm/arm_logic.h, "
                    "then upload (needs.md C7).")
    return data


def read_existing(path) -> tuple[dict, dict]:
    """The approvals already in a proposal file (and where each came from), to merge into. Empty if there is none."""
    if not path.exists():
        return {}, {}
    try:
        data = json.loads(path.read_text())
        approved = {j: {"plus": int(r["plus"]), "minus": int(r["minus"])}
                    for j, r in data.get("approved_deg", {}).items() if j in ARM_JOINTS}
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ArmError(f"{path} cannot be read ({error}): fix or delete it, then run again.") from None
    sources = data.get("approved_from", {})
    return approved, {j: sources.get(j, "earlier run") for j in approved}


def write_proposal(out, limits, approved: dict, sources: dict) -> dict:
    data = proposal(config.ARM_LIMITS_PATH, limits, approved, sources)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")
    return data


def say_summary(data: dict, out, say) -> None:
    say("\nProposed ranges (approved minus 5 degrees):")
    for joint in ARM_JOINTS:
        lo, hi = data["ranges_deg"][joint]
        if joint not in data["joints_done"]:
            say(f"  {joint}: not done yet")
            continue
        warn = "  <- no safe range: check this joint" if lo == hi else ""
        say(f"  {joint}: {lo}-{hi}  ({data['approved_from'][joint]}){warn}")
    missing = [j for j in ARM_JOINTS if j not in data["joints_done"]]
    if missing:
        say(f"INCOMPLETE: still to do: {' '.join(missing)} (python scripts/first_moves.py --joints {' '.join(missing)})")
    say(f"Saved to {out}. calib/arm_limits.json was not changed.")


def main(argv=None, read=input, sleep=time.sleep, say=print, proposed_path=None) -> int:
    parser = argparse.ArgumentParser(description="Find each joint's safe range, in small slow steps.")
    parser.add_argument("--dry-run", action="store_true", help="the fake arm; writes runs/arm_limits_proposed_dryrun.json")
    parser.add_argument("--max-delta", type=int, default=DEFAULT_MAX_DELTA, help="most degrees from home (default 45)")
    parser.add_argument("--joints", nargs="+", choices=ARM_JOINTS, help="only these joints, in this order")
    parser.add_argument("--base-minus", type=int, help="the base's approved minus step, watched earlier (no move)")
    parser.add_argument("--base-plus", type=int, help="the base's approved plus step, watched earlier (no move)")
    args = parser.parse_args(argv)
    if not STEP_DEG <= args.max_delta <= 90:
        say("--max-delta must be between 5 and 90.")
        return 1
    out = proposed_path or (DRY_RUN_PROPOSED_PATH if args.dry_run else PROPOSED_PATH)
    if out == config.ARM_LIMITS_PATH:
        say("Refusing to write calib/arm_limits.json.")
        return 1
    joints = args.joints or list(ARM_JOINTS)
    if len(set(joints)) != len(joints):
        say(f"--joints lists a joint twice: {' '.join(joints)}")
        return 1
    limits = arm_module.load_limits()
    seeded = args.base_minus is not None or args.base_plus is not None
    if seeded:
        if args.base_minus is None or args.base_plus is None:
            say("Give both --base-minus and --base-plus.")
            return 1
        if "base" in joints:
            say("--base-minus/--base-plus record the base without moving it: leave base out of --joints "
                "(for example --joints shoulder elbow).")
            return 1
        for direction, value in (("minus", args.base_minus), ("plus", args.base_plus)):
            most = room(limits, "base", direction, args.max_delta)
            if not 0 <= value <= most:
                say(f"--base-{direction} {value} must be between 0 and {most} (the base's room {direction}).")
                return 1
    try:
        approved, sources = read_existing(out)
    except ArmError as error:
        say(f"STOPPED: {error}")
        return 1
    kept = [j for j in ARM_JOINTS if j in approved and j not in joints and not (seeded and j == "base")]
    for joint in kept:
        r = approved[joint]
        say(f"Keeping {joint} from {out}: +{r['plus']} / -{r['minus']} ({sources[joint]})")
    if seeded:  # written before anything moves, so it is kept even if the session fails
        approved["base"] = {"plus": args.base_plus, "minus": args.base_minus}
        sources["base"] = f"typed in (--base-minus/--base-plus) {time.strftime('%Y-%m-%d %H:%M')}"
        write_proposal(out, limits, approved, sources)
        say(f"Recorded base +{args.base_plus} / -{args.base_minus} (typed in, not moved), saved to {out}")
    try:
        arm = session.connect(args.dry_run, limits, echo=say)
    except ArmError as error:
        say(f"STOPPED: {error}")
        return 1

    def save_joint(joint, result):
        approved[joint] = result
        sources[joint] = f"run {time.strftime('%Y-%m-%d %H:%M')}"
        write_proposal(out, limits, approved, sources)
        say(f"  {joint} saved to {out}")

    def body():
        session.confirm_and_enable(arm, read, say, sleep)  # the board starts disarmed
        explore(arm, limits, args.max_delta, lambda q: session.ask(q, read), say, sleep, joints=joints,
                on_joint_done=save_joint)
        say_summary(write_proposal(out, limits, approved, sources), out, say)

    return session.guarded(arm, body, say, sleep)


if __name__ == "__main__":
    sys.exit(main())
