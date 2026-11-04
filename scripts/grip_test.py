"""First powered session, step 2: find grip_hold, the grip angle that holds a die, at the home pose.

    python scripts/grip_test.py --lift "shoulder -4"            the real arm
    python scripts/grip_test.py --lift "shoulder -4 elbow +2" --start 50 --dry-run

Uses the joint ranges in calib/arm_limits_proposed.json (from first_moves.py) and the grip settings in
calib/arm_limits.json (first_moves never measures the grip); the real arm must be unlocked.
The closing direction comes from the closed limit (on this claw a LOWER angle closes it, 4 Nov).
1. The arm goes home with the claw open (90). While you place the die between the open jaws nothing is sent;
   press Enter when it is there.
2. The claw closes towards the closed limit in 2-degree steps, starting at --start (default 50); the last step is
   the limit itself, with a warning. Each step: close, hold 1.5 s, open again, then you are asked "holding? y/n".
   It opens between steps because the firmware lets go of any grip held closed for 5 s, and your answer can take
   longer than that. No y down to the limit: it stops with "no hold found down to the limit".
3. At the first y, grip_hold is that angle plus 3 degrees further closed. If that would be at or past the closed
   limit it stops the same way: it never holds or lifts at the limit. Otherwise the lift check at grip_hold:
   close, lift (--lift: one or more joints, 10 degrees in total at most), hold 2 s, lower, open. The planned time
   the grip is closed must stay under max_grip_hold_s minus 0.5 s, or nothing moves.
4. If the die stayed held, grip_hold is written into the proposal as grip_hold_deg: copy it into
   calib/arm_limits.json ('grip close' uses it) and use it as grip_closed when teaching. arm_limits.json is never
   changed here.
s at any prompt, or Ctrl+C, sends S (stop) then D (servos off) and exits.
"""
import argparse
import dataclasses
import json
import sys
import time

from mimic import arm as arm_module
from mimic import config, session
from mimic.arm import ArmError
from mimic.kinematics import ARM_JOINTS

GRIP_STEP_DEG = 2
HOLD_EXTRA_DEG = 3           # grip_hold = the first angle that held, plus this much further closed
DEFAULT_START_DEG = 50       # the first angle tried (Girik, 4 Nov: the empty jaws do not touch down to 40)
NO_HOLD = "no hold found down to the limit"
STEP_HOLD_S = 1.5
LIFT_HOLD_S = 2.0
MAX_LIFT_DEG = 10
STEP_MS_PER_DEGREE = 30      # closing steps: gentle, but quick enough to stay well inside the grip timer
HOLD_MARGIN_S = 0.5
PROPOSED_PATH = config.ROOT / "calib" / "arm_limits_proposed.json"


class GripTestError(RuntimeError):
    pass


def parse_lift(text: str) -> dict:
    """ "shoulder -4 elbow +2" -> {"shoulder": -4, "elbow": 2}; 10 degrees in total at most."""
    words = text.split()
    if not words or len(words) % 2:
        raise GripTestError(f"--lift needs joint/degree pairs, like \"shoulder -4\": got {text!r}")
    lift = {}
    for joint, degrees in zip(words[::2], words[1::2]):
        if joint not in ("base", "shoulder", "elbow"):
            raise GripTestError(f"--lift: {joint!r} is not base, shoulder or elbow")
        lift[joint] = lift.get(joint, 0) + int(degrees)
    total = sum(abs(d) for d in lift.values())
    if total > MAX_LIFT_DEG:
        raise GripTestError(f"--lift adds up to {total} degrees, more than {MAX_LIFT_DEG}")
    return lift


def _ms(arm, pose: dict, ms_per_degree: int) -> int:
    current = arm.known_pose()
    change = max(abs(round(pose[j]) - round(current[j])) for j in pose)
    return max(400, change * ms_per_degree)


def _move(arm, pose: dict, ms: int, sleep) -> None:
    arm.move(pose, ms)
    sleep(ms / 1000 + 0.1)


def check_start(limits, start: int) -> None:
    """--start must close the claw (be on the closing side of open) and not be past the closed limit."""
    d, open_deg, limit = limits.grip_close_dir, limits.grip_open_deg, limits.grip_closed_limit_deg
    if not ((start - open_deg) * d > 0 and (start - limit) * d <= 0):
        raise GripTestError(f"--start {start} must be between open ({open_deg}) and the closed limit ({limit}), "
                            f"on the closing side.")


def hold_angle(limits, held_at: int) -> int | None:
    """HOLD_EXTRA_DEG further closed than the first angle that held, or None if that would be at or past the
    closed limit: it never holds at the limit."""
    d, limit = limits.grip_close_dir, limits.grip_closed_limit_deg
    hold = held_at + d * HOLD_EXTRA_DEG
    return None if (hold - limit) * d >= 0 else hold


def next_step(limits, angle: int | None, start: int) -> int | None:
    """The next angle to try: --start first, then 2 degrees further closed; the last step is the limit itself."""
    d, limit = limits.grip_close_dir, limits.grip_closed_limit_deg
    if angle is None:
        return start
    if angle == limit:
        return None
    step = angle + d * GRIP_STEP_DEG
    return limit if (step - limit) * d > 0 else step


def run_test(arm, lift: dict, ask, say=print, sleep=time.sleep, start: int = DEFAULT_START_DEG) -> int | None:
    """Returns grip_hold: the first angle that held, plus 3 degrees further closed, if it held through the lift."""
    limits = arm.limits
    check_start(limits, start)
    allowed = limits.max_grip_hold_s - HOLD_MARGIN_S
    home = dict(limits.home_deg, grip=limits.grip_open_deg)
    _move(arm, home, _ms(arm, home, STEP_MS_PER_DEGREE), sleep)
    say("Answer each question only after the claw has stopped: anything typed while it moves is ignored.")
    ask("Place the die between the open jaws (nothing is being sent), then press Enter: ")

    angle = None
    while True:
        angle = next_step(limits, angle, start)  # towards the limit (a lower angle on this claw)
        if angle is None:
            say(f"{NO_HOLD} ({limits.grip_closed_limit_deg}): check the die size, the pads and the claw.")
            return None
        warning = arm_module.grip_limit_warning(limits, angle)
        if warning:
            say(warning)
        closed = dict(home, grip=angle)
        close_ms = _ms(arm, closed, STEP_MS_PER_DEGREE)
        if close_ms / 1000 + STEP_HOLD_S > allowed:
            raise GripTestError(f"Closing to {angle} and holding would keep the grip closed for "
                                f"{close_ms / 1000 + STEP_HOLD_S:.1f} s, over {allowed:.1f} s.")
        say(f"  grip {angle}")
        _move(arm, closed, close_ms, lambda s: None)
        sleep(close_ms / 1000 + STEP_HOLD_S)
        _move(arm, home, _ms(arm, home, STEP_MS_PER_DEGREE), sleep)
        if _yes(ask, "holding? y/n (s = stop): "):
            break

    held_at = angle
    angle = hold_angle(limits, held_at)
    if angle is None:
        say(f"{NO_HOLD}: it held at {held_at}, but {HOLD_EXTRA_DEG} degrees further closed would be at or past the "
            f"closed limit ({limits.grip_closed_limit_deg}), so it does not hold or lift there.")
        return None
    say(f"Held at {held_at}: grip_hold = {angle} ({HOLD_EXTRA_DEG} degrees further closed, short of the limit)")

    # the lift check: close, lift, hold 2 s, lower, open, at normal speed so the grip timer is respected
    closed = dict(home, grip=angle)
    lifted = dict(closed, **{j: closed[j] + d for j, d in lift.items()})
    steps = [("close", closed), ("lift", lifted), ("lower", closed)]
    previous, timed = arm.known_pose(), []
    for label, pose in steps:
        change = max(abs(round(pose[j]) - round(previous[j])) for j in pose)
        timed.append((label, pose, max(400, change * limits.min_ms_per_degree)))
        previous = pose
    hold_s = sum(ms for _, _, ms in timed) / 1000 + LIFT_HOLD_S
    if hold_s > allowed:
        raise GripTestError(f"The lift test would hold the grip closed for {hold_s:.1f} s, over the "
                            f"{allowed:.1f} s allowed. Nothing was moved.")
    say(f"Lift test at grip {angle}: planned grip hold {hold_s:.1f} s (limit {allowed:.1f} s)")
    for label, pose, ms in timed:
        _move(arm, pose, ms, sleep)
        if label == "lift":
            sleep(LIFT_HOLD_S)
    _move(arm, home, _ms(arm, home, limits.min_ms_per_degree), sleep)
    if _yes(ask, "was the die held all the way through the lift? y/n (s = stop): "):
        return angle
    say(f"Not held through the lift. Try again, for example with --start {held_at}.")
    return None


def _yes(ask, question: str) -> bool:
    while True:
        answer = ask(question)
        if answer in ("y", "n"):
            return answer == "y"


def main(argv=None, read=input, sleep=time.sleep, say=print, proposed_path=PROPOSED_PATH) -> int:
    parser = argparse.ArgumentParser(description="Find the grip angle that holds a cube.")
    parser.add_argument("--lift", required=True, help='e.g. "shoulder -4" or "shoulder -4 elbow +2" (10 degrees max)')
    parser.add_argument("--dry-run", action="store_true", help="the fake arm")
    parser.add_argument("--start", type=int, default=DEFAULT_START_DEG,
                        help=f"first grip angle tried (default {DEFAULT_START_DEG}); then 2-degree steps towards the "
                             "closed limit")
    args = parser.parse_args(argv)
    try:
        lift = parse_lift(args.lift)
    except (GripTestError, ValueError) as error:
        say(f"STOPPED: {error}")
        return 1
    if proposed_path == config.ARM_LIMITS_PATH:
        say("Refusing to write calib/arm_limits.json.")
        return 1
    if not proposed_path.exists():
        say(f"STOPPED: {proposed_path} not found. Run scripts/first_moves.py first.")
        return 1
    try:
        done = json.loads(proposed_path.read_text()).get("joints_done", list(ARM_JOINTS))
    except ValueError as error:
        say(f"STOPPED: {proposed_path} cannot be read: {error}")
        return 1
    missing = [j for j in ARM_JOINTS if j not in done]
    if missing:
        say(f"STOPPED: the proposal is incomplete (missing: {', '.join(missing)}): finish first_moves.py first "
            f"(python scripts/first_moves.py --joints {' '.join(missing)}).")
        return 1
    try:
        active = arm_module.load_limits()
        arm = session.connect(args.dry_run, active, echo=say)  # a locked arm is refused here
        proposed = arm_module.load_limits(proposed_path)         # the joint ranges you approved
        arm.limits = dataclasses.replace(                        # ...and the grip settings in force now
            proposed, grip_open_deg=active.grip_open_deg, grip_closed_limit_deg=active.grip_closed_limit_deg,
            grip_range_deg=active.grip_range_deg, grip_hold_deg=active.grip_hold_deg)
        check_start(arm.limits, args.start)
    except (ArmError, GripTestError) as error:
        say(f"STOPPED: {error}")
        return 1

    def body():
        session.confirm_and_enable(arm, read, say, sleep)  # the board starts disarmed
        angle = run_test(arm, lift, lambda q: session.ask(q, read), say, sleep, start=args.start)
        if angle is None:
            return 2
        data = json.loads(proposed_path.read_text())
        data["grip_hold_deg"] = angle
        proposed_path.write_text(json.dumps(data, indent=2) + "\n")
        say(f"grip_hold_deg = {angle} written to {proposed_path}: copy it into calib/arm_limits.json "
            "('grip close' uses it) and use it as grip_closed when teaching.")

    return session.guarded(arm, body, say, sleep)


if __name__ == "__main__":
    sys.exit(main())
