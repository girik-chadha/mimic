"""First powered session, step 2: find the grip angle that holds a cube, at the home pose.

    python scripts/grip_test.py --lift "shoulder -4"            the real arm
    python scripts/grip_test.py --lift "shoulder -4 elbow +2" --dry-run

Uses the ranges in calib/arm_limits_proposed.json (from first_moves.py), and the real arm must be unlocked.
1. The arm goes home with the claw open. While you place the cube between the open jaws nothing is sent;
   press Enter when it is there.
2. The claw closes in 2-degree steps. Each step: close, hold 1.5 s, open again, then you are asked
   "did it hold the cube? y/n". It opens between steps because the firmware lets go of any grip held
   closed for 5 s, and your answer can take longer than that.
3. At the first y: close, lift (--lift: one or more joints, 10 degrees in total at most), hold 2 s,
   lower, open. The planned time the grip is closed must stay under max_grip_hold_s minus 0.5 s, or
   nothing moves.
4. If the cube stayed held, the angle is written into the proposal as grip_hold_deg (use it as
   grip_closed when teaching). calib/arm_limits.json is never changed.
s at any prompt, or Ctrl+C, sends S (stop) and exits.
"""
import argparse
import json
import sys
import time

from mimic import arm as arm_module
from mimic import config, session
from mimic.arm import ArmError

GRIP_STEP_DEG = 2
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


def run_test(arm, lift: dict, ask, say=print, sleep=time.sleep) -> int | None:
    """Returns the grip angle that held the cube through the lift, or None."""
    limits = arm.limits
    allowed = limits.max_grip_hold_s - HOLD_MARGIN_S
    home = dict(limits.home_deg, grip=limits.grip_open_deg)
    _move(arm, home, _ms(arm, home, STEP_MS_PER_DEGREE), sleep)
    ask("Place the cube between the open jaws (nothing is being sent), then press Enter: ")

    direction = 1 if limits.grip_closed_limit_deg > limits.grip_open_deg else -1
    angle = limits.grip_open_deg
    while True:
        angle += direction * GRIP_STEP_DEG
        if (angle - limits.grip_closed_limit_deg) * direction > 0:
            say("Reached the grip's closed limit without a hold: check the cube size and the claw.")
            return None
        closed = dict(home, grip=angle)
        close_ms = _ms(arm, closed, STEP_MS_PER_DEGREE)
        if close_ms / 1000 + STEP_HOLD_S > allowed:
            raise GripTestError(f"Closing to {angle} and holding would keep the grip closed for "
                                f"{close_ms / 1000 + STEP_HOLD_S:.1f} s, over {allowed:.1f} s.")
        say(f"  grip {angle}")
        _move(arm, closed, close_ms, lambda s: None)
        sleep(close_ms / 1000 + STEP_HOLD_S)
        _move(arm, home, _ms(arm, home, STEP_MS_PER_DEGREE), sleep)
        if _yes(ask, "did it hold the cube? y/n (s = stop): "):
            break

    # the real test: close, lift, hold 2 s, lower, open, at normal speed so the grip timer is respected
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
    if _yes(ask, "was the cube held all the way through the lift? y/n (s = stop): "):
        return angle
    say(f"Not held through the lift. Try again; the next run can start nearer {angle}.")
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
        active = arm_module.load_limits()
        arm = session.connect(args.dry_run, active, echo=say)  # a locked arm is refused here
        arm.limits = arm_module.load_limits(proposed_path)       # the ranges you approved
    except ArmError as error:
        say(f"STOPPED: {error}")
        return 1

    def body():
        angle = run_test(arm, lift, lambda q: session.ask(q, read), say, sleep)
        if angle is None:
            return 2
        data = json.loads(proposed_path.read_text())
        data["grip_hold_deg"] = angle
        proposed_path.write_text(json.dumps(data, indent=2) + "\n")
        say(f"grip_hold_deg = {angle} written to {proposed_path} (use it as grip_closed when teaching).")

    return session.guarded(arm, body, say)


if __name__ == "__main__":
    sys.exit(main())
