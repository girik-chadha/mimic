"""Move the arm one joint at a time from the terminal.

    python scripts/arm_jog.py --dry-run

Type one command per line:
    base +5        shoulder -3        elbow +10        (degrees, one joint at a time)
    grip open      grip close
    home           status             stop             quit
    s (or Ctrl+C)  stop the arm and exit

--dry-run prints every line that would be sent, without an arm. Without --dry-run it talks to the
real arm through the Windows serial bridge, but only once calib/arm_limits.json is no longer a
placeholder and has been checked by hand; until then it refuses before opening the port.
"""
import argparse
import sys

from mimic import arm as arm_module
from mimic import session
from mimic.arm import JOINTS, Arm, ArmError

HELP = "commands: base|shoulder|elbow|grip +N/-N, grip open, grip close, home, status, stop, quit, s (stop + exit)"
MIN_STEP_MS = 400


def handle(arm: Arm, line: str) -> str:
    """Carry out one typed command and say what happened."""
    words = line.strip().lower().split()
    try:
        if words == ["home"]:
            arm.home()
            return "home"
        if words == ["status"]:
            return str(arm.status())
        if words == ["stop"]:
            arm.stop()
            return "stopped"
        if words in (["grip", "open"], ["grip", "close"]):
            arm.grip(closed=words[1] == "close")
            return f"grip {words[1]}"
        if len(words) == 2 and words[0] in JOINTS and words[1][:1] in "+-" and words[1][1:].isdigit():
            step = int(words[1])
            pose = arm.known_pose()
            pose[words[0]] += step
            arm.move(pose, ms=max(MIN_STEP_MS, abs(step) * arm.limits.min_ms_per_degree))
            return f"{words[0]} now {pose[words[0]]}"
    except ArmError as error:
        return f"refused: {error}"
    return HELP


def make_arm(dry_run: bool, limits) -> Arm:
    return session.connect(dry_run, limits)  # a real arm is refused while locked, before the bridge opens


def main():
    parser = argparse.ArgumentParser(description="Move the arm one joint at a time.")
    parser.add_argument("--dry-run", action="store_true", help="print the commands instead of sending them")
    args = parser.parse_args()
    try:
        arm = make_arm(args.dry_run, arm_module.load_limits())
    except ArmError as error:
        print(f"STOPPED: {error}")
        sys.exit(1)
    print("DRY RUN: nothing is sent to an arm." if args.dry_run else "")
    print(HELP)
    sys.exit(session.guarded(arm, lambda: session.command_loop(lambda line: handle(arm, line), sys.stdin)))


if __name__ == "__main__":
    main()
