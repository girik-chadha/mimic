"""Move the arm one joint at a time from the terminal.

    python scripts/arm_jog.py --dry-run

Type one command per line:
    base +5        shoulder -3        elbow +10        (degrees, one joint at a time)
    grip open      grip close
    home           status             stop             quit

--dry-run prints every line that would be sent, without an arm. It is the only mode until the
arm connection is chosen on Friday (needs.md B3). A real arm is refused while calib/arm_limits.json
is a placeholder or has not been checked by hand.
"""
import argparse
import sys

from mimic import arm as arm_module
from mimic.arm import JOINTS, Arm, ArmError, FakeTransport

HELP = "commands: base|shoulder|elbow|grip +N/-N, grip open, grip close, home, status, stop, quit"
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
            pose = dict(arm.angles or arm.status())
            pose[words[0]] += step
            arm.move(pose, ms=max(MIN_STEP_MS, abs(step) * arm.limits.min_ms_per_degree))
            return f"{words[0]} now {pose[words[0]]}"
    except ArmError as error:
        return f"refused: {error}"
    return HELP


def make_arm(dry_run: bool, limits) -> Arm:
    if dry_run:
        return Arm(FakeTransport(limits, echo=print), limits)
    arm_module.refuse_unchecked_limits(limits)
    raise ArmError("No real arm connection yet: it is chosen on Friday (needs.md B3). Use --dry-run.")


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
    for line in sys.stdin:
        if line.strip().lower() in ("quit", "q", "exit"):
            break
        print(handle(arm, line))


if __name__ == "__main__":
    main()
