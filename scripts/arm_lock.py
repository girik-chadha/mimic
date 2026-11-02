"""Lift or put back the arm's safety lock, in both places at once (needs.md C5).

    python scripts/arm_lock.py unlock
    python scripts/arm_lock.py lock

The lock is four values that must always agree (a unit test checks):
    calib/arm_limits.json            "placeholder": true    "limits_checked_by_hand": false
    firmware/mimic_arm/arm_logic.h   LIMITS_ARE_PLACEHOLDER 1    LIMITS_CHECKED_BY_HAND 0
Unlocking asks you to type yes first. Locking never asks: locking is always safe.
Nothing reaches the arm until the firmware is uploaded again (needs.md C5).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIMITS = ROOT / "calib" / "arm_limits.json"
HEADER = ROOT / "firmware" / "mimic_arm" / "arm_logic.h"
QUESTION = ("Are the joint ranges, grip limit and home pose in BOTH calib/arm_limits.json and arm_logic.h "
            "real, conservative numbers that you have checked by hand? Type yes: ")


def _replace_once(pattern: str, new: str, text: str, where: str) -> str:
    text, count = re.subn(pattern, new, text)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {pattern!r} in {where}, found {count}. Nothing was changed.")
    return text


def set_lock(locked: bool, limits_path=LIMITS, header_path=HEADER) -> None:
    """Change all four values, or none (both files are checked before either is written)."""
    limits = limits_path.read_text()
    limits = _replace_once(r'"placeholder": (true|false)',
                           f'"placeholder": {"true" if locked else "false"}', limits, limits_path.name)
    limits = _replace_once(r'"limits_checked_by_hand": (true|false)',
                           f'"limits_checked_by_hand": {"false" if locked else "true"}', limits, limits_path.name)
    header = header_path.read_text()
    header = _replace_once(r"#define LIMITS_ARE_PLACEHOLDER [01]",
                           f"#define LIMITS_ARE_PLACEHOLDER {1 if locked else 0}", header, header_path.name)
    header = _replace_once(r"#define LIMITS_CHECKED_BY_HAND [01]",
                           f"#define LIMITS_CHECKED_BY_HAND {0 if locked else 1}", header, header_path.name)
    limits_path.write_text(limits)
    header_path.write_text(header)


def main(argv, ask=input, say=print, limits_path=LIMITS, header_path=HEADER) -> int:
    if argv not in (["lock"], ["unlock"]):
        say("Usage: python scripts/arm_lock.py lock|unlock")
        return 1
    if argv == ["unlock"]:
        if ask(QUESTION).strip().lower() != "yes":
            say("Still locked: nothing was changed.")
            return 1
        set_lock(False, limits_path, header_path)
        say("Unlocked in arm_limits.json and arm_logic.h. Next (needs.md C5): run the tests (update "
            "test_committed_limits_are_locked on purpose), copy the firmware to Windows and upload it again.")
    else:
        set_lock(True, limits_path, header_path)
        say("Locked in arm_limits.json and arm_logic.h. Upload the firmware again so the arm is locked too.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
