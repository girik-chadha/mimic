"""What every interactive arm script shares.

- connect(): the fake arm for a dry run, or the real arm through the serial bridge. A real arm is refused
  while calib/arm_limits.json is locked, BEFORE the bridge opens the serial port.
- ask(): every prompt accepts "s", which stops the arm and ends the script.
- confirm_and_enable(): asks for a typed yes, then switches the servos on (E). The board starts disarmed.
- guarded(): runs a script's body so that "s", Ctrl+C or any error sends S (stop) and then D (servos off, limp),
  and a normal end says "hold the arm" and disables it. Then it closes the bridge (which sends S as it exits).
"""
import time

from mimic import arm as arm_module
from mimic.arm import ENABLE_DEFAULT_DEG, JOINTS, Arm, FakeTransport

DISABLE_WARNING_S = 2.0  # at a normal end: time to take hold of the arm before it goes limp


class StopRequested(Exception):
    pass


class Declined(Exception):
    """Not enabled because the answer was not yes: nothing was moved."""


def connect(dry_run: bool, limits, echo=print) -> Arm:
    if dry_run:
        return Arm(FakeTransport(limits, echo=echo), limits)
    arm_module.refuse_unchecked_limits(limits)  # checked BEFORE the bridge (and the serial port) is opened
    return Arm(arm_module.SerialBridgeTransport(), limits)


def ask(prompt: str, read=input) -> str:
    """Ask a question. "s" at any prompt means stop now."""
    answer = read(prompt).strip().lower()
    if answer == "s":
        raise StopRequested()
    return answer


def confirm_and_enable(arm: Arm, read=input, say=print, sleep=None, angles: dict | None = None) -> dict:
    """Say what enabling does, ask for a typed yes, then switch the servos on. Returns the pose."""
    status = arm.status()
    if status["armed"]:
        say("The arm is already enabled: its servos are on.")
        return {j: status[j] for j in JOINTS}
    pose = angles or {j: ENABLE_DEFAULT_DEG for j in JOINTS}
    say("Enabling switches the servos on one at a time, 0.4 s apart (base, shoulder, elbow, claw), and each jumps "
        "straight to " + ", ".join(f"{j} {pose[j]}" for j in JOINTS) + " from wherever it is now.")
    answer = ask("Is the arm posed there by hand, the area clear, and a hand on the switch? Type yes to enable: ",
                 read)
    if answer != "yes":
        raise Declined("Not enabled: nothing was moved.")
    return arm.enable(angles, sleep=sleep)


def guarded(arm: Arm, body, say=print, sleep=None) -> int:
    """Run body(). On "s", Ctrl+C or any error: S then D (servos off). On a normal end: say "hold the arm", then D.
    Always close the bridge. Returns an exit code."""
    try:
        code = body() or 0
        _finish(arm, say, sleep)
        return code
    except Declined as declined:
        say(str(declined))
        return 1
    except (StopRequested, KeyboardInterrupt) as stop:
        _stop_and_disable(arm)
        say("\nStopped" + (" (Ctrl+C)" if isinstance(stop, KeyboardInterrupt) else "") +
            ": S and D sent, the servos are off.")
        return 130
    except Exception as error:
        say(f"\nSTOPPED: {type(error).__name__}: {error}")  # the reason first, then stop and disable
        _stop_and_disable(arm)
        say("S and D sent, the servos are off.")
        return 1
    finally:
        arm.transport.close()


def _stop_and_disable(arm: Arm) -> None:
    for step in (arm.stop, arm.disable):
        try:
            step()
        except Exception:
            pass  # the bridge may already be gone (closing it sends S anyway), or the firmware is older (no D)


def _finish(arm: Arm, say, sleep) -> None:
    """A normal end: if the servos are on, warn, wait, then switch them off."""
    try:
        if not arm.status()["armed"]:
            return  # nothing is on (disarmed, or the firmware is locked)
        say(f"Hold the arm: it goes limp in {DISABLE_WARNING_S:g} s (disabling).")
        (sleep or time.sleep)(DISABLE_WARNING_S)
        arm.disable()
        say("Disabled: the servos are off.")
    except Exception as error:
        say(f"Could not disable the arm: {error}")


def command_loop(handle, lines, say=print) -> None:
    """For jog-style scripts: one typed command per line. quit ends it; s stops the arm."""
    for line in lines:
        word = line.strip().lower()
        if word in ("quit", "q", "exit"):
            return
        if word == "s":
            raise StopRequested()
        say(handle(line))
