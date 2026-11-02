"""What every interactive arm script shares.

- connect(): the fake arm for a dry run, or the real arm through the serial bridge. A real arm is refused
  while calib/arm_limits.json is locked, BEFORE the bridge opens the serial port.
- ask(): every prompt accepts "s", which stops the arm and ends the script.
- guarded(): runs a script's body so that "s" or Ctrl+C sends S (stop) to the arm, then closes the bridge
  (which sends S again as it exits).
"""
from mimic import arm as arm_module
from mimic.arm import Arm, FakeTransport


class StopRequested(Exception):
    pass


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


def guarded(arm: Arm, body, say=print) -> int:
    """Run body(); on "s", Ctrl+C or any error send S to the arm. Always close the bridge. Returns an exit code."""
    try:
        return body() or 0
    except (StopRequested, KeyboardInterrupt) as stop:
        _stop(arm)
        say("\nStopped" + (" (Ctrl+C)" if isinstance(stop, KeyboardInterrupt) else "") + ": S sent to the arm.")
        return 130
    except Exception as error:
        _stop(arm)
        say(f"\nSTOPPED: {type(error).__name__}: {error} (S sent to the arm)")
        return 1
    finally:
        arm.transport.close()


def _stop(arm: Arm) -> None:
    try:
        arm.stop()
    except Exception:
        pass  # the bridge may already be gone; closing it below sends S anyway


def command_loop(handle, lines, say=print) -> None:
    """For jog-style scripts: one typed command per line. quit ends it; s stops the arm."""
    for line in lines:
        word = line.strip().lower()
        if word in ("quit", "q", "exit"):
            return
        if word == "s":
            raise StopRequested()
        say(handle(line))
