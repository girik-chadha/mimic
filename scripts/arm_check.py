"""First real test with the arm: READ-ONLY. Takes no arguments.

    python scripts/arm_check.py

It first asks you to confirm the batteries are disconnected and the switch is OFF. Only then does it
open the port (through the Windows serial bridge) and send exactly these four lines:

    ?                  -> A 90 90 90 90 L         (L = locked: no servo is attached or driven)
    S                  -> OK
    J 90 90 90 90 800  -> ERR limits not measured
    H                  -> ERR limits not measured

The raw J and H lines skip the Python lock on purpose: they are the only way to prove that the
firmware's own lock works. If J or H is ever accepted, S is sent straight away and the check reports
that the firmware is NOT locked.
"""
import re
import sys

from mimic.arm import ArmError, SerialBridgeTransport

QUESTION = "Batteries disconnected and switch OFF? Type yes: "
LINES = ["?", "S", "J 90 90 90 90 800", "H"]  # hard-coded: nothing else is ever sent
LOCKED_STATUS = re.compile(r"^A -?\d+ -?\d+ -?\d+ -?\d+ L$")
REFUSED = "ERR limits not measured"
EN_HINT = ("No reply to the first '?'. Press the EN button on the ESP32 once, wait 2 seconds, and run "
           "arm_check.py again. (Also check that no other program, such as an upload, has COM7 open.)")


def check(transport, say) -> int:
    """0 = firmware answers and is locked; 1 = something did not answer as expected; 2 = NOT locked."""
    try:
        reply = transport.send("?")
    except ArmError as error:
        say(f"  ?  -> no reply ({error})")
        say(EN_HINT)
        return 1
    say(f"  ?  -> {reply}")
    if not LOCKED_STATUS.match(reply):
        say("FAIL: '?' should end with L. If it has no L, the firmware is NOT locked: switch the arm off.")
        return 2 if reply.startswith("A ") else 1

    try:
        for line in LINES[1:]:
            reply = transport.send(line)
            say(f"  {line}  -> {reply}")
            if line == "S" and reply != "OK":
                say("FAIL: S should reply OK.")
                return 1
            if line != "S" and reply == "OK":
                stop_reply = transport.send("S")
                say(f"  S  -> {stop_reply}   (sent at once)")
                say(f"FIRMWARE IS NOT LOCKED: stop. It accepted {line!r}. Switch the arm off and check arm_logic.h.")
                return 2
            if line != "S" and reply != REFUSED:
                say(f"FAIL: {line!r} should reply {REFUSED!r}.")
                return 1
    except ArmError as error:
        say(f"FAIL: {error}")
        return 1
    say("PASS: the firmware answers, and it is locked (J and H refused, no servo driven).")
    return 0


def run(argv, ask, make_transport, say) -> int:
    if argv:
        say("arm_check.py takes no arguments.")
        return 1
    if ask(QUESTION).strip().lower() != "yes":
        say("Stopped: the port was not opened.")
        return 1
    try:
        transport = make_transport()
    except ArmError as error:
        say(f"STOPPED: {error}")
        return 1
    try:
        return check(transport, say)
    finally:
        transport.close()  # the bridge sends S once more as it closes


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:], input, SerialBridgeTransport, print))
