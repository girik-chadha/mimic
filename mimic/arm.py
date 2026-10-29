"""Talk to the arm: one command line out, one reply line back (docs/plan.md 2.2).

    J <base> <shoulder> <elbow> <grip> <ms>   ->  OK
    H                                         ->  OK
    ?                                         ->  A <base> <shoulder> <elbow> <grip>
    S                                         ->  OK

Safety is in two places on purpose. Python refuses anything outside the limits in
calib/arm_limits.json (an out-of-range angle here means a bug upstream). The firmware clamps to
the same limits as the last line of defence. A unit test checks both use the same numbers.
No real arm is moved while the limits are placeholders or have not been checked by hand.
"""
import json
from dataclasses import dataclass

from mimic import config

JOINTS = ("base", "shoulder", "elbow", "grip")
MAX_LINE_CHARS = 64  # the firmware rejects longer lines


class ArmError(RuntimeError):
    pass


@dataclass
class ArmLimits:
    pins: dict[str, int]
    ranges_deg: dict[str, tuple[int, int]]   # base, shoulder, elbow
    grip_open_deg: int
    grip_closed_limit_deg: int               # never close further (the grip servo overheats)
    home_deg: dict[str, int]
    min_ms_per_degree: int                   # speed limit
    max_grip_hold_s: float                   # the firmware relaxes a closed grip after this long
    placeholder: bool
    limits_checked_by_hand: bool

    def range_of(self, joint: str) -> tuple[int, int]:
        if joint == "grip":
            return tuple(sorted((self.grip_open_deg, self.grip_closed_limit_deg)))
        return self.ranges_deg[joint]


def load_limits(path=config.ARM_LIMITS_PATH) -> ArmLimits:
    with open(path) as f:
        data = json.load(f)
    limits = ArmLimits(
        pins=data["pins"],
        ranges_deg={j: tuple(r) for j, r in data["ranges_deg"].items()},
        grip_open_deg=data["grip_open_deg"],
        grip_closed_limit_deg=data["grip_closed_limit_deg"],
        home_deg=data["home_deg"],
        min_ms_per_degree=data["min_ms_per_degree"],
        max_grip_hold_s=data["max_grip_hold_s"],
        placeholder=data["placeholder"],
        limits_checked_by_hand=data["limits_checked_by_hand"],
    )
    if sorted(limits.pins) != sorted(JOINTS) or len(set(limits.pins.values())) != 4:
        raise ArmError(f"arm_limits.json needs 4 different pins, one per joint: {limits.pins}")
    for joint in JOINTS[:3]:
        lo, hi = limits.ranges_deg[joint]
        if not 0 <= lo < hi <= 180:
            raise ArmError(f"arm_limits.json: {joint} range {lo}-{hi} must be low-high within 0-180.")
    if limits.grip_open_deg == limits.grip_closed_limit_deg:
        raise ArmError("arm_limits.json: grip open and grip closed limit cannot be the same angle.")
    for joint in JOINTS:
        lo, hi = limits.range_of(joint)
        if not lo <= limits.home_deg[joint] <= hi:
            raise ArmError(f"arm_limits.json: home {joint} {limits.home_deg[joint]} is outside its range {lo}-{hi}.")
    if limits.min_ms_per_degree <= 0 or limits.max_grip_hold_s <= 0:
        raise ArmError("arm_limits.json: min_ms_per_degree and max_grip_hold_s must be above 0.")
    return limits


def refuse_unchecked_limits(limits: ArmLimits) -> None:
    """The same guard as layout.json: no real move until the limits are real and checked."""
    if limits.placeholder:
        raise ArmError("calib/arm_limits.json still holds placeholder numbers: "
                       "measure the arm and set placeholder to false.")
    if not limits.limits_checked_by_hand:
        raise ArmError("calib/arm_limits.json has not been checked by hand: jog each joint gently, "
                       "then set limits_checked_by_hand to true.")


class FakeTransport:
    """Stands in for the real arm: remembers every line sent and answers like the firmware would."""
    is_real = False

    def __init__(self, limits: ArmLimits, echo=None):
        self.sent = []
        self.angles = [limits.home_deg[j] for j in JOINTS]
        self.home = list(self.angles)
        self.echo = echo  # e.g. print, to show each line in a dry run

    def send(self, line: str) -> str:
        self.sent.append(line)
        parts = line.split()
        if line == "?":
            reply = "A " + " ".join(str(a) for a in self.angles)
        elif line == "H":
            self.angles, reply = list(self.home), "OK"
        elif parts and parts[0] == "J" and len(parts) == 6:
            self.angles, reply = [int(p) for p in parts[1:5]], "OK"
        elif line == "S":
            reply = "OK"
        else:
            reply = "ERR bad command"
        if self.echo:
            self.echo(f"  -> {line}    <- {reply}")
        return reply


class Arm:
    def __init__(self, transport, limits: ArmLimits):
        self.transport = transport
        self.limits = limits
        self.angles = None  # last known pose; None = unknown, ask the arm

    # ---------- sending ----------

    def _send(self, line: str) -> str:
        if len(line) > MAX_LINE_CHARS:
            raise ArmError(f"Command is {len(line)} characters, over the {MAX_LINE_CHARS} the firmware accepts.")
        return self.transport.send(line).strip()

    def _send_expect_ok(self, line: str) -> None:
        reply = self._send(line)
        if reply != "OK":
            raise ArmError(f"The arm replied {reply!r} to {line!r}.")

    def _refuse_real_move_with_unchecked_limits(self) -> None:
        if self.transport.is_real:
            refuse_unchecked_limits(self.limits)

    # ---------- the four commands ----------

    def status(self) -> dict:
        reply = self._send("?")
        parts = reply.split()
        if len(parts) != 5 or parts[0] != "A":
            raise ArmError(f"The arm replied {reply!r} to '?'.")
        self.angles = dict(zip(JOINTS, (int(p) for p in parts[1:])))
        return dict(self.angles)

    def move(self, angles: dict, ms: int = 800) -> None:
        self._refuse_real_move_with_unchecked_limits()
        if set(angles) != set(JOINTS):
            raise ArmError(f"A move needs exactly these joints {JOINTS}, got {sorted(angles)}.")
        target = {j: int(round(angles[j])) for j in JOINTS}
        for joint, angle in target.items():
            lo, hi = self.limits.range_of(joint)
            if not lo <= angle <= hi:
                raise ArmError(f"{joint} {angle} degrees is outside its range {lo}-{hi}.")
        if not isinstance(ms, int) or ms < 0:
            raise ArmError(f"Move time must be a whole number of ms, got {ms!r}.")
        current = self.angles or self.status()
        biggest = max(abs(target[j] - current[j]) for j in JOINTS)
        needed = biggest * self.limits.min_ms_per_degree
        if ms < needed:
            raise ArmError(f"Too fast: {biggest} degrees in {ms} ms; the speed limit needs at least {needed} ms.")
        self._send_expect_ok("J " + " ".join(str(target[j]) for j in JOINTS) + f" {ms}")
        self.angles = target

    def grip(self, closed: bool) -> None:
        current = self.angles or self.status()
        pose = dict(current)
        pose["grip"] = self.limits.grip_closed_limit_deg if closed else self.limits.grip_open_deg
        change = abs(pose["grip"] - current["grip"])
        self.move(pose, ms=max(500, change * self.limits.min_ms_per_degree))

    def home(self) -> None:
        self._refuse_real_move_with_unchecked_limits()
        self._send_expect_ok("H")
        self.angles = dict(self.limits.home_deg)

    def stop(self) -> None:
        """Always allowed: stopping must always work."""
        self._send_expect_ok("S")
        self.angles = None  # it stopped somewhere along the way; ask before the next move
