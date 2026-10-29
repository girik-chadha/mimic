"""Plan the moves: which object goes to which slot, and in what order. Plain, exact Python, no AI.

The target row starts empty, so each object goes straight from the mess to its own slot and a
placement can never land on another object. Slot 1 is filled first, then slot 2, and so on.
"""
import logging
import math
from dataclasses import dataclass
from itertools import combinations

log = logging.getLogger(__name__)

TOO_CLOSE_CM = 1.0     # two real objects cannot be this close: SAM has found one object under two names
SAME_POINT_CM = 0.01   # two slots closer than this are the same point (allows for rounding)


@dataclass
class Move:
    name: str
    from_cm: tuple[float, float]
    to_cm: tuple[float, float]
    slot: int  # 1 = the first slot in the target row


def extra_objects(positions_cm: dict, target_order: list[str]) -> list[str]:
    """Objects found on the table that are not part of the target. They are left where they are."""
    return [name for name in positions_cm if name not in target_order]


def plan_moves(positions_cm: dict, target_order: list[str], slots_cm: list[tuple]) -> list[Move]:
    """The i-th name in target_order goes from where it is now to slot i (counting from 1)."""
    if not target_order:
        raise ValueError("The target order is empty: there is nothing to build.")
    repeated = sorted({name for name in target_order if target_order.count(name) > 1})
    if repeated:
        raise ValueError(f"These names appear more than once in the target: {repeated}")
    if len(target_order) > len(slots_cm):
        raise ValueError(f"{len(target_order)} objects to place but only {len(slots_cm)} slots.")
    missing = [name for name in target_order if name not in positions_cm]
    if missing:
        raise ValueError(f"Not found on the table: {missing}")

    for (i, a), (j, b) in combinations(enumerate(slots_cm, start=1), 2):
        if math.dist(a, b) < SAME_POINT_CM:
            raise ValueError(f"Slots {i} and {j} are the same point {tuple(a)}.")
    for a, b in combinations(target_order, 2):
        gap = math.dist(positions_cm[a], positions_cm[b])
        if gap < TOO_CLOSE_CM:
            raise ValueError(f"{a!r} and {b!r} are only {gap:.2f} cm apart: probably the same object found twice.")

    extras = extra_objects(positions_cm, target_order)
    if extras:
        log.warning("Not in the target, left where they are: %s", extras)

    return [Move(name, tuple(positions_cm[name]), tuple(slots_cm[i]), slot=i + 1)
            for i, name in enumerate(target_order)]
