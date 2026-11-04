"""Ring mode (scripts/run_mimic.py --ring): the whole task with the elbow taped (needs.md A10d).

    demo photo -> Spark names the dice -> each die's nearest ring slot (calib/ring.json slots_deg) -> target order
    mess photo -> each die's position -> its base angle on the ring (refused more than 1 cm off it)
    before any move: every pick and drop planned and checked (the 4.5 s grip rule, the limits), and every die at
    least 4 cm from every other die and from every slot it is not going to (the open jaws come down on the ring)
    for each slot: pick the die at its angle, drop it at the slot angle plus a correction (pick_once's moves),
        park at camera_clear, photo (it calibrates itself), measure where it landed:
        within 1.5 cm of the slot: next slot. Otherwise the angular miss shifts the next drop the other way (the total
        shift at most 2 degrees), and the die is picked up again from where it landed, at most 2 more times; then
        FAIL and go on. A miss along the radius cannot be corrected (the ring's radius is fixed): it is logged.

Only numeric corrections (no Spark diagnosis). Needs only the measured dots in layout.json (its slots, mess area and
reach are not used) and a fitted calib/ring.json. Logs go to runs/<time>/ like loop.run.
"""
import json
import math
import time
from pathlib import Path

from mimic import arm as arm_module
from mimic import brain, config, meta_api, pickdrop, plan, scene
from mimic import ring as ring_module
from mimic.ring import RingError

RUN_BUDGET_USD = 0.30
PASS_CM = 1.5
MAX_CORRECTION_DEG = 2.0
MAX_TRIES = 3                 # the first try plus 2 retries
MIN_SPACING_CM = 4.0          # between dice, and between a die and any slot it is not going to
SLOT_SPACING_CM = MIN_SPACING_CM + PASS_CM  # a die that passed may sit 1.5 cm off its slot, still 4 cm from the next
SLOT_TOLERANCE_CM = 1.5       # demo photo: a die this close to a slot is in it


class RingLayoutError(ValueError):
    pass


def _deg_for_cm(ring, cm: float) -> float:
    return math.degrees(cm / ring.radius_cm) * abs(ring.scale)


def check_room(ring, limits) -> None:
    """Refuse, with the numbers, slots too close together or a ring arc too short for the slots and the dice."""
    slots = sorted(ring.slots_deg)
    lo, hi = limits.range_of("base")
    for angle in slots:
        if not lo <= angle <= hi:
            raise RingLayoutError(f"Slot at base {angle:g} is outside the base range {lo}-{hi} in arm_limits.json.")
    for a, b in zip(slots, slots[1:]):
        gap = math.dist(ring.point_at(a), ring.point_at(b))
        if gap < SLOT_SPACING_CM:
            raise RingLayoutError(f"Slots at base {a:g} and {b:g} are {gap:.1f} cm apart on the ring; they need at least "
                                  f"{SLOT_SPACING_CM} cm ({_deg_for_cm(ring, SLOT_SPACING_CM):.1f} degrees of base at "
                                  f"radius {ring.radius_cm:.1f} cm): {MIN_SPACING_CM} cm for the jaws, plus the "
                                  f"{PASS_CM} cm a placed die may be off its slot.")
    arc_cm = math.radians((hi - lo) / abs(ring.scale)) * ring.radius_cm
    # the slots, one gap to the scattered dice, and the dice spaced among themselves
    need_cm = (len(slots) - 1) * SLOT_SPACING_CM + len(slots) * MIN_SPACING_CM
    if arc_cm < need_cm:
        raise RingLayoutError(f"The ring has {arc_cm:.1f} cm of arc between base {lo} and {hi} (radius "
                              f"{ring.radius_cm:.1f} cm); {len(slots)} slots ({SLOT_SPACING_CM} cm apart) and "
                              f"{len(slots)} scattered dice ({MIN_SPACING_CM} cm apart and from the slots) need about "
                              f"{need_cm:.1f} cm ({_deg_for_cm(ring, need_cm):.0f} degrees of base).")


def preflight(layout, ring, limits) -> None:
    """Everything that can be checked before anything is opened."""
    scene.refuse_unmeasured_dots(layout)
    ring.check_ready()
    check_room(ring, limits)
    pickdrop.plan_park(limits, dict(limits.home_deg), ring.camera_clear, ring.shoulder_stop)  # camera_clear in range


def load_and_check():
    """For run_mimic.py --ring: load and check everything before the serial port, camera or API is touched."""
    layout = scene.load_layout()
    ring = ring_module.load_ring()
    limits = arm_module.load_limits()
    preflight(layout, ring, limits)
    arm_module.refuse_unchecked_limits(limits)
    return layout, ring, limits


def pick_drop_plan(ring, limits, start: dict, pick_deg: float, drop_deg: float):
    return pickdrop.plan(limits, start, base_pick=round(pick_deg), base_drop=round(drop_deg),
                         shoulder_pick=ring.shoulder_table, grip_hold=ring.grip_hold, lift=ring.lift,
                         approach=ring.approach, stop_at=ring.shoulder_stop, command_s=ring.command_s)


def check_clear(name: str, slot_deg: float, positions: dict, ring) -> None:
    """The slot a die is going to must be clear of every other die (the open jaws come down there)."""
    slot = ring.point_at(slot_deg)
    for other, pt in positions.items():
        if other != name and math.dist(pt, slot) < MIN_SPACING_CM:
            raise RingLayoutError(f"{other} at ({pt[0]:.1f}, {pt[1]:.1f}) cm is {math.dist(pt, slot):.1f} cm from the "
                                  f"slot at base {slot_deg:g} that {name} is going to (needs {MIN_SPACING_CM} cm). "
                                  "Clear it by hand.")


def check_mess(positions: dict, moves: list, ring, limits) -> None:
    """Before any move: every die reachable, spaced out, and every pick and drop within the 4.5 s grip rule."""
    names = list(positions)
    for name in names:
        try:
            ring.base_angle_for(*positions[name])
        except RingError as error:
            raise RingLayoutError(f"{name}: {error}") from None
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gap = math.dist(positions[a], positions[b])
            if gap < MIN_SPACING_CM:
                raise RingLayoutError(f"{a} and {b} are {gap:.1f} cm apart; dice need at least {MIN_SPACING_CM} cm "
                                      "between them on the ring. Spread them out.")
    park = dict(limits.home_deg, **ring.camera_clear)
    for move in moves:
        slot_deg = ring.slots_deg[move.slot - 1]
        for name in names:
            if name != move.name and math.dist(positions[name], ring.point_at(slot_deg)) < MIN_SPACING_CM:
                raise RingLayoutError(f"{name} is {math.dist(positions[name], ring.point_at(slot_deg)):.1f} cm from slot "
                                      f"{move.slot} (base {slot_deg:g}), where {move.name} is going: move it at least "
                                      f"{MIN_SPACING_CM} cm away.")
        try:
            pick_drop_plan(ring, limits, park, ring.angle_for(*positions[move.name]), slot_deg)
        except pickdrop.PickError as error:
            raise RingLayoutError(f"{move.name} to slot {move.slot}: {error}") from None


def run(arm, capture, layout, ring, limits, wait_for_user=input, out_root=config.RUNS_DIR, say=print,
        sleep=time.sleep) -> dict:
    preflight(layout, ring, limits)
    out_dir = Path(out_root) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = out_dir / "attempts.jsonl"
    started = time.monotonic()
    summary = {"mode": "ring", "success": False, "moves": [], "attempts": 0, "max_error_cm": None, "cost_usd": 0.0,
               "seconds": 0.0, "error": None, "out_dir": str(out_dir)}
    slots_cm = [ring.point_at(a) for a in ring.slots_deg]

    meta_api.reset()
    budget_before = config.BUDGET_USD
    config.BUDGET_USD = min(config.BUDGET_USD, RUN_BUDGET_USD)

    def photo(name: str):
        park = pickdrop.plan_park(limits, arm.known_pose(), ring.camera_clear, ring.shoulder_stop)
        pickdrop.run_steps(arm, park, say, sleep)
        path = capture(out_dir / name)
        return path, scene.calibrate(path, layout, out_dir / f"{Path(name).stem}_dots.jpg")

    try:
        wait_for_user("Put the dice in the ring slots the way you want them, then press Enter: ")
        demo, demo_map = photo("demo.jpg")
        names = brain.name_objects(demo)
        demo_positions = scene.read_positions(demo, names, demo_map, layout)
        target = scene.assign_slots(demo_positions, slots_cm, SLOT_TOLERANCE_CM)
        say(f"Target order (slot 1 first): {', '.join(target)}")

        wait_for_user("Now scatter the dice on the ring (away from the slots), then press Enter: ")
        latest, latest_map = photo("mess.jpg")
        positions = scene.read_positions(latest, target, latest_map, layout)
        moves = plan.plan_moves(positions, target, slots_cm)
        check_mess(positions, moves, ring, limits)  # before anything moves

        shots = 0
        for move in moves:
            slot_deg = ring.slots_deg[move.slot - 1]
            from_cm, correction, result, error_cm = positions[move.name], 0.0, "FAIL", None
            for attempt in range(1, MAX_TRIES + 1):
                check_clear(move.name, slot_deg, positions, ring)
                try:
                    pick_deg = ring.base_angle_for(*from_cm)
                except RingError as error:
                    say(f"  {move.name} -> slot {move.slot}: cannot pick it up again: {error}")
                    break
                drop_deg = slot_deg + correction
                steps, hold_s, _ = pick_drop_plan(ring, limits, arm.known_pose(), pick_deg, drop_deg)
                say(f"  {move.name}: pick at base {round(pick_deg)}, drop at {round(drop_deg)} "
                    f"(claw closed {hold_s:.2f} s)")
                pickdrop.run_steps(arm, steps, say, sleep)
                shots += 1
                latest, latest_map = photo(f"after_{shots}.jpg")
                positions = scene.read_positions(latest, target, latest_map, layout)
                measured = positions[move.name]
                error_cm = math.dist(measured, move.to_cm)
                angle_miss = ring.angle_for(*measured) - slot_deg
                radial_miss = ring.off_ring_cm(*measured)
                passed = error_cm <= PASS_CM
                result = "PASS" if passed else ("RETRY" if attempt < MAX_TRIES else "FAIL")
                next_correction = None
                if not passed and attempt < MAX_TRIES:
                    next_correction = max(-MAX_CORRECTION_DEG, min(MAX_CORRECTION_DEG, correction - angle_miss))
                summary["attempts"] += 1
                with open(attempts_path, "a") as f:
                    f.write(json.dumps({
                        "mode": "ring", "move": move.slot, "name": move.name, "attempt": attempt,
                        "slot_deg": slot_deg, "pick_deg": round(pick_deg), "drop_deg": round(drop_deg),
                        "slot_cm": move.to_cm, "measured_cm": measured, "miss_cm": round(error_cm, 2),
                        "angle_miss_deg": round(angle_miss, 2), "radial_miss_cm": round(radial_miss, 2),
                        "result": result, "next_correction_deg": next_correction, "claw_closed_s": round(hold_s, 2),
                        "photo": Path(latest).name, "cost_so_far_usd": round(meta_api.cost_so_far(), 4),
                        "time_s": round(time.monotonic() - started, 1)}) + "\n")
                say(f"  {move.name} -> slot {move.slot}, try {attempt}: {error_cm:.1f} cm from the slot "
                    f"({angle_miss:+.1f} degrees, {radial_miss:+.1f} cm along the radius) -> {result}")
                if passed:
                    break
                if abs(radial_miss) > PASS_CM:
                    say("  most of that miss is along the radius, which a base correction cannot fix")
                if next_correction is not None:
                    say(f"  -> aiming {next_correction:+.1f} degrees from the slot")
                    correction, from_cm = next_correction, measured
            summary["moves"].append({"name": move.name, "slot": move.slot,
                                     "result": "PASS" if result == "PASS" else "FAIL",
                                     "final_error_cm": round(error_cm, 2) if error_cm is not None else None})

        summary["success"] = all(m["result"] == "PASS" for m in summary["moves"])
        errors = [m["final_error_cm"] for m in summary["moves"] if m["final_error_cm"] is not None]
        summary["max_error_cm"] = max(errors) if errors else None
        return summary
    except Exception as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        try:
            arm.stop()
        except Exception:
            pass
        raise
    finally:
        config.BUDGET_USD = budget_before
        summary["cost_usd"] = round(meta_api.cost_so_far(), 4)
        summary["seconds"] = round(time.monotonic() - started, 1)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        say(f"Run log: {out_dir}  (cost ${summary['cost_usd']:.4f}, cap ${RUN_BUDGET_USD:.2f})")
