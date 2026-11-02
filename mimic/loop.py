"""The whole run: watch the demo, read the mess, then pick, place, check and correct each block.

    demo photo -> Spark names the blocks -> slot order
    mess photo -> where each block is -> plan
    for each move: pick from where the block is in the latest photo, place in its slot, photo, measure.
        within 1.5 cm: next move. Otherwise aim the other way (at most 2 cm off the slot), pick it up
        from where it landed, and try again, at most 2 more times; then log FAIL and go on.

Every photo calibrates itself from its own dots, so the camera may move between photos. The arm moves
to camera_clear (a taught pose out of the picture) before every photo, and SAM is only ever asked while
the arm is parked there, so a budget stop can never happen while a block is held.
Everything goes to runs/<time>/: photos, attempts.jsonl (one line per attempt) and summary.json.
"""
import json
import math
import time
from pathlib import Path

from mimic import arm as arm_module
from mimic import brain, config, kinematics, meta_api, motion, plan, scene

RUN_BUDGET_USD = 0.30          # a full run with the arm; the budget guard stops it beyond this
MAX_DIAGNOSES_PER_RUN = 2      # each Spark diagnosis costs about $0.02
CONSISTENT_WITHIN_CM = 0.5     # Spark's "where I see it" must be this close to the camera's measurement
PASS_CM = 1.5
MAX_CORRECTION_CM = 2.0
MAX_TRIES = 3  # the first try plus 2 retries
WRONG_WAY_MIN_MISS_CM = 0.5  # a Spark fix pointing the same way as a miss bigger than this is ignored


def _clamp(value: float) -> float:
    return max(-MAX_CORRECTION_CM, min(MAX_CORRECTION_CM, value))


def reject_fix(fix: dict, miss) -> str | None:
    """Why Spark's fix must not be used, or None if it may be. The code decides, not Spark."""
    for axis, value, missed in (("x", fix["dx_cm"], miss[0]), ("y", fix["dy_cm"], miss[1])):
        if abs(value) > MAX_CORRECTION_CM:
            return f"Spark's fix {value:+.1f} cm in {axis} is over {MAX_CORRECTION_CM:.0f} cm"
        if abs(missed) > WRONG_WAY_MIN_MISS_CM and value * missed > 0:
            return (f"Spark's fix {value:+.1f} cm in {axis} points the same way the block missed "
                    f"({missed:+.1f} cm), so it would make the miss worse")
    return None


def consistency(advice: dict, miss) -> tuple[bool | None, str]:
    """Does what Spark says it sees agree with what the camera measured? (agrees, words); agrees is None
    when it cannot be checked. Diagnosis only happens after the camera has found the block, so a
    not_found answer always contradicts the measurement."""
    if advice["problem"] == "not_found":
        return False, "contradicts measurement: Spark says it cannot see the block, but the camera measured it"
    seen = advice.get("seen_offset_cm")
    if seen is None:
        return None, "not checked: Spark gave no position"
    gap = math.dist(seen, miss)
    if gap > CONSISTENT_WITHIN_CM:
        return False, (f"contradicts measurement: Spark sees it at ({seen[0]:+.1f}, {seen[1]:+.1f}) cm from the slot, "
                       f"the camera measured ({miss[0]:+.1f}, {miss[1]:+.1f}) cm ({gap:.1f} cm apart)")
    return True, f"agrees with the measurement (within {gap:.1f} cm)"


def choose_correction(slot, measured, previous, before_photo, after_photo, move,
                      diagnose=brain.diagnose, use_spark_fix=None) -> tuple[tuple[float, float], dict]:
    """How far from the slot to aim next time, in sheet cm, and what was decided.

    The numeric correction decides: the opposite of the total miss so far. Spark looks at the photos,
    explains the miss and suggests a fix, which is logged and printed beside it but NOT used, unless
    MIMIC_USE_SPARK_FIX=1; even then only if it passes reject_fix and agrees with the measurement.
    Either way the total is limited to 2 cm each way on x and on y.
    """
    use_spark_fix = config.USE_SPARK_FIX if use_spark_fix is None else use_spark_fix
    miss = (measured[0] - slot[0], measured[1] - slot[1])
    numeric = (_clamp(previous[0] - miss[0]), _clamp(previous[1] - miss[1]))
    info = {"correction_source": "numeric", "problem": None, "explanation": None, "spark_fix": None,
            "spark_seen_offset_cm": None, "consistency": None, "spark_fix_note": None, "fallback_reason": None}
    try:
        advice = diagnose(before_photo, after_photo, move, miss)
    except Exception as error:  # unusable reply, API error, the cost cap is close, or the per-run limit
        info["fallback_reason"] = f"no usable diagnosis: {error}"
        return numeric, info
    agrees, words = consistency(advice, miss)
    info.update(problem=advice["problem"], explanation=advice["explanation"], spark_fix=advice["fix"],
                spark_seen_offset_cm=advice.get("seen_offset_cm"), consistency=words)
    reason = reject_fix(advice["fix"], miss) or (words if agrees is False else None)
    if not use_spark_fix:
        info["spark_fix_note"] = "logged only (set MIMIC_USE_SPARK_FIX=1 to use it)"
        return numeric, info
    if reason:
        info["spark_fix_note"] = f"not used: {reason}"
        info["fallback_reason"] = reason
        return numeric, info
    info["correction_source"] = "spark"
    info["spark_fix_note"] = "used"
    return (_clamp(previous[0] + advice["fix"]["dx_cm"]), _clamp(previous[1] + advice["fix"]["dy_cm"])), info


def _say_decision(info: dict, correction, say) -> None:
    if info["explanation"]:
        say(f"  Spark: {info['explanation']} ({info['problem']})")
        warn = "!! " if info["consistency"].startswith("contradicts") else ""
        say(f"  {warn}Spark's position: {info['consistency']}")
        fix = info["spark_fix"]
        say(f"  Spark's fix ({fix['dx_cm']:+.1f}, {fix['dy_cm']:+.1f}) cm: {info['spark_fix_note']}")
    elif info["fallback_reason"]:
        say(f"  Spark: {info['fallback_reason']}")
    source = "Spark's fix" if info["correction_source"] == "spark" else "numeric correction"
    say(f"  -> {source}: aiming ({correction[0]:+.1f}, {correction[1]:+.1f}) cm from the slot")


def load_and_check():
    """Load the sheet layout, pose grid and arm limits, and check them all, before anything is opened
    (no serial port, no camera, no API call). Used by scripts/run_mimic.py and scripts/benchmark.py."""
    layout = scene.load_layout()
    grid = kinematics.load_grid()
    limits = arm_module.load_limits()
    scene.refuse_placeholder(layout)
    grid.check_ready()
    grid.check_against(limits)
    arm_module.refuse_unchecked_limits(limits)
    return layout, grid, limits


def preflight(arm, layout, grid, limits) -> None:
    """Everything that can be checked before any photo, any API call or any move."""
    scene.refuse_placeholder(layout)
    grid.check_ready()
    grid.check_against(limits)
    if arm.transport.is_real:
        arm_module.refuse_unchecked_limits(limits)
    for n, slot in enumerate(layout.slots_cm, start=1):
        for height in ("hover", "grab"):
            grid.to_angles(slot[0], slot[1], height)  # raises if a slot is outside the taught area


def run(arm, capture, layout, grid, limits, wait_for_user=input, out_root=config.RUNS_DIR,
        say=print, sleep=time.sleep) -> dict:
    preflight(arm, layout, grid, limits)
    out_dir = Path(out_root) / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = out_dir / "attempts.jsonl"
    started = time.monotonic()
    summary = {"success": False, "moves": [], "attempts": 0, "max_error_cm": None, "cost_usd": 0.0,
               "seconds": 0.0, "error": None, "out_dir": str(out_dir)}

    meta_api.reset()
    budget_before = config.BUDGET_USD
    config.BUDGET_USD = min(config.BUDGET_USD, RUN_BUDGET_USD)
    diagnoses_left = [MAX_DIAGNOSES_PER_RUN]

    def limited_diagnose(*args):
        if diagnoses_left[0] <= 0:
            raise brain.DiagnosisError(f"skipped: at most {MAX_DIAGNOSES_PER_RUN} diagnoses per run")
        diagnoses_left[0] -= 1
        return brain.diagnose(*args)

    def photo(name: str):
        motion.go_camera_clear(arm, grid, limits, sleep)
        path = capture(out_dir / name)
        calib = scene.calibrate(path, layout, out_dir / f"{Path(name).stem}_dots.jpg")
        return path, calib

    def where(name: str, path, calib) -> tuple[float, float]:
        return scene.read_positions(path, [name], calib, layout)[name]

    try:
        arm.home()
        wait_for_user("Arrange the blocks in the slots the way you want them, then press Enter: ")
        demo, demo_map = photo("demo.jpg")
        naming = []
        names = brain.name_objects(demo, naming)
        target = scene.read_target(demo, names, demo_map, layout)
        say(f"Target order (slot 1 first): {', '.join(target)}")

        wait_for_user("Now scatter the blocks in the mess area, then press Enter: ")
        latest, latest_map = photo("mess.jpg")
        moves = plan.plan_moves(scene.read_positions(latest, target, latest_map, layout), target, layout.slots_cm)

        shots, first_misses = 0, []
        for move in moves:
            from_cm = where(move.name, latest, latest_map)  # where the block is now, in the latest photo
            correction, result, error_cm = (0.0, 0.0), "FAIL", None
            for attempt in range(1, MAX_TRIES + 1):
                aim = (move.to_cm[0] + correction[0], move.to_cm[1] + correction[1])
                hold_s = motion.pick_and_place(arm, grid, limits, from_cm, aim, sleep, say)
                before = latest
                shots += 1
                latest, latest_map = photo(f"after_{shots}.jpg")
                measured = where(move.name, latest, latest_map)
                miss = (measured[0] - move.to_cm[0], measured[1] - move.to_cm[1])
                error_cm = math.hypot(*miss)
                passed = error_cm <= PASS_CM
                result = "PASS" if passed else ("RETRY" if attempt < MAX_TRIES else "FAIL")
                if attempt == 1:
                    first_misses.append(None if passed else miss)
                next_correction, decision = None, {}
                if not passed and attempt < MAX_TRIES:
                    next_correction, decision = choose_correction(move.to_cm, measured, correction, before, latest,
                                                                  move, limited_diagnose)
                summary["attempts"] += 1
                with open(attempts_path, "a") as f:
                    f.write(json.dumps({
                        "move": move.slot, "name": move.name, "slot_cm": move.to_cm, "attempt": attempt,
                        "from_cm": from_cm, "aim_cm": aim, "measured_cm": measured,
                        "miss_cm": {"dx": round(miss[0], 2), "dy": round(miss[1], 2), "distance": round(error_cm, 2)},
                        "result": result, "next_correction_cm": next_correction, "planned_grip_hold_s": hold_s,
                        "photo": Path(latest).name, "cost_so_far_usd": round(meta_api.cost_so_far(), 4),
                        "time_s": round(time.monotonic() - started, 1), **decision}) + "\n")
                say(f"  {move.name} -> slot {move.slot}, try {attempt}: missed by {error_cm:.1f} cm -> {result}")
                if next_correction:
                    _say_decision(decision, next_correction, say)
                if passed:
                    break
                if next_correction:
                    correction, from_cm = next_correction, measured  # pick it up from where it landed
            summary["moves"].append({"name": move.name, "slot": move.slot, "result": "PASS" if result == "PASS" else "FAIL",
                                     "final_error_cm": round(error_cm, 2)})
            _warn_if_same_direction(first_misses, say)

        summary["success"] = all(m["result"] == "PASS" for m in summary["moves"])
        summary["max_error_cm"] = max(m["final_error_cm"] for m in summary["moves"])
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


def _warn_if_same_direction(first_misses: list, say) -> None:
    """Two moves in a row missing the same way on the first try suggest the arm or the sheet has moved."""
    if len(first_misses) < 2 or None in first_misses[-2:]:
        return
    (ax, ay), (bx, by) = first_misses[-2:]
    cos = (ax * bx + ay * by) / (math.hypot(ax, ay) * math.hypot(bx, by))
    if cos > math.cos(math.radians(45)):
        say("  WARNING: two moves in a row missed the same way. Has the arm or the sheet moved? "
            "The taught grid assumes the arm base never moves relative to the sheet.")
