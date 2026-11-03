"""First powered session, step 3: how repeatable is the arm? (needs.md B6)

    python scripts/repeatability.py --pose 100 80 95 90
    python scripts/repeatability.py --pose 100 80 95 90 --shots 10 --approach 5 --target 20.5 12.0

Sends the same pose N times (default 10), going home in between. It comes at the pose alternately from
one side and the other (all three arm joints +approach, then -approach degrees), so play in the gears
(backlash) shows up. After each shot it takes a photo, finds the claw-tip marker with SAM ("gripper marker"),
and converts it to sheet cm with that photo's own dots. The arm must not hide any dot in the photo.

Main output, the spread: mean position, standard deviation, largest distance from the mean, and the
backlash (the gap between the two approach directions). Verdict against the 1.5 cm placement tolerance:
twice the standard deviation under 0.5 cm means full corrections are fine; over 1 cm means apply only part
of each miss. --target (optional, ruler cm) adds the mean offset and its direction.
The marker sits above the table, so its cm position has a constant offset (parallax): that moves the mean,
not the spread. Cost cap $0.10 (2 SAM calls per shot, about $0.005). s or Ctrl+C sends S and exits.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

from mimic import arm as arm_module
from mimic import camera, config, meta_api, scene, session
from mimic.arm import ArmError

MARKER_PROMPT = "gripper marker"
RUN_BUDGET_USD = 0.10
TOLERANCE_CM = 1.5
FULL_CORRECTION_BELOW_CM = 0.5   # 2 x standard deviation
PART_CORRECTION_ABOVE_CM = 1.0


def measure(arm, capture, layout, pose: dict, shots: int, approach: int, out_dir: Path, say=print,
            sleep=time.sleep) -> list[dict]:
    """One dict per shot: {"shot", "from": "+" or "-", "cm": (x, y), "photo"}."""
    results = []
    for shot in range(1, shots + 1):
        sign = 1 if shot % 2 else -1
        arm.home()
        sleep(2.0)
        near = dict(pose, **{j: pose[j] + sign * approach for j in ("base", "shoulder", "elbow")})
        for target in (near, pose):
            current = arm.known_pose()
            ms = max(800, max(abs(round(target[j]) - round(current[j])) for j in target) * arm.limits.min_ms_per_degree)
            arm.move(target, ms)
            sleep(ms / 1000 + 0.5)
        photo = capture(out_dir / f"shot_{shot}.jpg")
        calib = scene.calibrate(photo, layout, out_dir / f"shot_{shot}_dots.jpg")
        marker = calib.px_to_cm(scene._one_box(photo, MARKER_PROMPT).centre)
        results.append({"shot": shot, "from": "+" if sign > 0 else "-", "cm": marker, "photo": Path(photo).name})
        say(f"  shot {shot} (from {'+' if sign > 0 else '-'}): marker at ({marker[0]:.2f}, {marker[1]:.2f}) cm")
    return results


def _mean(points):
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def spread(results: list[dict], target=None) -> dict:
    points = [r["cm"] for r in results]
    mean = _mean(points)
    distances = [math.dist(p, mean) for p in points]
    std = math.sqrt(sum(d * d for d in distances) / (len(points) - 1)) if len(points) > 1 else 0.0
    numbers = {"shots": len(points), "mean_cm": mean, "std_cm": std, "max_from_mean_cm": max(distances)}
    plus = [r["cm"] for r in results if r["from"] == "+"]
    minus = [r["cm"] for r in results if r["from"] == "-"]
    if plus and minus:
        a, b = _mean(plus), _mean(minus)
        numbers["backlash_cm"] = math.dist(a, b)
        numbers["backlash_vector_cm"] = (a[0] - b[0], a[1] - b[1])
    if target is not None:
        offset = (mean[0] - target[0], mean[1] - target[1])
        numbers["mean_offset_cm"] = offset
        numbers["mean_offset_distance_cm"] = math.hypot(*offset)
    return numbers


def verdict(std_cm: float) -> str:
    two_sigma = 2 * std_cm
    if two_sigma < FULL_CORRECTION_BELOW_CM:
        return (f"2 x standard deviation is {two_sigma:.2f} cm, under {FULL_CORRECTION_BELOW_CM} cm: the arm lands in "
                f"the same place each time, so full corrections are fine.")
    if two_sigma > PART_CORRECTION_ABOVE_CM:
        return (f"2 x standard deviation is {two_sigma:.2f} cm, over {PART_CORRECTION_ABOVE_CM} cm: much of each miss "
                f"is random, so apply only part of the miss when correcting (e.g. half).")
    return (f"2 x standard deviation is {two_sigma:.2f} cm, between {FULL_CORRECTION_BELOW_CM} and "
            f"{PART_CORRECTION_ABOVE_CM} cm: full corrections should work, but may overshoot now and then.")


def _direction(dx: float, dy: float) -> str:
    sides = []
    if abs(dx) >= 0.05:
        sides.append(f"{abs(dx):.2f} cm {'right' if dx > 0 else 'left'}")
    if abs(dy) >= 0.05:
        sides.append(f"{abs(dy):.2f} cm {'away from you' if dy > 0 else 'towards you'}")
    return ", ".join(sides) or "no clear direction"


def report(numbers: dict, say=print) -> None:
    mx, my = numbers["mean_cm"]
    say(f"\nSpread over {numbers['shots']} shots:")
    say(f"  mean position        ({mx:.2f}, {my:.2f}) cm  (includes a constant parallax offset)")
    say(f"  standard deviation   {numbers['std_cm']:.2f} cm")
    say(f"  largest from mean    {numbers['max_from_mean_cm']:.2f} cm  (placement tolerance {TOLERANCE_CM} cm)")
    if "backlash_cm" in numbers:
        say(f"  backlash             {numbers['backlash_cm']:.2f} cm between the + and - approaches "
            f"({_direction(*numbers['backlash_vector_cm'])})")
    if "mean_offset_cm" in numbers:
        say(f"  mean offset from target  {numbers['mean_offset_distance_cm']:.2f} cm: "
            f"{_direction(*numbers['mean_offset_cm'])}")
    say("Verdict: " + verdict(numbers["std_cm"]))


def main(argv=None, capture=camera.capture, sleep=time.sleep, say=print, out_root=config.RUNS_DIR) -> int:
    parser = argparse.ArgumentParser(description="Send one pose N times and measure the spread with the camera.")
    parser.add_argument("--pose", type=int, nargs=4, required=True, metavar=("BASE", "SHOULDER", "ELBOW", "GRIP"))
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--approach", type=int, default=5, help="degrees to come from either side (default 5)")
    parser.add_argument("--target", type=float, nargs=2, metavar=("X", "Y"), help="ruler position of the marker, cm")
    args = parser.parse_args(argv)
    layout = scene.load_layout()
    limits = arm_module.load_limits()
    try:
        scene.refuse_unmeasured_dots(layout)  # it only measures with the camera: the dots are enough
        arm = session.connect(False, limits)  # refused while locked, before the bridge opens
    except (ArmError, scene.SceneError) as error:
        say(f"STOPPED before opening anything: {error}")
        return 1
    pose = dict(zip(("base", "shoulder", "elbow", "grip"), args.pose))
    out_dir = Path(out_root) / f"repeatability_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def body():
        meta_api.reset()
        cache_before, budget_before = meta_api.cache_enabled, config.BUDGET_USD
        meta_api.cache_enabled = False  # every photo is new anyway; this keeps the cost honest
        config.BUDGET_USD = min(config.BUDGET_USD, RUN_BUDGET_USD)
        try:
            results = measure(arm, capture, layout, pose, args.shots, args.approach, out_dir, say, sleep)
        finally:
            meta_api.cache_enabled, config.BUDGET_USD = cache_before, budget_before
        numbers = spread(results, args.target)
        report(numbers, say)
        say(f"Cost: ${meta_api.cost_so_far():.4f} (cap ${RUN_BUDGET_USD:.2f}). Saved in {out_dir}")
        (out_dir / "repeatability.json").write_text(json.dumps({"pose": pose, "results": results, "numbers": numbers,
                                                                "cost_usd": meta_api.cost_so_far()}, indent=2))

    return session.guarded(arm, body, say)


if __name__ == "__main__":
    sys.exit(main())
