"""Find the ring the claw tip reaches (elbow taped, needs.md A10d): 5 base angles, the camera measures the tip.

    python scripts/ring_calibrate.py

Put the gripper marker on the claw tip, facing the camera. Needs the measured dots (layout.json dots_measured) and
the arm unlocked. After a typed yes to enable (pose the shoulder near 90 by hand first), for each point:
1. you type a base angle (spread them out: for example 80, 90, 100, 110, 120);
2. the arm lifts if it is low, turns the base, and lowers the claw onto the table in 2-degree steps (shoulder 173);
3. a photo, with the arm in view: it calibrates itself from the 6 dots (so no dot may be hidden), and SAM finds
   the "gripper marker" (exactly one box), which gives the tip's sheet position;
4. the arm lifts again. A photo that cannot calibrate, or has no single marker, is skipped: try another angle.
After 5 points (more are allowed) type done. The ring is fitted (mimic/ring.py: centre, radius, and the base scale,
which needs 4 or more points) and each point's error is shown; any point over 0.5 cm off refuses the fit. Type yes
to save calib/ring.json: the fit and the points, placeholder false. Slots, camera_clear and the pick numbers already
in ring.json are kept. Cost about $0.005 per point (2 SAM calls), capped at $0.05. Then "Hold the arm" and disable.
"""
import argparse
import sys
import time
from pathlib import Path

from mimic import arm as arm_module
from mimic import camera, config, meta_api, ring, sam, scene, session
from mimic.arm import ArmError
from mimic.pickdrop import ENABLE_NOTE, PickError, plan_lift, plan_touch, run_steps

POINTS_NEEDED = 5
MARKER_PROMPT = "gripper marker"
RUN_BUDGET_USD = 0.05


def measure_tip(photo: str, layout, out_dir: Path, label: str) -> tuple[float, float]:
    """The marker's sheet position in this photo (which calibrates itself). Raises SceneError if it cannot."""
    c = scene.calibrate(photo, layout, out_dir / f"{label}_dots.jpg")
    boxes = sam.find(photo, MARKER_PROMPT)
    if len(boxes) != 1:
        raise scene.SceneError(f"SAM found {len(boxes)} boxes for {MARKER_PROMPT!r}, not exactly 1.")
    return c.px_to_cm(boxes[0].centre)  # the marker is at table height: no height correction


def collect(arm, limits, the_ring, layout, capture, ask, say, sleep, out_dir: Path) -> list:
    """Asks for base angles and measures the tip at each, until at least POINTS_NEEDED and "done"."""
    points = []
    while True:
        prompt = (f"Base angle for point {len(points) + 1}" +
                  (" (or done): " if len(points) >= POINTS_NEEDED else f" ({POINTS_NEEDED} needed): "))
        answer = ask(prompt)
        if answer == "done":
            if len(points) >= POINTS_NEEDED:
                return points
            say(f"  {POINTS_NEEDED} points are needed; there are {len(points)}.")
            continue
        if not answer.lstrip("-").isdigit():
            say("  type a whole base angle, for example 100")
            continue
        base = int(answer)
        if base in [p[0] for p in points]:
            say(f"  base {base} is already measured: choose another angle")
            continue
        try:
            steps = plan_touch(limits, arm.known_pose(), base, the_ring.shoulder_table, the_ring.approach,
                               the_ring.stop_at)
        except PickError as error:
            say(f"  refused: {error}")
            continue
        run_steps(arm, steps, say, sleep)
        label = f"ring_{len(points) + 1}_base{base}"
        try:
            x, y = measure_tip(capture(out_dir / f"{label}.jpg"), layout, out_dir, label)
            points.append((base, x, y))
            say(f"  point {len(points)}: base {base} -> tip at ({x:.2f}, {y:.2f}) cm")
        except scene.SceneError as error:
            say(f"  skipped: {error} Try another angle.")
        run_steps(arm, plan_lift(limits, arm.known_pose(), the_ring.shoulder_table - the_ring.lift,
                                 the_ring.stop_at), say, sleep)


def main(argv=None, read=input, capture=camera.capture, sleep=time.sleep, say=print, ring_path=ring.RING_PATH,
         out_root=config.RUNS_DIR) -> int:
    parser = argparse.ArgumentParser(description="Fit the ring the claw tip reaches, from 5 base angles.")
    parser.parse_args(argv)
    try:
        layout = scene.load_layout()
        scene.refuse_unmeasured_dots(layout)  # each photo calibrates itself from the dots
        the_ring = ring.load_ring(ring_path)
        limits = arm_module.load_limits()
        arm = session.connect(False, limits)   # a locked arm is refused here, before the bridge opens
    except (ArmError, scene.SceneError, ring.RingError) as error:
        say(f"STOPPED before opening anything: {error}")
        return 1
    out_dir = Path(out_root) / f"ring_calibrate_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def body():
        session.confirm_and_enable(arm, read, say, sleep, note=ENABLE_NOTE)
        meta_api.reset()
        budget_before = config.BUDGET_USD
        config.BUDGET_USD = min(config.BUDGET_USD, RUN_BUDGET_USD)
        try:
            points = collect(arm, limits, the_ring, layout, capture, lambda q: session.ask(q, read), say, sleep,
                             out_dir)
        finally:
            config.BUDGET_USD = budget_before
        try:
            result = ring.fit(points)
        except ring.RingError as error:
            say(f"Not saved: {error}")
            return 1
        say(f"Ring: centre ({result.centre_cm[0]:.2f}, {result.centre_cm[1]:.2f}) cm, radius {result.radius_cm:.2f} cm, "
            f"base scale {result.scale:+.3f}, offset {result.offset_deg:.1f} degrees")
        for (base, x, y), error in zip(points, result.errors_cm):
            say(f"  base {base}: ({x:.2f}, {y:.2f}) cm, {error:.2f} cm from the ring")
        say(f"Cost: ${meta_api.cost_so_far():.4f}. Photos and labelled dots in {out_dir}")
        if session.ask(f"Type yes to save {ring_path}: ", read) != "yes":
            say("Not saved.")
            return 1
        note = (f"FITTED by scripts/ring_calibrate.py on {time.strftime('%Y-%m-%d %H:%M')} from {len(points)} points "
                "(the tip at table height; the marker found by SAM in self-calibrated photos). slots_deg and "
                "camera_clear are set by hand. See mimic/ring.py and needs.md A10d.")
        ring.save_ring(ring.with_fit(the_ring, result, points, note), ring_path)
        say(f"Saved {ring_path}.")

    return session.guarded(arm, body, say, sleep)


if __name__ == "__main__":
    sys.exit(main())
