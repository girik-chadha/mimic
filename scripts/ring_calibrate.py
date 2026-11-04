"""Find the ring the claw tip reaches (elbow taped, needs.md A10d): 5 base angles, the camera measures the tip.

    python scripts/ring_calibrate.py
    python scripts/ring_calibrate.py --table-shoulder 171      a new table-height shoulder angle (saved with the fit)
    python scripts/ring_calibrate.py --marker-name "yellow tape"   another SAM prompt for the marker (saved too)
    python scripts/ring_calibrate.py --camera-frames 30            read more frames per photo (camera_frames: 15)
    python scripts/ring_calibrate.py --camera-name Camo            another webcam (camera_name, or MIMIC_CAMERA)

Put the gripper marker on the claw tip, facing the camera. Needs the measured dots (layout.json dots_measured) and
the arm unlocked. After a typed yes to enable (at the start pose, calib/start_pose.json: pose the arm like it by
hand first), for each point:
1. you type a base angle (spread them out: for example 80, 90, 100, 110, 120);
2. the arm lifts if it is low, turns the base, and lowers the claw onto the table in 2-degree steps, to shoulder_table
   from calib/ring.json (or --table-shoulder), never past the shoulder stop (stop_at, or 3 degrees past the table);
3. a photo, with the arm in view: it calibrates itself from the 6 dots (so no dot may be hidden), and SAM looks for
   marker_name from calib/ring.json (default "small yellow rectangle"). Only boxes whose centre is on the sheet count,
   because the tip is lowered onto the paper (other yellow things, like the control board's pin sockets, are not);
   exactly one must remain, and it gives the tip's sheet position. Every box and where it fell is printed, and the
   photo is saved with all of them drawn and the chosen one marked (runs/ring_calibrate_<time>/..._marker.jpg);
4. the arm lifts again. A photo that cannot calibrate, has no single marker on the sheet, or is still too bright or
   too dark after one retake (brightness outside 40-235: lock the camera app's exposure), is skipped: try another
   angle.
Every photo's camera name, size and brightness are printed (camera.capture, camera_frames frames, keeping the last).
After 5 points (more are allowed) type done. The ring is fitted (mimic/ring.py: centre, radius, and the base scale,
which needs 4 or more points) and each point's error is shown; any point over 0.5 cm off refuses the fit. Type yes
to save calib/ring.json: the fit and the points, placeholder false, and the table angle used (so a --table-shoulder
value is saved with the ring it measured). Slots, camera_clear and the other pick numbers in ring.json are kept. Cost about $0.005 per point (2 SAM calls), capped at $0.05. Then "Hold the arm" and disable.
"""
import argparse
import dataclasses
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from mimic import arm as arm_module
from mimic import camera, config, meta_api, ring, sam, scene, session
from mimic.arm import ArmError
from mimic.pickdrop import PickError, plan_lift, plan_touch, run_steps

POINTS_NEEDED = 5
RUN_BUDGET_USD = 0.05
CHOSEN, OTHER, SHEET = (0, 180, 0), (0, 0, 220), (200, 120, 0)   # drawing colours (blue, green, red)


def on_sheet(boxes: list, c, layout) -> tuple[list, list[str]]:
    """The boxes whose centre is on the sheet (where the tip is lowered), and a line for every box saying where it fell."""
    (x0, y0), (x1, y1) = layout.sheet_min_cm, layout.sheet_max_cm
    inside, lines = [], []
    for b in boxes:
        x, y = c.px_to_cm(b.centre)
        on = x0 <= x <= x1 and y0 <= y <= y1
        lines.append(f"({b.centre[0]:.0f}, {b.centre[1]:.0f}) px -> ({x:.1f}, {y:.1f}) cm, "
                     f"{'on the sheet' if on else 'off the sheet'}")
        if on:
            inside.append(b)
    return inside, lines


def _pt(p):
    return (int(round(p[0])), int(round(p[1])))


def draw_marker(photo: str, boxes: list, chosen, c, layout, out_path) -> None:
    """The photo with the sheet outlined, every matched box drawn, and the chosen one marked."""
    img = cv2.imread(photo)
    (x0, y0), (x1, y1) = layout.sheet_min_cm, layout.sheet_max_cm
    corners = [_pt(c.cm_to_px(p)) for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    cv2.polylines(img, [np.array(corners, np.int32)], True, SHEET, 2)
    for b in boxes:
        colour = CHOSEN if b is chosen else OTHER
        cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), colour, 3)
        cv2.putText(img, "chosen" if b is chosen else "not used", (b.x1, b.y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    colour, 2)
    cv2.imwrite(str(out_path), img)


def measure_tip(photo: str, layout, out_dir: Path, label: str, marker_name: str, say=print) -> tuple[float, float]:
    """The marker's sheet position in this photo (which calibrates itself). Raises SceneError if there is not exactly
    one matching box on the sheet. Saves the photo with every box drawn and the chosen one marked."""
    c = scene.calibrate(photo, layout, out_dir / f"{label}_dots.jpg")
    boxes = sam.find(photo, marker_name)
    inside, lines = on_sheet(boxes, c, layout)
    chosen = inside[0] if len(inside) == 1 else None
    marked = out_dir / f"{label}_marker.jpg"
    draw_marker(photo, boxes, chosen, c, layout, marked)
    if chosen is None:
        raise scene.SceneError(f"SAM found {len(boxes)} boxes for {marker_name!r}, {len(inside)} on the sheet (exactly "
                               f"1 is needed): " + ("; ".join(lines) if lines else "none") + f". Photo: {marked}")
    if len(boxes) > 1:
        say(f"  {len(boxes)} boxes for {marker_name!r}: " + "; ".join(lines) + " -> using the one on the sheet")
    return c.px_to_cm(chosen.centre)  # the marker is at table height: no height correction


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
                               the_ring.shoulder_stop)
        except PickError as error:
            say(f"  refused: {error}")
            continue
        run_steps(arm, steps, say, sleep)
        label = f"ring_{len(points) + 1}_base{base}"
        try:
            x, y = measure_tip(capture(out_dir / f"{label}.jpg"), layout, out_dir, label, the_ring.marker_name, say)
            points.append((base, x, y))
            say(f"  point {len(points)}: base {base} -> tip at ({x:.2f}, {y:.2f}) cm")
        except (scene.SceneError, camera.CameraError) as error:
            say(f"  skipped: {error} Try another angle.")
        run_steps(arm, plan_lift(limits, arm.known_pose(), the_ring.shoulder_table - the_ring.lift,
                                 the_ring.shoulder_stop), say, sleep)


def main(argv=None, read=input, capture=None, sleep=time.sleep, say=print, ring_path=ring.RING_PATH,
         out_root=config.RUNS_DIR) -> int:
    parser = argparse.ArgumentParser(description="Fit the ring the claw tip reaches, from 5 base angles.")
    parser.add_argument("--table-shoulder", type=int,
                        help="shoulder angle at table height (default: shoulder_table in calib/ring.json)")
    parser.add_argument("--marker-name", help="SAM prompt for the tip marker (default: marker_name in calib/ring.json)")
    parser.add_argument("--camera-frames", type=int,
                        help="frames per photo grab, keeping the last (default: camera_frames in calib/ring.json)")
    parser.add_argument("--camera-name", help="the Windows webcam, e.g. Camo (default: MIMIC_CAMERA, then camera_name "
                                              "in calib/ring.json)")
    args = parser.parse_args(argv)
    try:
        layout = scene.load_layout()
        scene.refuse_unmeasured_dots(layout)  # each photo calibrates itself from the dots
        the_ring = ring.load_ring(ring_path)
        if args.table_shoulder is not None:
            the_ring = dataclasses.replace(the_ring, shoulder_table=args.table_shoulder)
        if args.marker_name:
            the_ring = dataclasses.replace(the_ring, marker_name=args.marker_name)
        frames = args.camera_frames or the_ring.camera_frames
        camera_name = camera.camera_name(args.camera_name, the_ring.camera_name)
        if capture is None:  # the real camera, with this many frames per grab
            def capture(path):
                return camera.capture(path, frames=frames, sleep=sleep, say=say, name=camera_name)
        limits = arm_module.load_limits()
        arm = session.connect(False, limits)   # a locked arm is refused here, before the bridge opens
    except (ArmError, scene.SceneError, ring.RingError) as error:
        say(f"STOPPED before opening anything: {error}")
        return 1
    out_dir = Path(out_root) / f"ring_calibrate_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    say(f"Table height: shoulder {the_ring.shoulder_table}; the shoulder is never sent past {the_ring.shoulder_stop}. "
        f"Marker: {the_ring.marker_name!r}. Camera: {camera_name}, {frames} frames per photo.")

    def body():
        session.confirm_and_enable(arm, read, say, sleep)  # at the start pose
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
        for line in ring.describe(result, points):
            say(line)
        say(f"Cost: ${meta_api.cost_so_far():.4f}. Photos and labelled dots in {out_dir}")
        if session.ask(f"Type yes to save {ring_path}: ", read) != "yes":
            say("Not saved.")
            return 1
        note = (f"FITTED by scripts/ring_calibrate.py on {time.strftime('%Y-%m-%d %H:%M')} from {len(points)} points "
                "(the tip at table height; the marker found by SAM in self-calibrated photos). slots_deg and "
                "camera_clear are set by hand. See mimic/ring.py and needs.md A10d.")
        ring.save_ring(ring.with_fit(the_ring, result, points, note, source="camera"), ring_path)
        say(f"Saved {ring_path}.")

    return session.guarded(arm, body, say, sleep)


if __name__ == "__main__":
    sys.exit(main())
