"""Which points to teach, before teaching (scripts/teach.py).

    python scripts/teach_plan.py
    python scripts/teach_plan.py --provisional     practise with the placeholder numbers in layout.json
    python scripts/teach_plan.py --push            also the extra points push mode needs (opt-in)

From calib/layout.json it lists the points to teach, in order: every slot first, then the centre of the
mess area, then its four corners. That is 8 points with 3 slots. The arm will only ever go inside the
shape these points make (their hull, plus 1 cm), so together they cover everywhere it needs to reach.
Each point needs 2 poses: hover (above the blocks) and grab (low enough to close on a block).
Any point the arm cannot reach (reach_cm: too far from its base, or too close to it) is refused.

--push adds the points push mode needs (run_mimic.py --push): the claw comes down 3 cm behind a block, on the
side away from its slot, so behind the mess area's corners (and beyond the last slot, to push an overshoot
back) it can land outside the shape above. Only points more than 1 cm outside it are listed.
"""
import argparse
import math
import sys

import cv2
import numpy as np

from mimic import kinematics, motion, scene

POSES_PER_POINT = ("hover", "grab")


def plan_points(layout) -> list[tuple[str, tuple[float, float]]]:
    points = [(f"slot {n}", tuple(slot)) for n, slot in enumerate(layout.slots_cm, start=1)]
    (x0, y0), (x1, y1) = layout.mess_area_cm
    points.append(("mess centre", ((x0 + x1) / 2, (y0 + y1) / 2)))
    points += [("mess bottom-left", (x0, y0)), ("mess bottom-right", (x1, y0)),
               ("mess top-right", (x1, y1)), ("mess top-left", (x0, y1))]
    return points


def _towards(a, b) -> tuple[float, float]:
    d = math.dist(a, b)
    return ((b[0] - a[0]) / d, (b[1] - a[1]) / d)


def push_points(layout, points: list) -> list[tuple[str, tuple[float, float]]]:
    """Where push mode's claw lands that the normal points do not cover (more than 1 cm outside their hull)."""
    (x0, y0), (x1, y1) = layout.mess_area_cm
    corners = {"bottom-left": (x0, y0), "bottom-right": (x1, y0), "top-right": (x1, y1), "top-left": (x0, y1)}
    centre = ((x0 + x1) / 2, (y0 + y1) / 2)
    candidates = []
    for label, corner in corners.items():   # a block in this corner, pushed to each slot
        for slot in layout.slots_cm:
            u = _towards(corner, slot)
            candidates.append((f"push start behind mess {label}", (corner[0] - u[0] * motion.PUSH_BEHIND_CM,
                                                                   corner[1] - u[1] * motion.PUSH_BEHIND_CM)))
    for n, slot in enumerate(layout.slots_cm, start=1):   # a block pushed past its slot, pushed back
        u = _towards(centre, slot)
        candidates.append((f"push back beyond slot {n}", (slot[0] + u[0] * motion.PUSH_BEHIND_CM,
                                                           slot[1] + u[1] * motion.PUSH_BEHIND_CM)))
    kept = []
    for label, pt in candidates:
        hull = cv2.convexHull(np.float32([p for _, p in points] + [p for _, p in kept]))
        if cv2.pointPolygonTest(hull, (float(pt[0]), float(pt[1])), True) >= -kinematics.MARGIN_CM:
            continue  # already inside the taught area, or near enough
        kept.append((label, pt))
    return kept


def main(argv=None, say=print) -> int:
    parser = argparse.ArgumentParser(description="List the points to teach with scripts/teach.py.")
    parser.add_argument("--provisional", action="store_true", help="allow the placeholder numbers in layout.json")
    parser.add_argument("--push", action="store_true", help="also list the extra points push mode needs")
    args = parser.parse_args(argv)
    layout = scene.load_layout()
    if layout.placeholder and not args.provisional:
        say("STOPPED: calib/layout.json still holds placeholder numbers. Measure the sheet, or use --provisional "
            "to practise.")
        return 1
    points = plan_points(layout)
    problems = [(label, pt, scene.reach_problem(pt, layout.reach_cm)) for label, pt in points]
    problems = [p for p in problems if p[2]]
    if problems:
        for label, pt, why in problems:
            say(f"REFUSED: {label} at ({pt[0]:.1f}, {pt[1]:.1f}) cm is {why}.")
        say("Move the slots or the mess area in layout.json so the arm can reach them, then run this again.")
        return 1

    base = layout.reach_cm["base"]
    tag = "PROVISIONAL (placeholder numbers) - " if layout.placeholder else ""
    say(f"{tag}Teach these {len(points)} points, in this order (cm from the bottom-left dot):")
    for n, (label, (x, y)) in enumerate(points, start=1):
        say(f"  {n}. {label:<18} ({x:5.1f}, {y:5.1f})   {math.dist((x, y), base):4.1f} cm from the arm's base")
    if len(points) > 8:
        say(f"  ({len(points)} points: more than the usual 6-8, because there are {len(layout.slots_cm)} slots.)")
    say(f"\nEach point needs {len(POSES_PER_POINT)} poses, {' and '.join(POSES_PER_POINT)}: "
        f"{len(points) * len(POSES_PER_POINT)} poses in all. Plus grip_open, grip_closed and camera_clear once.")
    say("The arm will only go inside the shape these points make (plus 1 cm).\n")
    say("In scripts/teach.py, at each point:")
    for label, (x, y) in points:
        say(f"  {label:<18} save hover {x:g} {y:g}   then   save grab {x:g} {y:g}")
    say("  then: save grip_open, save grip_closed, save clear, finish")
    if args.push:
        _say_push_points(layout, points, say)
    return 0


def _say_push_points(layout, points, say) -> None:
    extra = push_points(layout, points)
    say(f"\nPush mode (opt-in, run_mimic.py --push) also needs these {len(extra)} points, taught the same way:")
    for label, (x, y) in extra:
        why = scene.reach_problem((x, y), layout.reach_cm)
        if why:
            say(f"  SKIPPED: {label} at ({x:.1f}, {y:.1f}) cm is {why}: a push that needs it will be refused.")
        else:
            say(f"  {label:<36} save hover {x:.1f} {y:.1f}   then   save grab {x:.1f} {y:.1f}")
    say("Teach grab with the jaw tips 2-5 mm above the paper: push mode slides blocks at grab height.")
    say("grip_push is unknown until the grip test (needs.md C7 step 11); push mode refuses to run without it.")


if __name__ == "__main__":
    sys.exit(main())
