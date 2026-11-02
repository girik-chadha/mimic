"""Which points to teach, before teaching (scripts/teach.py).

    python scripts/teach_plan.py
    python scripts/teach_plan.py --provisional     practise with the placeholder numbers in layout.json

From calib/layout.json it lists the points to teach, in order: every slot first, then the centre of the
mess area, then its four corners. That is 8 points with 3 slots. The arm will only ever go inside the
shape these points make (their hull, plus 1 cm), so together they cover everywhere it needs to reach.
Each point needs 2 poses: hover (above the blocks) and grab (low enough to close on a block).
Any point the arm cannot reach (reach_cm: too far from its base, or too close to it) is refused.
"""
import argparse
import math
import sys

from mimic import scene

POSES_PER_POINT = ("hover", "grab")


def plan_points(layout) -> list[tuple[str, tuple[float, float]]]:
    points = [(f"slot {n}", tuple(slot)) for n, slot in enumerate(layout.slots_cm, start=1)]
    (x0, y0), (x1, y1) = layout.mess_area_cm
    points.append(("mess centre", ((x0 + x1) / 2, (y0 + y1) / 2)))
    points += [("mess bottom-left", (x0, y0)), ("mess bottom-right", (x1, y0)),
               ("mess top-right", (x1, y1)), ("mess top-left", (x0, y1))]
    return points


def main(argv=None, say=print) -> int:
    parser = argparse.ArgumentParser(description="List the points to teach with scripts/teach.py.")
    parser.add_argument("--provisional", action="store_true", help="allow the placeholder numbers in layout.json")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
