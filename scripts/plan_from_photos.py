"""Gate 1: two photos in, a plan out. No arm needed.

    python scripts/plan_from_photos.py demo.jpg mess.jpg

demo.jpg: the objects arranged in the slots the way you want them.
mess.jpg: the same objects scattered in the mess area. The camera may move between the photos:
each photo calibrates itself from the 6 dots on the sheet (all 6 must be visible).

Prints the names Spark chose, where each object is in cm, and the list of moves, then saves
annotated photos, each photo's labelled dots and plan.json to runs/<date-time>/.
"""
import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from mimic import brain, calib, config, meta_api, plan, sam, scene

RUN_BUDGET_USD = 0.15     # hard cap for one run of this script

# Drawing colours (OpenCV uses blue, green, red order). An object is drawn in the colour its name says.
NAMED_COLOURS = {"red": (0, 0, 220), "blue": (220, 90, 0), "green": (0, 160, 0), "yellow": (0, 190, 230),
                 "orange": (0, 130, 255), "purple": (170, 0, 140), "pink": (180, 100, 255), "black": (30, 30, 30)}
OTHER_COLOURS = [(200, 0, 200), (0, 200, 200), (120, 120, 0)]


def say(text=""):
    print(text, flush=True)


def check_dot_errors(c: calib.Calib) -> dict[str, float]:
    """How far (cm) each check dot lands from where layout.json says it is."""
    return {role: c.check(px, cm) for role, px, cm in zip(scene.ROLES, c.pixel_pts, c.table_pts_cm)
            if role in scene.CHECKS}


def calibrate_photo(label: str, photo: str, layout: scene.Layout, out_dir: Path) -> calib.Calib:
    labelled = out_dir / f"{label}_dots.jpg"
    try:
        c = scene.calibrate(photo, layout, labelled)  # stops if the dots are wrong or a check dot is > 0.5 cm off
    finally:
        say(f"  {label} photo: labelled dots saved to {labelled}")
    errors = ", ".join(f"{role} {e:.2f} cm" for role, e in check_dot_errors(c).items())
    say(f"  {label} photo: check dots {errors} (limit {calib.MAX_CHECK_ERROR_CM} cm)")
    return c


# ---------- drawing ----------

def colour_for(name: str, index: int):
    for word in name.split():
        if word in NAMED_COLOURS:
            return NAMED_COLOURS[word]
    return OTHER_COLOURS[index % len(OTHER_COLOURS)]


def _pt(p):
    return (int(round(p[0])), int(round(p[1])))


def _label(img, text, at, colour):
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4)
    cv2.putText(img, text, at, cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)


def _draw_sheet(img, c: calib.Calib, layout: scene.Layout):
    for px in c.pixel_pts:
        cv2.drawMarker(img, _pt(px), (0, 0, 0), cv2.MARKER_CROSS, 24, 2)
    for n, slot in enumerate(layout.slots_cm, start=1):
        centre = c.cm_to_px(slot)
        cv2.circle(img, _pt(centre), 18, (60, 60, 60), 2)
        _label(img, str(n), _pt((centre[0] - 6, centre[1] + 7)), (60, 60, 60))


def draw_demo(photo, out_path, c, layout, names, target):
    img = cv2.imread(photo)
    _draw_sheet(img, c, layout)
    for i, name in enumerate(names):
        colour = colour_for(name, i)
        for b in sam.find(photo, name):  # same request as before, so it comes from the cache
            cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), colour, 3)
            slot = target.index(name) + 1
            _label(img, f"{slot}: {name}", (b.x1, b.y1 - 8), colour)
    cv2.imwrite(str(out_path), img)


def draw_mess(photo, out_path, c, layout, names, positions, moves):
    img = cv2.imread(photo)
    _draw_sheet(img, c, layout)
    for i, name in enumerate(names):
        colour = colour_for(name, i)
        for b in sam.find(photo, name):
            cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), colour, 3)
            x, y = positions[name]
            _label(img, f"{name} ({x:.1f}, {y:.1f}) cm", (b.x1, b.y1 - 8), colour)
    for move in moves:
        colour = colour_for(move.name, names.index(move.name))
        start = _pt(c.cm_to_px(move.from_cm))
        end = _pt(c.cm_to_px(move.to_cm))
        cv2.arrowedLine(img, start, end, colour, 3, tipLength=0.04)
    cv2.imwrite(str(out_path), img)


# ---------- the run ----------

def run(demo: str, mess: str, layout_path=config.LAYOUT_PATH, out_root=config.RUNS_DIR) -> dict:
    layout = scene.load_layout(layout_path)
    scene.refuse_placeholder(layout)  # before spending anything

    meta_api.reset()
    budget_before = config.BUDGET_USD
    config.BUDGET_USD = min(config.BUDGET_USD, RUN_BUDGET_USD)
    try:
        out_dir = Path(out_root) / time.strftime("%Y%m%d-%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        say("Calibration (each photo from its own dots)")
        demo_map = calibrate_photo("demo", demo, layout, out_dir)
        mess_map = calibrate_photo("mess", mess, layout, out_dir)

        attempts = []
        names = brain.name_objects(demo, attempts)
        say()
        say("Spark named: " + ", ".join(names))
        for a in attempts:
            say(f"  round {a['round']}: {a['name']!r} -> {a['boxes']} box(es)")

        target = scene.read_target(demo, names, demo_map, layout)
        say()
        say("Target order (slot 1 first): " + ", ".join(target))

        positions = scene.read_positions(mess, target, mess_map, layout)
        say()
        say("Positions on the mess photo (cm):")
        for name in target:
            say(f"  {name:<20} ({positions[name][0]:6.1f}, {positions[name][1]:6.1f})")

        moves = plan.plan_moves(positions, target, layout.slots_cm)
        say()
        say("Moves:")
        for m in moves:
            say(f"  {m.slot}. {m.name:<20} ({m.from_cm[0]:5.1f}, {m.from_cm[1]:5.1f}) -> "
                f"slot {m.slot} ({m.to_cm[0]:5.1f}, {m.to_cm[1]:5.1f})")

        shutil.copy(demo, out_dir / "demo.jpg")
        shutil.copy(mess, out_dir / "mess.jpg")
        draw_demo(demo, out_dir / "demo_annotated.jpg", demo_map, layout, names, target)
        draw_mess(mess, out_dir / "mess_annotated.jpg", mess_map, layout, names, positions, moves)

        result = {
            "demo": str(demo), "mess": str(mess),
            "check_dot_errors_cm": {"demo": check_dot_errors(demo_map), "mess": check_dot_errors(mess_map)},
            "names": names, "naming_attempts": attempts,
            "target_order": target,
            "positions_cm": positions,
            "moves": [asdict(m) for m in moves],
            "cost_usd": meta_api.cost_so_far(),
            "out_dir": str(out_dir),
        }
        (out_dir / "plan.json").write_text(json.dumps(result, indent=2))
        say()
        say(f"Cost this run: ${meta_api.cost_so_far():.4f} (cap ${config.BUDGET_USD:.2f})")
        say(f"Saved: {out_dir}")
        return result
    finally:
        config.BUDGET_USD = budget_before


def main():
    parser = argparse.ArgumentParser(description="Two photos in (demo + mess), a plan out.")
    parser.add_argument("demo", help="photo of the objects arranged in the slots")
    parser.add_argument("mess", help="photo of the same objects scattered (the camera may have moved)")
    parser.add_argument("--layout", default=str(config.LAYOUT_PATH), help="sheet layout (default calib/layout.json)")
    args = parser.parse_args()
    try:
        run(args.demo, args.mess, args.layout)
    except (scene.SceneError, calib.CalibrationError, brain.NamingFailed, meta_api.BudgetExceeded, ValueError) as error:
        say()
        say(f"STOPPED: {error}")
        say(f"Cost so far: ${meta_api.cost_so_far():.4f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
