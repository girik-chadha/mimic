"""Read a photo of the sheet: where the dots and objects are, and which object sits in which slot.

Every photo calibrates itself: SAM finds the 6 dots on the sheet with one prompt, the code works out
which dot is which from where they are, builds the pixel/cm map from the 4 corner dots, and checks it
with the 2 check dots. So the camera is free to move between photos.

Everything SAM looks for must be found exactly as often as expected. Anything doubtful (wrong number of
dots, a check dot off, an object off the sheet or between slots) stops the run with a clear message
instead of sending the arm to a guess. The fixed facts about the sheet live in calib/layout.json.
"""
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from mimic import config, sam
from mimic.calib import MAX_CHECK_ERROR_CM, Calib, CalibrationError
from mimic.calib import fit as fit_calib

ROLES = ["bottom_left", "bottom_right", "top_right", "top_left", "centre_check", "right_check"]
CORNERS, CHECKS = ROLES[:4], ROLES[4:]
MIN_DOT_CLEARANCE_CM = 4.0  # slots (and so blocks) must stay this far from every dot
MAX_DOT_CANDIDATES = 12     # more dot-sized boxes than this: something is badly wrong in the view
EQUAL_FIT_CM = 0.1          # two sets of dots this close in fit count as "equally good": refuse


class SceneError(ValueError):
    pass


@dataclass
class HeightCorrection:
    camera_height_cm: float              # how high the lens is above the table
    under_lens_cm: tuple[float, float]   # the table point straight below the lens
    point_height_cm: float               # how high above the table SAM's box centre is (about half a block)


@dataclass
class Layout:
    dots_cm: dict[str, tuple[float, float]]   # role -> position in cm
    dot_prompt: str
    dot_size_frac: tuple[float, float]       # allowed dot size: box's longer side / image width
    slots_cm: list[tuple[float, float]]
    slot_tolerance_cm: float
    sheet_min_cm: tuple[float, float]
    sheet_max_cm: tuple[float, float]
    height: HeightCorrection | None = None  # None = correction off
    placeholder: bool = False               # True until the whole sheet (dots, slots, mess area, reach) is real
    dots_measured: bool = False             # True once the 6 dots are ruler-measured: enough to calibrate
    mess_area_cm: tuple | None = None       # ((x_min, y_min), (x_max, y_max)): where blocks are scattered
    reach_cm: dict | None = None            # {"base": (x, y), "min": cm, "max": cm}: what the arm can reach


def load_layout(path=config.LAYOUT_PATH) -> Layout:
    with open(path) as f:
        data = json.load(f)
    height = None
    h = data["height_correction"]
    if h["enabled"]:
        if None in (h["camera_height_cm"], h["under_lens_cm"], h["point_height_cm"]):
            raise SceneError("Height correction is enabled but not all of its numbers are filled in.")
        height = HeightCorrection(h["camera_height_cm"], tuple(h["under_lens_cm"]), h["point_height_cm"])
    if sorted(data["dots_cm"]) != sorted(ROLES):
        raise SceneError(f"layout.json dots_cm must list exactly these roles: {ROLES}")
    layout = Layout(
        dots_cm={role: tuple(data["dots_cm"][role]) for role in ROLES},
        dot_prompt=data["dot_prompt"],
        dot_size_frac=(data["dot_size_frac"]["min"], data["dot_size_frac"]["max"]),
        slots_cm=[tuple(s) for s in data["slots_cm"]],
        slot_tolerance_cm=data["slot_tolerance_cm"],
        sheet_min_cm=tuple(data["sheet_cm"]["min"]),
        sheet_max_cm=tuple(data["sheet_cm"]["max"]),
        height=height,
        placeholder=data["placeholder"],
        dots_measured=data.get("dots_measured", False),
        mess_area_cm=(tuple(data["mess_area_cm"]["min"]), tuple(data["mess_area_cm"]["max"])),
        reach_cm={"base": tuple(data["reach_cm"]["base"]), "min": data["reach_cm"]["min"],
                  "max": data["reach_cm"]["max"]},
    )
    if not layout.placeholder and not layout.dots_measured:
        raise SceneError("layout.json says placeholder is false but dots_measured is not true: the whole sheet "
                         "cannot be real while its dots are not measured.")
    _check_slot_tolerance(layout.slots_cm, layout.slot_tolerance_cm)
    for n, slot in enumerate(layout.slots_cm, start=1):
        for role, dot in layout.dots_cm.items():
            if math.dist(slot, dot) < MIN_DOT_CLEARANCE_CM:
                raise SceneError(f"Slot {n} is only {math.dist(slot, dot):.1f} cm from the {role} dot: a block "
                                 f"there would cover it. Keep slots at least {MIN_DOT_CLEARANCE_CM} cm from every dot.")
    (x0, y0), (x1, y1) = layout.mess_area_cm
    if not (x0 < x1 and y0 < y1):
        raise SceneError(f"mess_area_cm min {layout.mess_area_cm[0]} must be below and left of max {layout.mess_area_cm[1]}.")
    for role, dot in layout.dots_cm.items():
        gap = distance_to_rectangle(dot, layout.mess_area_cm)
        if gap < MIN_DOT_CLEARANCE_CM:
            raise SceneError(f"The mess area is only {gap:.1f} cm from the {role} dot: a scattered block could "
                             f"cover it. Keep the mess area at least {MIN_DOT_CLEARANCE_CM} cm from every dot.")
    if not 0 <= layout.reach_cm["min"] < layout.reach_cm["max"]:
        raise SceneError("reach_cm needs 0 <= min < max.")
    return layout


def distance_to_rectangle(point, rectangle) -> float:
    """0 inside the rectangle, otherwise the distance to its nearest edge or corner."""
    (x0, y0), (x1, y1) = rectangle
    return math.hypot(max(x0 - point[0], 0, point[0] - x1), max(y0 - point[1], 0, point[1] - y1))


def reach_problem(point, reach: dict) -> str | None:
    """Why the arm cannot reach this point, or None if it can."""
    distance = math.dist(point, reach["base"])
    if distance > reach["max"]:
        return f"{distance:.1f} cm from the arm's base, {distance - reach['max']:.1f} cm beyond its {reach['max']} cm reach"
    if distance < reach["min"]:
        return f"{distance:.1f} cm from the arm's base, too close (it cannot reach within {reach['min']} cm)"
    return None


def refuse_placeholder(layout: Layout) -> None:
    """For anything that uses the slots, the mess area or the reach (the arm, teaching): the whole file must be real."""
    if layout.placeholder or not layout.dots_measured:
        raise SceneError("calib/layout.json still holds placeholder numbers: "
                         "measure your sheet and set placeholder to false.")


def refuse_unmeasured_dots(layout: Layout) -> None:
    """For calibration, which needs only the 6 dots: refused until they are ruler-measured (dots_measured)."""
    if not layout.dots_measured:
        raise SceneError("calib/layout.json's dot positions are not measured yet: measure them with a ruler from "
                         "the bottom-left dot's centre and set dots_measured to true.")


# ---------- finding things with SAM ----------

def _one_box(image_path: str, name: str) -> sam.Box:
    boxes = sam.find(image_path, name)
    if len(boxes) != 1:
        raise SceneError(f"SAM found {len(boxes)} boxes for {name!r} in {image_path}; it must find exactly 1.")
    return boxes[0]


def dot_size(box: sam.Box) -> float:
    """A box's longer side as a fraction of the image width, so it does not depend on the photo's resolution."""
    return max(box.x2 - box.x1, box.y2 - box.y1) / box.w


def find_dots(image_path: str, layout: Layout) -> tuple[list[tuple[float, float]], list[sam.Box]]:
    """One SAM call. Returns (centres of dot-sized boxes, boxes dropped for being too small or too big)."""
    lo, hi = layout.dot_size_frac
    kept, dropped = [], []
    for box in sam.find(image_path, layout.dot_prompt):
        (kept if lo <= dot_size(box) <= hi else dropped).append(box)
    return [b.centre for b in kept], dropped


# ---------- which dot is which ----------

def _anticlockwise(points: list) -> list:
    """Order 4 corner points going round anticlockwise as seen on screen (the way BL -> BR -> TR -> TL goes).
    On screen y grows downwards, so an anticlockwise loop has a negative area sum."""
    cx = sum(p[0] for p in points) / 4
    cy = sum(p[1] for p in points) / 4
    ordered = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    area_sum = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ordered, ordered[1:] + ordered[:1]))
    return ordered if area_sum < 0 else ordered[::-1]


def _check_errors(corner_px: list, check_px: list, layout: Layout) -> list[float]:
    calib = fit_calib(corner_px, [layout.dots_cm[r] for r in CORNERS])
    return [calib.check(px, layout.dots_cm[role]) for px, role in zip(check_px, CHECKS)]


def _tries(dots_px: list, layout: Layout) -> list[tuple]:
    """Every way of naming six dots: (worst check error, turn, centres in ROLES order, check errors), best first.

    The 4 outermost dots are the corners. Walking round them anticlockwise, try each one as bottom-left
    (4 rotations), and both ways of matching the 2 inner dots to the 2 check dots: 8 orders in all.
    Mirror images are not tried yet (see docs/todo.md: the check dots must first move off the middle lines).
    """
    outer = cv2.convexHull(np.array(dots_px, dtype=np.float32), returnPoints=False).flatten().tolist()
    if len(outer) != 4:
        raise SceneError(f"The outer dots should make a four-sided shape, but {len(outer)} dots are on the "
                         "outside. Is a check dot outside the corner dots, or a corner dot missing?")
    corners = _anticlockwise([dots_px[i] for i in outer])
    inner = [p for i, p in enumerate(dots_px) if i not in outer]
    tries = []
    for turn in range(4):
        corner_px = corners[turn:] + corners[:turn]
        for check_px in (inner, inner[::-1]):
            errors = _check_errors(corner_px, check_px, layout)
            tries.append((max(errors), turn, corner_px + check_px, errors))
    return sorted(tries, key=lambda t: t[0])


def _refuse_two_turns(tries: list) -> None:
    best_turn = tries[0][1]
    if [t for t in tries if t[1] != best_turn][0][0] <= MAX_CHECK_ERROR_CM:
        raise SceneError("Two different ways of naming the dots both fit, so the photo cannot be read safely. "
                         "The check dots must not be symmetric.")


def assign_roles(dots_px: list, layout: Layout) -> list[tuple[float, float]]:
    """Exactly six dot centres in any order -> the same centres in ROLES order."""
    tries = _tries(dots_px, layout)
    best_error, _, best_order, best_errors = tries[0]
    if best_error > MAX_CHECK_ERROR_CM:
        details = ", ".join(f"{role} {e:.2f} cm off" for role, e in zip(CHECKS, best_errors))
        raise CalibrationError(f"No way of naming the dots fits: the best has {details} (limit "
                               f"{MAX_CHECK_ERROR_CM} cm). Are the dot positions in layout.json right?")
    _refuse_two_turns(tries)
    return best_order


def choose_dots(candidates: list, layout: Layout) -> tuple[list, list]:
    """Dot-sized centres (6 to 12) -> (the 6 dots in ROLES order, the leftovers).

    With exactly 6, this is assign_roles. With more (a claw marker, a table speck, a block that looks like a dot),
    every set of 6 is tried in its 8 orders, and the set whose check dots land within 0.5 cm is kept.
    If another set fits just as well (worst check error within 0.1 cm of the best), the photo is refused.
    """
    if len(candidates) < len(ROLES):
        raise SceneError(f"Found {len(candidates)} dots, need exactly {len(ROLES)}: {_list(candidates)}.")
    if len(candidates) == len(ROLES):
        return assign_roles(candidates, layout), []
    if len(candidates) > MAX_DOT_CANDIDATES:
        raise SceneError(f"Found {len(candidates)} dot-sized boxes, more than {MAX_DOT_CANDIDATES}: "
                         f"{_list(candidates)}. Clear the sheet of small things that look like dots.")
    fits = []
    for chosen in combinations(range(len(candidates)), len(ROLES)):
        try:
            tries = _tries([candidates[i] for i in chosen], layout)
        except SceneError:
            continue  # these six do not have 4 outer corners
        if tries[0][0] <= MAX_CHECK_ERROR_CM:
            fits.append((tries[0][0], chosen, tries))
    if not fits:
        raise CalibrationError(f"Found {len(candidates)} dot-sized boxes, but no 6 of them fit the layout "
                               f"(check dots within {MAX_CHECK_ERROR_CM} cm): {_list(candidates)}.")
    fits.sort(key=lambda f: f[0])
    if len(fits) > 1 and fits[1][0] - fits[0][0] <= EQUAL_FIT_CM:
        a, b = set(fits[0][1]), set(fits[1][1])
        clash = [candidates[i] for i in sorted(a ^ b)]
        raise SceneError(f"Two different sets of 6 dots fit the layout equally well ({fits[0][0]:.2f} and "
                         f"{fits[1][0]:.2f} cm); they differ at {_list(clash)}. Is something dot-sized touching a dot?")
    _, chosen, tries = fits[0]
    _refuse_two_turns(tries)
    return tries[0][2], [candidates[i] for i in range(len(candidates)) if i not in chosen]


def _list(points: list) -> str:
    return ", ".join(f"({x:.0f}, {y:.0f})" for x, y in points) or "none"


# ---------- calibrating one photo ----------

def draw_dots(image_path: str, dots_px: list, out_path, calib: Calib | None = None, leftovers=()) -> None:
    """Save the photo with each dot circled and named by its role (or '?' if the roles are not known).
    Dot-sized boxes that were not used (leftovers) are marked with an x."""
    img = cv2.imread(str(image_path))
    if img is None:
        return
    scale = max(1.0, img.shape[1] / 1280)
    for n, (x, y) in enumerate(dots_px):
        label = ROLES[n] if calib else "?"
        if calib and label in CHECKS:
            label += f" {calib.check((x, y), calib.table_pts_cm[n]):.2f} cm"
        cv2.circle(img, (int(x), int(y)), int(18 * scale), (0, 0, 255), 2)
        cv2.putText(img, label, (int(x + 22 * scale), int(y + 6 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, (0, 0, 255), max(1, int(2 * scale)))
    for x, y in leftovers:
        cv2.drawMarker(img, (int(x), int(y)), (0, 140, 255), cv2.MARKER_TILTED_CROSS, int(24 * scale), 2)
        cv2.putText(img, "x not a dot", (int(x + 16 * scale), int(y - 10 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, (0, 140, 255), max(1, int(scale)))
    if calib:
        corners = np.array([[int(x), int(y)] for x, y in dots_px[:4]], dtype=np.int32)
        cv2.polylines(img, [corners], True, (0, 160, 0), 1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def calibrate(image_path: str, layout: Layout, labelled_path=None) -> Calib:
    """One photo -> its own pixel/cm map, from its own dots. If `labelled_path` is given, the photo is saved
    there with every dot named by role (or marked '?' when calibration stops), so the roles can be checked by eye.
    """
    refuse_unmeasured_dots(layout)  # only the dots: slots and reach may still be placeholders
    dots, dropped = find_dots(image_path, layout)
    try:
        try:
            ordered, leftovers = choose_dots(dots, layout)
        except SceneError as error:
            if len(dots) < len(ROLES) and dropped:
                error.args = (f"{error} Dropped for size: " + ", ".join(
                    f"({b.centre[0]:.0f}, {b.centre[1]:.0f}) {dot_size(b):.4f}" for b in dropped) + ".",)
            if len(dots) < len(ROLES):
                error.args = (f"{error} Is a dot covered or out of view?",)
            raise
        calib = fit_calib(ordered, [layout.dots_cm[r] for r in ROLES], names=ROLES)
    except (SceneError, CalibrationError) as error:
        if labelled_path:
            draw_dots(image_path, dots, labelled_path)
            error.args = (f"{error} Labelled photo: {labelled_path}",)
        raise
    if labelled_path:
        draw_dots(image_path, ordered, labelled_path, calib, leftovers)
    return calib


# ---------- block height ----------

def correct_for_height(pt_cm, height: HeightCorrection | None) -> tuple[float, float]:
    """The map is only right for things flat on the sheet. A point raised above it (the middle of a block)
    looks further from the spot under the lens than it really is, so pull it back towards that spot.
    """
    if height is None:
        return (float(pt_cm[0]), float(pt_cm[1]))
    shrink = (height.camera_height_cm - height.point_height_cm) / height.camera_height_cm
    ux, uy = height.under_lens_cm
    return (ux + (pt_cm[0] - ux) * shrink, uy + (pt_cm[1] - uy) * shrink)


# ---------- reading a photo ----------

def read_positions(image_path: str, names: list[str], calib: Calib, layout: Layout) -> dict[str, tuple[float, float]]:
    """Where each named object is on the table, in cm."""
    positions = {}
    for name in names:
        pt = correct_for_height(calib.px_to_cm(_one_box(image_path, name).centre), layout.height)
        (x0, y0), (x1, y1) = layout.sheet_min_cm, layout.sheet_max_cm
        if not (x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1):
            raise SceneError(f"{name!r} is at ({pt[0]:.1f}, {pt[1]:.1f}) cm, which is off the sheet: "
                             "SAM has probably found something else.")
        positions[name] = pt
    return positions


def _check_slot_tolerance(slots_cm, tolerance_cm: float) -> None:
    if len(slots_cm) < 2:
        return
    spacing = min(math.dist(a, b) for a, b in combinations(slots_cm, 2))
    if tolerance_cm > spacing / 2:
        raise SceneError(f"Slot tolerance {tolerance_cm} cm is more than half the slot spacing ({spacing:.2f} cm): "
                         "an object could count as being in two slots.")


def assign_slots(positions_cm: dict, slots_cm: list[tuple], tolerance_cm: float) -> list[str]:
    """Put each object in its nearest slot and return the names in slot order (empty slots are skipped)."""
    _check_slot_tolerance(slots_cm, tolerance_cm)
    slot_of = {}
    for name, pt in positions_cm.items():
        distances = [math.dist(pt, slot) for slot in slots_cm]
        nearest = distances.index(min(distances))
        if distances[nearest] > tolerance_cm:
            raise SceneError(f"{name!r} is {distances[nearest]:.1f} cm from the nearest slot (slot {nearest + 1}), "
                             f"more than {tolerance_cm} cm: it is not in a slot.")
        for other, slot in slot_of.items():
            if slot == nearest:
                raise SceneError(f"{other!r} and {name!r} are both in slot {nearest + 1}.")
        slot_of[name] = nearest
    return sorted(slot_of, key=slot_of.get)


def read_target(image_path: str, names: list[str], calib: Calib, layout: Layout) -> list[str]:
    """Demo photo -> the names in slot order (slot 1 first)."""
    positions = read_positions(image_path, names, calib, layout)
    return assign_slots(positions, layout.slots_cm, layout.slot_tolerance_cm)
