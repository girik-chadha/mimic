"""Read a photo of the sheet: where the stickers and objects are, and which object sits in which slot.

Everything SAM looks for must be found exactly once. Anything doubtful (0 or 2+ boxes, an object off
the sheet, an object between slots) stops the run with a clear message instead of sending the arm
to a guess. The fixed facts about the sheet (stickers, slots, size) live in calib/layout.json.
"""
import json
import math
from dataclasses import dataclass
from itertools import combinations

from mimic import config, sam
from mimic.calib import Calib
from mimic.calib import fit as fit_calib


class SceneError(ValueError):
    pass


@dataclass
class HeightCorrection:
    camera_height_cm: float              # how high the lens is above the table
    under_lens_cm: tuple[float, float]   # the table point straight below the lens
    point_height_cm: float               # how high above the table SAM's box centre is (about half a block)


@dataclass
class Layout:
    sticker_names: list[str]
    stickers_cm: list[tuple[float, float]]
    slots_cm: list[tuple[float, float]]
    slot_tolerance_cm: float
    sheet_min_cm: tuple[float, float]
    sheet_max_cm: tuple[float, float]
    height: HeightCorrection | None = None  # None = correction off


def load_layout(path=config.LAYOUT_PATH) -> Layout:
    with open(path) as f:
        data = json.load(f)
    height = None
    h = data["height_correction"]
    if h["enabled"]:
        if None in (h["camera_height_cm"], h["under_lens_cm"], h["point_height_cm"]):
            raise SceneError("Height correction is enabled but not all of its numbers are filled in.")
        height = HeightCorrection(h["camera_height_cm"], tuple(h["under_lens_cm"]), h["point_height_cm"])
    layout = Layout(
        sticker_names=[s["name"] for s in data["stickers"]],
        stickers_cm=[tuple(s["cm"]) for s in data["stickers"]],
        slots_cm=[tuple(s) for s in data["slots_cm"]],
        slot_tolerance_cm=data["slot_tolerance_cm"],
        sheet_min_cm=tuple(data["sheet_cm"]["min"]),
        sheet_max_cm=tuple(data["sheet_cm"]["max"]),
        height=height,
    )
    _check_slot_tolerance(layout.slots_cm, layout.slot_tolerance_cm)
    return layout


# ---------- finding things with SAM ----------

def _one_box(image_path: str, name: str) -> sam.Box:
    boxes = sam.find(image_path, name)
    if len(boxes) != 1:
        raise SceneError(f"SAM found {len(boxes)} boxes for {name!r} in {image_path}; it must find exactly 1.")
    return boxes[0]


def find_stickers(image_path: str, names: list[str]) -> list[tuple[float, float]]:
    """Pixel centre of each calibration sticker, in the same order as `names`."""
    return [_one_box(image_path, name).centre for name in names]


def calibrate(image_path: str, layout: Layout) -> Calib:
    """One photo of the sheet -> the pixel/cm map, checked by the check sticker(s)."""
    return fit_calib(find_stickers(image_path, layout.sticker_names), layout.stickers_cm)


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
