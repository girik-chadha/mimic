"""Pixels <-> table centimetres, from coloured stickers on the A3 sheet.

The camera sees the flat sheet at a slight angle, so equal distances on the table are not equal
in pixels. A homography is a 3x3 grid of numbers that undoes that: it takes any pixel on the sheet
to its position in cm (origin at the bottom-left sticker), and back again.

fit() builds the map from the first 4 stickers (the corners). Any stickers after those are check
points: they are NOT used to build the map, so how far off they land is an honest test of it.
If a check point lands more than 0.5 cm from where it was measured, fit() raises CalibrationError.
"""
import json
import math
from itertools import combinations

import cv2
import numpy as np

MAX_CHECK_ERROR_CM = 0.5
MIN_TRIANGLE_CM2 = 1.0  # any 3 corner stickers must make a real triangle, not a line


class CalibrationError(ValueError):
    pass


def _apply(H: np.ndarray, pt) -> tuple[float, float]:
    # Multiply [x, y, 1] by the matrix, then divide by the third number. That division is what
    # makes far-away things smaller: it is the perspective part of the map.
    x, y, w = H @ np.array([pt[0], pt[1], 1.0])
    return (float(x / w), float(y / w))


class Calib:
    def __init__(self, H, pixel_pts, table_pts_cm):
        self.H = np.asarray(H, dtype=float)
        self.H_inv = np.linalg.inv(self.H)
        self.pixel_pts = [tuple(map(float, p)) for p in pixel_pts]
        self.table_pts_cm = [tuple(map(float, p)) for p in table_pts_cm]

    def px_to_cm(self, pt) -> tuple[float, float]:
        return _apply(self.H, pt)

    def cm_to_px(self, pt) -> tuple[float, float]:
        return _apply(self.H_inv, pt)

    def check(self, pixel_pt, cm_pt) -> float:
        """How far (in cm) the map puts this pixel from where it was measured with a ruler."""
        return math.dist(self.px_to_cm(pixel_pt), cm_pt)

    def fit_report(self) -> float:
        """The worst error in cm over all stickers. The 4 corners are always ~0; check points are the real test."""
        return max(self.check(p, c) for p, c in zip(self.pixel_pts, self.table_pts_cm))

    def save(self, path) -> None:
        data = {"H": self.H.tolist(), "pixel_pts": self.pixel_pts, "table_pts_cm": self.table_pts_cm,
                "note": "first 4 points are the corner stickers that build the map; the rest are check points"}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path) -> "Calib":
        with open(path) as f:
            data = json.load(f)
        return cls(data["H"], data["pixel_pts"], data["table_pts_cm"])


def _triangle_area(a, b, c) -> float:
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2


def fit(pixel_pts: list[tuple], table_pts_cm: list[tuple]) -> Calib:
    """First 4 points: corner stickers that build the map. Points 5 and up: check stickers."""
    if len(pixel_pts) != len(table_pts_cm):
        raise ValueError(f"{len(pixel_pts)} pixel points but {len(table_pts_cm)} table points.")
    if len(pixel_pts) < 4:
        raise ValueError(f"Need at least 4 stickers, got {len(pixel_pts)}.")
    for a, b, c in combinations(table_pts_cm[:4], 3):
        if _triangle_area(a, b, c) < MIN_TRIANGLE_CM2:
            raise ValueError(f"Corner stickers {a}, {b}, {c} are in a straight line: no map is possible.")

    H, _ = cv2.findHomography(np.array(pixel_pts[:4], dtype=float), np.array(table_pts_cm[:4], dtype=float), 0)
    if H is None:
        raise ValueError("OpenCV could not fit a map to these 4 corner stickers.")
    calib = Calib(H, pixel_pts, table_pts_cm)

    problems = []
    for n, (px, cm) in enumerate(zip(pixel_pts[4:], table_pts_cm[4:]), start=5):
        error = calib.check(px, cm)
        if error > MAX_CHECK_ERROR_CM:
            mapped = calib.px_to_cm(px)
            problems.append(f"check sticker {n} is {error:.2f} cm off: pixel ({px[0]:.0f}, {px[1]:.0f}) "
                            f"maps to ({mapped[0]:.2f}, {mapped[1]:.2f}) cm but was measured at "
                            f"({cm[0]:.2f}, {cm[1]:.2f}) cm")
    if problems:
        raise CalibrationError(f"Calibration failed (limit {MAX_CHECK_ERROR_CM} cm): " + "; ".join(problems)
                               + ". Re-measure that sticker and the corner stickers.")
    return calib
