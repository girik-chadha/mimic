"""The ring: with the elbow taped (needs.md A10d), the claw tip reaches the table only on a circle around the base axis.

    the tip at base angle theta is at  (cx + R cos phi, cy + R sin phi)  in sheet cm,  with  theta = scale * phi + offset

phi is the direction from the circle's centre, in degrees. scale is 1 (or -1) if one servo degree is one real
degree. The fit uses the base angles as well as the positions, so even points on a short arc pin the circle down:
for a fixed scale, the circle is a plain least-squares fit. With 4 or more points other scales (0.85-1.15) are tried
too, but one is only kept if it fits clearly better than 1 (an error at least 25% smaller): on a short arc a
different scale and a different radius describe almost the same points, so small camera errors alone would
otherwise pick a wrong scale. Inside the calibrated arc both descriptions put the claw in the same places.

calib/ring.json holds the fit (from scripts/ring_calibrate.py), the slots (base angles on the ring, set by hand),
a camera_clear pose (by hand), and the pick numbers that ring_calibrate.py and run_mimic.py --ring use: shoulder_table
(the shoulder angle at table height), grip_hold, lift, approach, command_s (seconds per command for the bridge) and
stop_at. stop_at may be left null: then the shoulder is never sent more than STOP_MARGIN_DEG past shoulder_table.
A point more than max_off_ring_cm from the circle is refused: the claw cannot reach it.
"""
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from mimic import config

RING_PATH = config.ROOT / "calib" / "ring.json"
MAX_POINT_ERROR_CM = 0.5      # a calibration point further than this from the fitted ring: refuse the fit
MIN_SPAN_DEG = 20             # calibration base angles must cover at least this much
MIN_POINTS = 3
MIN_POINTS_FOR_SCALE = 4      # with fewer, one servo degree is taken to be one real degree
SCALE_RANGE = (0.85, 1.15)
SCALE_STEP = 0.005
CLEARLY_BETTER = 0.75         # a scale other than 1 is kept only if its error is at most this fraction of scale 1's
STOP_MARGIN_DEG = 3           # with stop_at null: the shoulder is never sent past shoulder_table + this


class RingError(ValueError):
    pass


@dataclass
class Fit:
    centre_cm: tuple[float, float]
    radius_cm: float
    scale: float
    offset_deg: float
    errors_cm: list[float]    # each calibration point's distance from where the fit puts it
    scale_note: str = ""      # how the scale was chosen, for the printout


@dataclass
class Ring:
    placeholder: bool = True
    note: str = ""
    centre_cm: tuple | None = None
    radius_cm: float | None = None
    scale: float | None = None
    offset_deg: float | None = None
    points: list = field(default_factory=list)  # [{"base_deg", "x_cm", "y_cm", "error_cm"}]
    slots_deg: list = field(default_factory=list)
    camera_clear: dict | None = None             # {"base", "shoulder"}: the arm out of the photo
    shoulder_table: int = 173
    grip_hold: int = 23
    lift: int = 20
    approach: int = 10
    stop_at: int | None = None   # None: shoulder_table + STOP_MARGIN_DEG (see shoulder_stop)
    command_s: float = 0.05
    max_off_ring_cm: float = 1.0
    marker_name: str = "small yellow rectangle"  # the SAM prompt for the claw-tip marker (ring_calibrate.py)

    @property
    def shoulder_stop(self) -> int:
        """The shoulder is never sent past this: stop_at if set, otherwise 3 degrees past the table angle."""
        return self.stop_at if self.stop_at is not None else self.shoulder_table + STOP_MARGIN_DEG

    # ---------- using the fit ----------

    def _needs_fit(self) -> None:
        if self.radius_cm is None:
            raise RingError("calib/ring.json has no fit yet: run scripts/ring_calibrate.py.")

    def point_at(self, base_deg: float) -> tuple[float, float]:
        """Where the claw tip meets the table at this base angle (sheet cm)."""
        self._needs_fit()
        phi = math.radians((base_deg - self.offset_deg) / self.scale)
        return (self.centre_cm[0] + self.radius_cm * math.cos(phi), self.centre_cm[1] + self.radius_cm * math.sin(phi))

    def off_ring_cm(self, x: float, y: float) -> float:
        """How far a point is from the ring, along the radius (+ outside, - inside)."""
        self._needs_fit()
        return math.dist((x, y), self.centre_cm) - self.radius_cm

    def angle_for(self, x: float, y: float) -> float:
        """The base angle that points the claw at (x, y), whether or not the claw can reach it."""
        self._needs_fit()
        phi = math.degrees(math.atan2(y - self.centre_cm[1], x - self.centre_cm[0]))
        theta = self.scale * phi + self.offset_deg
        period = 360 * abs(self.scale)  # phi is only known up to whole turns: pick the turn nearest the calibration
        middle = np.mean([p["base_deg"] for p in self.points]) if self.points else 90.0
        return theta + period * round((middle - theta) / period)

    def base_angle_for(self, x: float, y: float) -> float:
        """The base angle for a point on the ring; refused if it is more than max_off_ring_cm off it."""
        off = self.off_ring_cm(x, y)
        if abs(off) > self.max_off_ring_cm:
            raise RingError(f"({x:.1f}, {y:.1f}) cm is {abs(off):.1f} cm {'outside' if off > 0 else 'inside'} the "
                            f"ring (allowed {self.max_off_ring_cm} cm): the claw cannot reach it.")
        return self.angle_for(x, y)

    def check_ready(self) -> None:
        """Everything ring mode needs, before anything is opened."""
        if self.placeholder:
            raise RingError("calib/ring.json is still a placeholder: run scripts/ring_calibrate.py.")
        self._needs_fit()
        if not self.slots_deg:
            raise RingError("calib/ring.json has no slots_deg: set the slots' base angles by hand.")
        if not self.camera_clear:
            raise RingError("calib/ring.json has no camera_clear pose (base and shoulder): set it by hand.")


# ---------- fitting ----------

def _fit_for_scale(points: list, scale: float):
    """For one scale the fit is linear: x = cx + A cos(psi) - B sin(psi), y = cy + A sin(psi) + B cos(psi),
    with psi = theta / scale, R = hypot(A, B) and the rotation atan2(B, A)."""
    rows, values = [], []
    for theta, x, y in points:
        psi = math.radians(theta / scale)
        rows += [[1, 0, math.cos(psi), -math.sin(psi)], [0, 1, math.sin(psi), math.cos(psi)]]
        values += [x, y]
    (cx, cy, a, b), *_ = np.linalg.lstsq(np.array(rows), np.array(values), rcond=None)
    radius, rotation = math.hypot(a, b), math.degrees(math.atan2(b, a))
    offset = -scale * rotation  # theta = scale * (phi - rotation)
    ring = Ring(centre_cm=(float(cx), float(cy)), radius_cm=radius, scale=scale, offset_deg=offset)
    errors = [math.dist(ring.point_at(theta), (x, y)) for theta, x, y in points]
    return ring, errors


def _rms(errors) -> float:
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def _best_fit(points: list):
    """The fit with scale 1, or (with 4+ points) another scale if it fits clearly better. Returns (ring, errors, note)."""
    def best_of(sizes):
        fits = [_fit_for_scale(points, sign * size) for size in sizes for sign in (1, -1)]
        return min(fits, key=lambda f: _rms(f[1]))

    ring, errors = best_of([1.0])
    if len(points) < MIN_POINTS_FOR_SCALE:
        return ring, errors, "scale 1 (one servo degree = one real degree)"
    steps = round((SCALE_RANGE[1] - SCALE_RANGE[0]) / SCALE_STEP)
    sizes = [SCALE_RANGE[0] + i * SCALE_STEP for i in range(steps + 1)]
    other, other_errors = best_of(sizes)
    if _rms(other_errors) > CLEARLY_BETTER * _rms(errors):
        return ring, errors, "scale kept at 1: no other scale fits clearly better on this arc"
    if abs(other.scale) in (sizes[0], sizes[-1]):
        raise RingError(f"The best base scale is {abs(other.scale):.3f}, at the edge of {SCALE_RANGE[0]}-"
                        f"{SCALE_RANGE[1]}: the points do not look like one ring. Check the marker and the tape.")
    return other, other_errors, f"scale {abs(other.scale):.3f} fitted (clearly better than 1)"


def _likely_bad_point(points: list) -> str:
    """Which single point, left out, lets the others fit: a least-squares fit spreads one bad point's error onto its
    neighbours, so the worst error is not always the bad point."""
    if len(points) < MIN_POINTS + 1:
        return ""
    best = None
    for i in range(len(points)):
        rest = points[:i] + points[i + 1:]
        try:
            _, errors, _ = _best_fit(rest)
        except RingError:
            continue
        if best is None or max(errors) < best[1]:
            best = (points[i][0], max(errors))
    if best and best[1] <= MAX_POINT_ERROR_CM:
        return f" Without the point at base {best[0]:g} the others fit within {best[1]:.2f} cm: re-measure that one."
    return ""


def fit(points: list) -> Fit:
    """points: [(base_deg, x_cm, y_cm), ...], at least 3, from the claw tip at table height."""
    if len(points) < MIN_POINTS:
        raise RingError(f"The ring needs at least {MIN_POINTS} calibration points; there are {len(points)}.")
    angles = [p[0] for p in points]
    if len(set(angles)) != len(angles):
        raise RingError(f"Two calibration points have the same base angle: {sorted(angles)}.")
    if max(angles) - min(angles) < MIN_SPAN_DEG:
        raise RingError(f"The calibration base angles cover only {max(angles) - min(angles):g} degrees; at least "
                        f"{MIN_SPAN_DEG} are needed.")
    ring, errors, scale_note = _best_fit(points)
    worst = max(range(len(points)), key=lambda i: errors[i])
    if errors[worst] > MAX_POINT_ERROR_CM:
        raise RingError(f"The point at base {points[worst][0]:g} is {errors[worst]:.2f} cm from the fitted ring "
                        f"(limit {MAX_POINT_ERROR_CM} cm): was the marker found in the right place, or did the tape "
                        "slip?" + _likely_bad_point(points))
    return Fit(ring.centre_cm, ring.radius_cm, ring.scale, ring.offset_deg, errors, scale_note)


# ---------- the file ----------

def load_ring(path=RING_PATH) -> Ring:
    with open(path) as f:
        data = json.load(f)
    data["centre_cm"] = tuple(data["centre_cm"]) if data.get("centre_cm") else None
    return Ring(**data)


def save_ring(ring: Ring, path=RING_PATH) -> None:
    data = asdict(ring)
    data["centre_cm"] = list(ring.centre_cm) if ring.centre_cm else None
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def with_fit(ring: Ring, result: Fit, points: list, note: str) -> Ring:
    """A copy of ring with a new fit (slots, camera_clear and the pick numbers are kept)."""
    data = asdict(ring)
    data.update(placeholder=False, note=note, centre_cm=result.centre_cm, radius_cm=result.radius_cm,
                scale=result.scale, offset_deg=result.offset_deg,
                points=[{"base_deg": t, "x_cm": round(x, 2), "y_cm": round(y, 2), "error_cm": round(e, 3)}
                        for (t, x, y), e in zip(points, result.errors_cm)])
    return Ring(**data)
