"""Sheet centimetres -> arm angles, from a grid of taught poses (calib/pose_grid.json).

No arm geometry maths: at each taught point we stored, by hand, the angles that put the gripper
above it (hover) and low enough to grab (grab). For any other point, to_angles blends the poses of
the 4 nearest taught points, giving closer points more say (weight 1 / distance squared). At a
taught point it returns exactly what was taught. The arm is only sent inside the area the taught
points cover (their convex hull) plus a 1 cm margin.
"""
import json
import math
from dataclasses import dataclass

import cv2
import numpy as np

from mimic import config

ARM_JOINTS = ("base", "shoulder", "elbow")   # the grip is separate: any pose can be open or closed
HEIGHTS = ("hover", "grab")
MIN_POINTS = 4
NEAREST = 4
MARGIN_CM = 1.0
POSE_GRID_PATH = config.ROOT / "calib" / "pose_grid.json"


class KinematicsError(ValueError):
    pass


@dataclass
class TaughtPoint:
    x_cm: float
    y_cm: float
    hover: dict | None = None   # {"base", "shoulder", "elbow"}
    grab: dict | None = None


@dataclass
class PoseGrid:
    points: list[TaughtPoint]
    grip_open: int | None
    grip_closed: int | None
    camera_clear: dict | None   # {"base", "shoulder", "elbow"}, used with the grip open
    placeholder: bool
    grip_push: int | None = None  # push mode only: 2 degrees more open than where the empty jaw tips touch

    def complete_points(self) -> list[TaughtPoint]:
        return [p for p in self.points if p.hover and p.grab]

    def check_ready(self) -> None:
        if self.placeholder:
            raise KinematicsError("calib/pose_grid.json is still a placeholder: teach it with scripts/teach.py.")
        if len(self.complete_points()) < MIN_POINTS:
            raise KinematicsError(f"The pose grid needs at least {MIN_POINTS} points with both hover and grab; "
                                  f"it has {len(self.complete_points())}.")
        if self.grip_open is None or self.grip_closed is None or self.camera_clear is None:
            raise KinematicsError("The pose grid needs grip_open, grip_closed and camera_clear.")

    def check_against(self, limits) -> None:
        """Every taught angle must be inside the arm's limits (calib/arm_limits.json)."""
        lo, hi = limits.range_of("grip")
        for name, value in (("grip_open", self.grip_open), ("grip_closed", self.grip_closed),
                            ("grip_push", self.grip_push)):
            if value is not None and not lo <= value <= hi:
                raise KinematicsError(f"{name} {value} is outside the grip range {lo}-{hi} in arm_limits.json.")
        if self.grip_push is not None and self.grip_push == limits.grip_closed_limit_deg:
            raise KinematicsError(f"grip_push {self.grip_push} is the closed limit: pushing there would stall the "
                                  "claw, and a stalled claw overheats. Use 2 degrees more open than where the tips just touch.")
        poses = [("camera_clear", self.camera_clear)] + [
            (f"{h} at ({p.x_cm}, {p.y_cm})", getattr(p, h)) for p in self.points for h in HEIGHTS]
        for label, pose in poses:
            for joint in ARM_JOINTS:
                if pose:
                    lo, hi = limits.range_of(joint)
                    if not lo <= pose[joint] <= hi:
                        raise KinematicsError(f"{label}: {joint} {pose[joint]} is outside its range {lo}-{hi}.")

    def to_angles(self, x_cm: float, y_cm: float, height: str) -> dict:
        """The (base, shoulder, elbow) angles that put the gripper at hover or grab height over this point."""
        if height not in HEIGHTS:
            raise KinematicsError(f"height must be one of {HEIGHTS}, got {height!r}")
        self.check_ready()
        taught = [p for p in self.points if getattr(p, height)]

        hull = cv2.convexHull(np.array([[p.x_cm, p.y_cm] for p in taught], dtype=np.float32))
        inside_by = cv2.pointPolygonTest(hull, (float(x_cm), float(y_cm)), True)  # negative = outside
        if inside_by < -MARGIN_CM:
            raise KinematicsError(f"({x_cm:.1f}, {y_cm:.1f}) cm is {-inside_by:.1f} cm outside the taught area "
                                  f"(allowed: {MARGIN_CM} cm). Teach a point nearer there.")

        by_distance = sorted(taught, key=lambda p: math.dist((p.x_cm, p.y_cm), (x_cm, y_cm)))[:NEAREST]
        nearest = by_distance[0]
        if math.dist((nearest.x_cm, nearest.y_cm), (x_cm, y_cm)) < 1e-6:
            return {j: float(getattr(nearest, height)[j]) for j in ARM_JOINTS}  # exactly a taught point
        weights = [1 / math.dist((p.x_cm, p.y_cm), (x_cm, y_cm)) ** 2 for p in by_distance]
        total = sum(weights)
        return {j: sum(w * getattr(p, height)[j] for w, p in zip(weights, by_distance)) / total
                for j in ARM_JOINTS}


def load_grid(path=POSE_GRID_PATH) -> PoseGrid:
    with open(path) as f:
        data = json.load(f)
    return PoseGrid(
        points=[TaughtPoint(p["x_cm"], p["y_cm"], p.get("hover"), p.get("grab")) for p in data["points"]],
        grip_open=data["grip_open"],
        grip_closed=data["grip_closed"],
        camera_clear=data["camera_clear"],
        placeholder=data["placeholder"],
        grip_push=data.get("grip_push"),
    )


def save_grid(grid: PoseGrid, path=POSE_GRID_PATH, note: str | None = None) -> None:
    old_note = None
    try:
        with open(path) as f:
            old_note = json.load(f).get("note")
    except (OSError, ValueError):
        pass
    data = {
        "placeholder": grid.placeholder,
        "note": note or old_note or "Taught poses, see mimic/kinematics.py.",
        "grip_open": grid.grip_open,
        "grip_closed": grid.grip_closed,
        "camera_clear": grid.camera_clear,
        "grip_push": grid.grip_push,
        "points": [{"x_cm": p.x_cm, "y_cm": p.y_cm, "hover": p.hover, "grab": p.grab} for p in grid.points],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
