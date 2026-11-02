"""Teach the pose grid: jog the arm over a spot on the sheet, then save its pose (calib/pose_grid.json).

    python scripts/teach.py --dry-run      practise without an arm (saves to runs/pose_grid_dryrun.json)
    python scripts/teach.py                the real arm (refused while the arm is locked)

Teach 6-8 points covering every slot and the whole mess area: the arm will only go inside the hull of
the taught points, plus 1 cm. At each point save a hover pose (above the blocks) and a grab pose (low
enough to close on a block). Every jog and set goes through Arm.move, so the range and speed limits apply.
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

from mimic import arm as arm_module
from mimic import config, kinematics, session
from mimic.arm import JOINTS, Arm, ArmError
from mimic.kinematics import ARM_JOINTS, KinematicsError, TaughtPoint

JOG_STEPS = (1, 2, 5, 10)
MIN_STEP_MS = 400
DRY_RUN_GRID = config.RUNS_DIR / "pose_grid_dryrun.json"
HELP = """commands:
  base|shoulder|elbow|grip +N / -N    jog one joint by N degrees (N = 1, 2, 5 or 10)
  set <joint> <angle>                 move one joint to an angle
  save hover <x> <y>                  save the current pose as hover over (x, y) cm
  save grab <x> <y>                   save the current pose as grab over (x, y) cm
  save grip_open | save grip_closed   save the current grip angle
  save clear                          save the current pose as camera_clear (arm out of the photo)
  list                                show what is saved
  finish                              check the grid and mark it as taught (asks for yes)
  home | status | stop | quit         s (or Ctrl+C) stops the arm and exits"""


class Teacher:
    def __init__(self, arm: Arm, grid_path, ask=input):
        self.arm = arm
        self.grid_path = Path(grid_path)
        self.ask = ask
        self.grid = kinematics.load_grid(self.grid_path)

    def _move_to(self, pose: dict) -> None:
        current = self.arm.known_pose()
        biggest = max(abs(round(pose[j]) - round(current[j])) for j in JOINTS)
        self.arm.move(pose, ms=max(MIN_STEP_MS, biggest * self.arm.limits.min_ms_per_degree))

    def _save(self) -> None:
        kinematics.save_grid(self.grid, self.grid_path)

    def _point_at(self, x: float, y: float) -> TaughtPoint:
        for point in self.grid.points:
            if abs(point.x_cm - x) < 0.05 and abs(point.y_cm - y) < 0.05:
                return point
        point = TaughtPoint(x, y)
        self.grid.points.append(point)
        return point

    def handle(self, line: str) -> str:
        words = line.strip().lower().split()
        try:
            if words in (["home"], ["status"], ["stop"]):
                if words == ["home"]:
                    self.arm.home()
                elif words == ["stop"]:
                    self.arm.stop()
                return str(self.arm.status()) if words == ["status"] else words[0]
            if len(words) == 2 and words[0] in JOINTS and re.fullmatch(r"[+-]\d+", words[1]):
                step = int(words[1])
                if abs(step) not in JOG_STEPS:
                    return f"refused: jog steps are {', '.join(map(str, JOG_STEPS))} degrees"
                pose = self.arm.known_pose()
                pose[words[0]] += step
                self._move_to(pose)
                return f"{words[0]} now {pose[words[0]]}"
            if len(words) == 3 and words[0] == "set" and words[1] in JOINTS and re.fullmatch(r"\d+", words[2]):
                pose = self.arm.known_pose()
                pose[words[1]] = int(words[2])
                self._move_to(pose)
                return f"{words[1]} now {pose[words[1]]}"
            if len(words) == 4 and words[0] == "save" and words[1] in ("hover", "grab"):
                x, y = float(words[2]), float(words[3])
                pose = self.arm.known_pose()
                setattr(self._point_at(x, y), words[1], {j: pose[j] for j in ARM_JOINTS})
                self._save()
                return f"saved {words[1]} over ({x}, {y}) cm: {pose}"
            if words in (["save", "grip_open"], ["save", "grip_closed"]):
                setattr(self.grid, words[1], self.arm.known_pose()["grip"])
                self._save()
                return f"saved {words[1]} = {getattr(self.grid, words[1])}"
            if words == ["save", "clear"]:
                pose = self.arm.known_pose()
                self.grid.camera_clear = {j: pose[j] for j in ARM_JOINTS}
                self._save()
                return f"saved camera_clear = {self.grid.camera_clear}"
            if words == ["list"]:
                lines = [f"grip_open {self.grid.grip_open}, grip_closed {self.grid.grip_closed}, "
                         f"camera_clear {self.grid.camera_clear}"]
                lines += [f"({p.x_cm}, {p.y_cm}): hover {'yes' if p.hover else 'no'}, grab {'yes' if p.grab else 'no'}"
                          for p in self.grid.points]
                return "\n".join(lines)
            if words == ["finish"]:
                return self._finish()
        except (ArmError, KinematicsError, ValueError) as error:
            return f"refused: {error}"
        return HELP

    def _finish(self) -> str:
        self.grid.placeholder = False
        try:
            self.grid.check_ready()
            self.grid.check_against(self.arm.limits)
        except KinematicsError:
            self.grid.placeholder = True
            raise
        answer = self.ask(f"{len(self.grid.complete_points())} points cover every slot and the whole mess area? "
                          "Type yes to mark the grid as taught: ")
        if answer.strip().lower() != "yes":
            self.grid.placeholder = True
            return "not finished: the grid is still a placeholder"
        self._save()
        return f"finished: {self.grid_path} is marked as taught"


def make_arm(dry_run: bool, limits) -> Arm:
    return session.connect(dry_run, limits)  # a real arm is refused while locked, before the bridge opens


def main():
    parser = argparse.ArgumentParser(description="Teach the pose grid.")
    parser.add_argument("--dry-run", action="store_true", help="no arm; saves to runs/pose_grid_dryrun.json")
    args = parser.parse_args()
    grid_path = DRY_RUN_GRID if args.dry_run else kinematics.POSE_GRID_PATH
    try:
        arm = make_arm(args.dry_run, arm_module.load_limits())
    except ArmError as error:
        print(f"STOPPED: {error}")
        sys.exit(1)
    if args.dry_run and not grid_path.exists():
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(kinematics.POSE_GRID_PATH, grid_path)
    teacher = Teacher(arm, grid_path, ask=session.ask)
    print(("DRY RUN: nothing is sent to an arm. " if args.dry_run else "") + f"Saving to {grid_path}")
    print(HELP)
    sys.exit(session.guarded(arm, lambda: session.command_loop(teacher.handle, sys.stdin)))


if __name__ == "__main__":
    main()
