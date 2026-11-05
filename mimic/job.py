"""The arm program: the expected simulation's carries as a script for the arm (the demo page's download).

A downloaded mimic_job_<time>.py holds only the job (JOB: the dice, where SAM measured them in the real photo, the
target, row or stack) and calls main() here. Run it with the project's Python, from anywhere:

    ~/mimic/.venv/bin/python mimic_job_<time>.py --dry-run    the fake arm: nothing moves, every line is printed
    ~/mimic/.venv/bin/python mimic_job_<time>.py              the real arm

--dry-run replays the carries on the made-up ring the simulation used, on the fake arm (FakeTransport).

On the real arm no angle is taken from the simulation; each is worked out from the real calibration:
- a row: each die is picked at the base angle of SAM's measured position on the real ring (calib/ring.json; refused
  more than 1 cm off it) and dropped at the real slot (ring.json slots_deg);
- a stack: each die's pick base angle, the stack's base angle and the shoulder angle to let go at for each level
  come from calib/stack.json, measured by hand. Refused while it is a placeholder, saying what to measure.
Before the bridge opens: every base angle must be inside 80-120 (the range approved on 4 Nov), every carry is planned
and checked (calib/arm_limits.json, the shoulder stop, the 4.5 s grip rule), and the limits must be checked by hand.
Then: the start pose, a typed yes to enable, each carry once (no camera, so no checks and no retries), and the
normal end (back to the start pose, "hold the arm", disable). s or Ctrl+C sends S then D.
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

from mimic import arm as arm_module
from mimic import config, pickdrop, session
from mimic import ring as ring_module
from mimic.arm import ArmError
from mimic.pickdrop import PickError
from mimic.ring import RingError

STACK_PATH = config.ROOT / "calib" / "stack.json"
APPROVED = pickdrop.BASE_APPROVED     # (80, 120)


class JobError(ValueError):
    pass


# ---------- where each carry goes ----------

def made_up_ring(job: dict, real) -> ring_module.Ring:
    """The ring the simulation used (for --dry-run): ring.json's pick numbers on the job's made-up circle."""
    geometry = dict(job["made_up_ring"], centre_cm=tuple(job["made_up_ring"]["centre_cm"]))
    return dataclasses.replace(real, placeholder=False, points=[], **geometry)


def what_to_measure(moved: list[str], levels: int, bottom: str | None = None) -> str:
    where = (f"the base angle where {bottom} lies (it stays put and the pile is built on it)" if bottom
             else "the base angle to build the stack at")
    return ("Real stacking needs calib/stack.json, measured by hand on the arm (elbow taped), then placeholder set to "
            f"false: pick_base_deg, the base angle where each die to be carried sits on the ring ({', '.join(moved)}); "
            f"drop_base_deg, {where}; drop_shoulder_by_level, the shoulder angle to let go at for each of the {levels} "
            "levels (level 1 on the table, then each one die higher: a smaller angle), found by jogging with "
            "scripts/arm_jog.py.")


def load_stack(moved: list[str], levels: int, path=None, bottom: str | None = None) -> dict:
    """calib/stack.json, refused unless it has a pick angle for every die carried, the stack's base angle and a
    let-go shoulder angle for every level."""
    path = Path(path or STACK_PATH)
    needed = what_to_measure(moved, levels, bottom)
    if not path.exists():
        raise JobError(f"{path} is missing. " + needed)
    stack = json.loads(path.read_text())
    if stack.get("placeholder", True):
        raise JobError(f"{path} is still a placeholder. " + needed)
    missing = [n for n in moved if not isinstance(stack.get("pick_base_deg", {}).get(n), int)]
    shoulders = stack.get("drop_shoulder_by_level") or []
    if missing or not isinstance(stack.get("drop_base_deg"), int) or len(shoulders) < levels or not all(
            isinstance(v, int) for v in shoulders[:levels]):
        raise JobError(f"{path} is incomplete" + (f" (no pick angle for {', '.join(missing)})" if missing else "") +
                       ". " + needed)
    return stack


def carries(job: dict, ring, dry_run: bool, stack_path=None) -> list[dict]:
    """[{"name", "pick", "drop", "shoulder_drop"}], one per die carried (a pile built where its bottom die lies does
    not carry that die). Real angles come from the real calibration only."""
    order, stack = job["target"], job["stack"]
    moved = [c["name"] for c in job["carries"]]
    slot_of = {c["name"]: c["slot"] for c in job["carries"]}
    if dry_run:  # the simulation's own angles, on its made-up ring
        def drop(c):
            return round(ring.angle_for(*c["to_cm"])) if "to_cm" in c else round(ring.slots_deg[c["slot"] - 1])
        return [{"name": c["name"], "pick": round(ring.angle_for(*job["placed_cm"][c["name"]])), "drop": drop(c),
                 "shoulder_drop": None} for c in job["carries"]]
    if stack:   # level = the die's place in the pile, bottom first
        measured = load_stack(moved, len(order), stack_path, order[0] if job.get("stack_in_place") else None)
        return [{"name": n, "pick": measured["pick_base_deg"][n], "drop": measured["drop_base_deg"],
                 "shoulder_drop": measured["drop_shoulder_by_level"][order.index(n)]} for n in moved]
    ring.check_ready()
    if max(slot_of[n] for n in moved) > len(ring.slots_deg):
        raise JobError(f"calib/ring.json has {len(ring.slots_deg)} slots; this job needs {max(slot_of.values())}.")
    out = []
    for n in moved:
        try:
            pick = ring.base_angle_for(*job["seen_cm"][n])   # where SAM measured it, on the real ring
        except RingError as error:
            raise JobError(f"{n}: {error} Put it on the ring, or take a new photo.") from None
        out.append({"name": n, "pick": round(pick), "drop": round(ring.slots_deg[slot_of[n] - 1]),
                    "shoulder_drop": None})
    return out


def check_approved(planned: list[dict]) -> None:
    outside = [f"{c['name']} {what} at base {c[what]}" for c in planned for what in ("pick", "drop")
               if not APPROVED[0] <= c[what] <= APPROVED[1]]
    if outside:
        raise JobError(f"Base angles outside the approved {APPROVED[0]}-{APPROVED[1]}: {'; '.join(outside)}. "
                       "Nothing was moved.")


def plan_all(limits, start: dict, ring, planned: list[dict]) -> list:
    """Every carry planned and checked from the pose the one before ends in. Returns [(carry, steps, hold_s)]."""
    out, pose = [], dict(start)
    for c in planned:
        steps, hold_s = plan_one(limits, pose, ring, c)
        out.append((c, steps, hold_s))
        pose = dict(steps[-1].angles)
    return out


def plan_one(limits, pose: dict, ring, c: dict):
    try:
        steps, hold_s, _ = pickdrop.plan(limits, pose, base_pick=c["pick"], base_drop=c["drop"],
                                         shoulder_pick=ring.shoulder_table, grip_hold=ring.grip_hold, lift=ring.lift,
                                         approach=ring.approach, stop_at=ring.shoulder_stop, command_s=ring.command_s,
                                         shoulder_drop=c["shoulder_drop"])
    except PickError as error:
        raise JobError(f"{c['name']}: {error}") from None
    return steps, hold_s


# ---------- the script ----------

def main(job: dict, argv=None, read=input, say=print, sleep=None) -> int:
    parser = argparse.ArgumentParser(description="Mimic arm program: run the planned carries.")
    parser.add_argument("--dry-run", action="store_true", help="the fake arm: nothing moves, every line is printed")
    args = parser.parse_args(argv)
    dry_run = args.dry_run
    if dry_run:
        sleep = sleep or (lambda s: None)   # nothing moves, so nothing to wait for
    say(f"Mimic arm program from {job['run']}: {'stack' if job['stack'] else 'row'}, "
        f"{', '.join(job['target'])}. No camera: each carry runs once, as planned.")
    try:
        limits = arm_module.load_limits()
        start = arm_module.load_start_pose(say=say)
        real = ring_module.load_ring()
        the_ring = made_up_ring(job, real) if dry_run else real
        planned = carries(job, the_ring, dry_run)
        if dry_run:
            say("Dry run: the expected simulation's angles on its made-up ring, on the fake arm.")
        else:
            check_approved(planned)
        for c, steps, hold_s in plan_all(limits, start, the_ring, planned):
            say(f"  {c['name']}: pick at base {c['pick']}, drop at base {c['drop']}" +
                (f", let go at shoulder {c['shoulder_drop']}" if c["shoulder_drop"] else "") +
                f" ({len(steps)} moves, claw closed {hold_s:.2f} s of {limits.max_grip_hold_s - 0.5:.1f} s)")
        arm = session.connect(dry_run, limits, echo=say if dry_run else None)  # a locked arm is refused here
    except (JobError, RingError, ArmError) as error:
        say(f"REFUSED before opening anything: {error}")
        return 1

    def body():
        session.confirm_and_enable(arm, read, say, sleep, angles=start)
        for c in planned:
            steps, _ = plan_one(limits, arm.known_pose(), the_ring, c)   # from where the arm really is
            say(f"{c['name']}: pick at base {c['pick']}, drop at base {c['drop']}")
            pickdrop.run_steps(arm, steps, say, sleep or time.sleep)

    return session.guarded(arm, body, say, sleep)


# ---------- the download ----------

def job_from_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "simulation_summary.json").read_text())
    if not summary.get("carries"):
        raise JobError(f"{run_dir.name} has no planned carries (it was refused or is from an older version).")
    return {"run": run_dir.name, "photo": summary["photo"], "stack": bool(summary["stack"]),
            "stack_in_place": bool(summary.get("stack_in_place")), "target": summary["target"],
            "carries": summary["carries"], "seen_cm": summary["seen_cm"],
            "placed_cm": summary["placed_cm"], "made_up_ring": summary["made_up_ring"]}
