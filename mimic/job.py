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
import io
import json
import pprint
import time
import zipfile
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


def what_to_measure(order: list[str]) -> str:
    return ("Real stacking needs calib/stack.json, measured by hand on the arm (elbow taped), then placeholder set to "
            f"false: pick_base_deg, the base angle where each die sits on the ring ({', '.join(order)}); "
            "drop_base_deg, the base angle to build the stack at; drop_shoulder_by_level, the shoulder angle to let "
            f"go at for each of the {len(order)} levels (level 1 on the table, then each one die higher: a smaller "
            "angle), found by jogging with scripts/arm_jog.py.")


def load_stack(order: list[str], path=None) -> dict:
    path = Path(path or STACK_PATH)
    if not path.exists():
        raise JobError(f"{path} is missing. " + what_to_measure(order))
    stack = json.loads(path.read_text())
    if stack.get("placeholder", True):
        raise JobError(f"{path} is still a placeholder. " + what_to_measure(order))
    missing = [n for n in order if not isinstance(stack.get("pick_base_deg", {}).get(n), int)]
    levels = stack.get("drop_shoulder_by_level") or []
    if missing or not isinstance(stack.get("drop_base_deg"), int) or len(levels) < len(order) or not all(
            isinstance(v, int) for v in levels[:len(order)]):
        raise JobError(f"{path} is incomplete" + (f" (no pick angle for {', '.join(missing)})" if missing else "") +
                       ". " + what_to_measure(order))
    return stack


def carries(job: dict, ring, dry_run: bool, stack_path=None) -> list[dict]:
    """[{"name", "pick", "drop", "shoulder_drop"}] in order. Real angles come from the real calibration only."""
    order, stack = job["target"], job["stack"]
    slot_of = {c["name"]: c["slot"] for c in job["carries"]}
    if dry_run:  # the simulation's own angles, on its made-up ring
        return [{"name": n, "pick": round(ring.angle_for(*job["placed_cm"][n])),
                 "drop": round(ring.slots_deg[slot_of[n] - 1]), "shoulder_drop": None} for n in order]
    if stack:
        measured = load_stack(order, stack_path)
        return [{"name": n, "pick": measured["pick_base_deg"][n], "drop": measured["drop_base_deg"],
                 "shoulder_drop": measured["drop_shoulder_by_level"][level]} for level, n in enumerate(order)]
    ring.check_ready()
    if max(slot_of[n] for n in order) > len(ring.slots_deg):
        raise JobError(f"calib/ring.json has {len(ring.slots_deg)} slots; this job needs {max(slot_of.values())}.")
    out = []
    for n in order:
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

SCRIPT = '''"""Mimic arm program {stamp}, from the expected simulation {folder}.

    ~/mimic/.venv/bin/python {script} --dry-run    the fake arm: nothing moves, every line is printed
    ~/mimic/.venv/bin/python {script}              the real arm (asks for a typed yes first)

The work is done by mimic/job.py, with the project's arm.py, serial bridge, limits and start pose. On the real arm
every angle is worked out again from calib/ring.json (a row) or calib/stack.json (a stack), never taken from the
simulation, and anything outside base 80-120 is refused. No camera: each carry runs once. s or Ctrl+C: S then D.
"""
import sys

from mimic.job import main

JOB = {job}

if __name__ == "__main__":
    sys.exit(main(JOB))
'''

README = """# Mimic arm program {stamp}

Made by the demo page from the expected simulation `{folder}` (photo: `{photo}`). Target: {kind} {target}.

- `{script}`: the program. Run it with the project's Python: `~/mimic/.venv/bin/python {script} --dry-run` first
  (the fake arm: nothing moves, every command is printed), then without `--dry-run` for the real arm.
- `moves.json`: the expected simulation's move list, every line the simulated arm was sent. It was planned on a
  MADE-UP ring, so the real arm's angles will differ: the program works them out again.

What the program does on the real arm, and what it refuses:
1. A row: each die is picked where SAM measured it, on the real ring (`calib/ring.json`, which must be calibrated, with
   `slots_deg` set), and dropped in its real slot. A die more than 1 cm off the ring is refused.
2. A stack: the pick angles, the stack's base angle and the shoulder angle to let go at for each level come from
   `calib/stack.json`, measured by hand. While it is a placeholder the program refuses and says what to measure.
3. Every base angle must be inside 80-120; every carry is planned and checked against `calib/arm_limits.json`, the
   shoulder stop and the 4.5 s grip rule before the serial bridge is opened.
4. The start pose, a typed yes to enable, each carry once (no camera: no checks, no retries), then back to the start
   pose, "hold the arm", and the servos off. `s` or Ctrl+C sends S then D.

The dice must be where the photo showed them when the program runs. For checking and correcting each placement with
the camera, use `python scripts/run_mimic.py --ring` instead.
"""


def job_from_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "simulation_summary.json").read_text())
    if not summary.get("carries"):
        raise JobError(f"{run_dir.name} has no planned carries (it was refused or is from an older version).")
    return {"run": run_dir.name, "photo": summary["photo"], "stack": bool(summary["stack"]),
            "target": summary["target"], "carries": summary["carries"], "seen_cm": summary["seen_cm"],
            "placed_cm": summary["placed_cm"], "made_up_ring": summary["made_up_ring"]}


def program_zip(run_dir) -> tuple[str, bytes]:
    """(the zip's file name, its bytes): mimic_job_<time>.py, moves.json and README.md."""
    run_dir = Path(run_dir)
    job = job_from_run(run_dir)
    stamp = run_dir.name.removesuffix("_simulation")
    script = f"mimic_job_{stamp}.py"
    fields = dict(stamp=stamp, folder=run_dir.name, script=script, photo=job["photo"],
                  kind="a stack (bottom first):" if job["stack"] else "a row (slot 1 first):",
                  target=", ".join(job["target"]))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(script, SCRIPT.format(job=pprint.pformat(job, width=110, sort_dicts=False), **fields))
        z.writestr("moves.json", (run_dir / "simulation_moves.json").read_bytes())
        z.writestr("README.md", README.format(**fields))
    return f"mimic_arm_program_{stamp}.zip", buffer.getvalue()
