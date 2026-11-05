"""The arm program the demo page offers: a standalone Python script and an ESP32 Arduino sketch, made from the plan.

Both files are filled in by code from templates (string.Template), never written by AI. The plan is the carries the
simulation planned (mimic/job.py), turned into joint angles step by step with mimic/pickdrop.py, the same planner
and checks every other script uses.

The angles come from the real calibration when calib/ring.json (and, for a stack, calib/stack.json) is measured.
While either is a placeholder, the files still hold a full plan, with the simulation's angles, and say so at the top:
the Python script then only does --dry-run, and the sketch starts with #error so it cannot be uploaded by mistake.
"""
import io
import json
import zipfile
from pathlib import Path
from string import Template

from mimic import arm as arm_module
from mimic import job as job_module
from mimic import pickdrop, session
from mimic import ring as ring_module
from mimic.arm import JOINTS

SIM_NOTE = ("SIMULATION ANGLES: {reasons}, so these angles come from the expected simulation on a made-up ring. "
            "Calibrate first, then make this program again.")


def _describe(step, carry: dict, target: list, stack: bool, slot: int, after_close: bool) -> str:
    name = carry["name"]
    below = target[target.index(name) - 1] if stack and target.index(name) > 0 else None
    dest = f"the {below}" if below else f"slot {slot}"
    label = step.label
    if label == "open, base to pick":
        return f"open the claw and turn to the {name}"
    if label == "shoulder above the pick":
        return f"shoulder above the {name}"
    if label.startswith("lower to"):
        return f"lower onto {dest}" if after_close else f"lower onto the {name}"
    return {"close": f"close the claw on the {name}", "lift": f"lift the {name}", "swing to drop": f"swing to {dest}",
            "open": f"let go of the {name}", "lift away": "lift away"}.get(label, label)


def plan_rows(run_dir) -> dict:
    """The plan for one simulation folder: the steps as rows, the constants, and whether the angles are real."""
    job = job_module.job_from_run(run_dir)
    limits = arm_module.load_limits()
    start = arm_module.load_start_pose(say=None)
    real = ring_module.load_ring()
    reasons = []
    if real.placeholder:
        reasons.append("calib/ring.json is still a placeholder (the real ring is not calibrated)")
    if job["stack"]:
        stack_path = Path(job_module.STACK_PATH)
        stack = json.loads(stack_path.read_text()) if stack_path.exists() else {"placeholder": True}
        if stack.get("placeholder", True):
            reasons.append("calib/stack.json is still a placeholder (the stack levels are not measured)")
    simulation = bool(reasons)
    the_ring = job_module.made_up_ring(job, real) if simulation else real
    planned = job_module.carries(job, the_ring, dry_run=simulation)
    slot_of = {c["name"]: c["slot"] for c in job["carries"]}
    rows, pose = [], dict(start)
    for carry, steps, _ in job_module.plan_all(limits, start, the_ring, planned):
        closed = False
        for step in steps:
            closed = closed or step.label == "close"
            rows.append({"base": step.angles["base"], "shoulder": step.angles["shoulder"], "grip": step.angles["grip"],
                         "ms": step.ms, "wait": step.wait_s,
                         "what": f"{carry['name']}: " + _describe(step, carry, job["target"], job["stack"],
                                                                 slot_of[carry["name"]], closed and step.label != "close")})
            pose = dict(step.angles)
    back = max(session.START_RAMP_MIN_MS,
               max(abs(start[j] - pose[j]) for j in JOINTS) * session.START_RAMP_MS_PER_DEGREE)
    rows.append({"base": start["base"], "shoulder": start["shoulder"], "grip": start["grip"], "ms": back, "wait": 0.0,
                 "what": "back to the start pose"})
    task = (("stack " if job["stack"] else "line up ") + ", ".join(job["target"]) +
            (" (bottom first)" if job["stack"] else " (slot 1 first)"))
    return {"job": job, "limits": limits, "start": start, "ring": the_ring, "rows": rows, "simulation": simulation,
            "note": SIM_NOTE.format(reasons=" and ".join(reasons)) if simulation else "", "task": task,
            "stamp": Path(run_dir).name.removesuffix("_simulation")}


# ---------- the standalone Python script ----------

PYTHON = Template('''#!/usr/bin/env python3
"""Mimic arm program $stamp: $task.
$note_block
Run it on the computer the arm's USB cable is plugged into (on Windows the arm shows up as a COM port). It needs only
Python 3 and pyserial (pip install pyserial), and only for the real arm.

    python $script --dry-run              print every step; nothing is sent to the arm
    python $script --port COM7            the real arm: asks for a typed yes, then runs every step once

Before the port is opened, every angle is checked against the arm's limits, every move against the speed limit,
every stretch with the claw closed against $max_closed_s s, and every base angle against $approved_lo-$approved_hi.
Press s (on Windows) or Ctrl+C at any time: the arm is sent S (stop), then D (servos off).
There is no camera here: each carry runs once, exactly as planned. Made from the plan by code, not by AI.
"""
import argparse
import sys
import time

SIMULATION_ANGLES = $simulation    # True: the simulation's angles, so this program only does --dry-run

# ---- the arm, copied from calib/arm_limits.json and calib/start_pose.json when this was made ----
LIMITS = $limits
BASE_APPROVED = ($approved_lo, $approved_hi)    # base angles approved for this arm
MIN_MS_PER_DEGREE = $min_ms                  # speed limit: at least this many ms per degree moved
MAX_CLAW_CLOSED_S = $max_closed_s               # the firmware opens a claw held closed for $hold_s s; stay under it
START = $start
ELBOW = $elbow                               # the elbow is fixed in place (its servo failed): always sent as this
GRIP_OPEN, GRIP_HOLD = $grip_open, $grip_hold
SETTLE_S, COMMAND_S = $settle_s, $command_s           # the wait after each move, and the time per command on the link
BAUD = 115200

# ---- the moves, one line per step ----
STEPS = [
    # base, shoulder, grip (degrees), move time (ms), extra wait (s)
$py_rows
]


def check_plan():
    """Everything checked before the port is opened. Returns the problems found (none: all fine)."""
    problems, pose, closed_s = [], dict(START), 0.0
    for n, (base, shoulder, grip, ms, wait) in enumerate(STEPS, start=1):
        for joint, angle in (("base", base), ("shoulder", shoulder), ("elbow", ELBOW), ("grip", grip)):
            lowest, highest = LIMITS[joint]
            if not lowest <= angle <= highest:
                problems.append(f"step {n}: {joint} {angle} is outside {lowest}-{highest}")
        biggest = max(abs(base - pose["base"]), abs(shoulder - pose["shoulder"]), abs(grip - pose["grip"]))
        if ms < biggest * MIN_MS_PER_DEGREE:
            problems.append(f"step {n}: {biggest} degrees in {ms} ms breaks the speed limit")
        if grip == GRIP_HOLD:                      # the claw is closed during this step
            closed_s += ms / 1000 + SETTLE_S + wait + COMMAND_S
            if closed_s + COMMAND_S > MAX_CLAW_CLOSED_S:
                problems.append(f"step {n}: the claw would be closed for {closed_s + COMMAND_S:.2f} s")
        else:
            closed_s = 0.0
        pose = {"base": base, "shoulder": shoulder, "grip": grip}
    return problems


class Arm:
    """The arm's serial protocol: one command line out, one reply line back ("OK", "A b s e g", "ERR ...")."""

    def __init__(self, port_name):
        import serial                              # pyserial: needed only for the real arm
        self.port = serial.Serial()
        self.port.port, self.port.baudrate, self.port.timeout = port_name, BAUD, 3
        self.port.dtr = False                      # low before opening, so the ESP32 is not reset
        self.port.rts = False
        self.port.open()
        time.sleep(2.0)                            # if it reset anyway, let it boot
        self.port.reset_input_buffer()

    def send(self, line):
        self.port.write((line + "\\n").encode())
        while True:
            reply = self.port.readline().decode(errors="replace").strip()
            if not reply:
                raise RuntimeError(f"no reply to {line!r}")
            if reply.startswith(("OK", "ERR", "A ")):   # skip anything else the board prints
                return reply

    def ok(self, line):
        reply = self.send(line)
        if reply != "OK":
            raise RuntimeError(f"the arm replied {reply!r} to {line!r}")

    def stop_and_disable(self):
        for line in ("S", "D"):
            try:
                self.send(line)
            except Exception:
                pass


def stop_pressed():
    try:
        import msvcrt                              # Windows only: press s to stop
    except ImportError:
        return False
    while msvcrt.kbhit():
        if msvcrt.getwch().lower() == "s":
            return True
    return False


def command(base, shoulder, grip, ms):
    return f"J {base} {shoulder} {ELBOW} {grip} {ms}"


def run(arm):
    status = arm.send("?")
    if status.endswith(" L"):
        raise RuntimeError("the arm's firmware is locked: it drives no servo")
    print("Enabling switches the servos on one at a time and each jumps to the start pose:", START)
    answer = input("Pose the arm like the start pose (upper arm vertical, forearm horizontal, claw open), area clear, "
                   "a hand on the switch. Type yes to enable: ")
    if answer.strip().lower() != "yes":
        print("Not enabled: nothing was moved.")
        return 1
    arm.ok("E " + " ".join(str(START[j]) for j in ("base", "shoulder", "elbow", "grip")))
    time.sleep(1.3)                                # the servos switch on 0.4 s apart
    for n, (base, shoulder, grip, ms, wait) in enumerate(STEPS, start=1):
        if stop_pressed():
            raise KeyboardInterrupt
        print(f"step {n} of {len(STEPS)}: {command(base, shoulder, grip, ms)}")
        arm.ok(command(base, shoulder, grip, ms))
        time.sleep(ms / 1000 + SETTLE_S + wait)
    print("Done. Hold the arm: it goes limp in 2 s.")
    time.sleep(2.0)
    arm.ok("D")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mimic arm program $stamp")
    parser.add_argument("--dry-run", action="store_true", help="print every step; nothing is sent")
    parser.add_argument("--port", help="the arm's serial port, for example COM7")
    args = parser.parse_args(argv)
    problems = check_plan()
    if problems:
        print("REFUSED before opening the port:\\n  " + "\\n  ".join(problems))
        return 1
    if args.dry_run:
        for n, (base, shoulder, grip, ms, wait) in enumerate(STEPS, start=1):
            print(f"step {n:>2}: {command(base, shoulder, grip, ms)}" + (f"  then wait {wait} s" if wait else ""))
        print(f"Dry run: {len(STEPS)} steps checked, nothing was sent." +
              (" (Simulation angles: the real arm is refused.)" if SIMULATION_ANGLES else ""))
        return 0
    if SIMULATION_ANGLES:
        print("REFUSED: these are the simulation's angles, not the real arm's. Calibrate the ring (and measure "
              "calib/stack.json for a stack), then make this program again. --dry-run still works.")
        return 1
    outside = [n for n, row in enumerate(STEPS, start=1) if not BASE_APPROVED[0] <= row[0] <= BASE_APPROVED[1]]
    if outside:
        print(f"REFUSED: steps {outside} turn the base outside the approved {BASE_APPROVED[0]}-{BASE_APPROVED[1]}.")
        return 1
    if not args.port:
        print("Give the arm's serial port, for example --port COM7.")
        return 1
    arm = Arm(args.port)
    try:
        return run(arm)
    except KeyboardInterrupt:
        arm.stop_and_disable()
        print("\\nStopped: S and D sent, the servos are off.")
        return 130
    except Exception as error:
        print(f"\\nSTOPPED: {error}")
        arm.stop_and_disable()
        print("S and D sent, the servos are off.")
        return 1
    finally:
        arm.port.close()


if __name__ == "__main__":
    sys.exit(main())
''')


# ---------- the ESP32 Arduino sketch ----------

SKETCH = Template('''// Mimic arm program $stamp: $task.
$ino_note// Made from the plan by code (a template filled in), not by AI. For the KEYESTUDIO ESP32 arm, with the ESP32Servo
// library. Upload it, open the serial monitor at 115200 baud, pose the arm like the start pose, then type go and
// press Enter: nothing moves before that. No camera: each carry runs once, exactly as planned.
$ino_error
#include <ESP32Servo.h>

// ---- the arm: pins, limits and start pose (calib/arm_limits.json, calib/start_pose.json) ----
const int PINS[4] = {$pins};           // base, shoulder, elbow, grip
const int LOWEST[4] = {$lowest};
const int HIGHEST[4] = {$highest};
const int START_POSE[4] = {$start_list};
const int ELBOW = $elbow;                         // the elbow is fixed in place: always this
const int GRIP_OPEN = $grip_open;                     // this claw closes with a lower angle
const unsigned long MAX_GRIP_HOLD_MS = $hold_ms;    // a claw closed this long is opened again
const unsigned long ATTACH_GAP_MS = 400;        // the servos switch on one at a time, this far apart
const unsigned long SETTLE_MS = $settle_ms;

// ---- the moves: base, shoulder, grip (degrees), move time (ms), extra wait (ms) ----
struct Step { int base; int shoulder; int grip; unsigned int ms; unsigned int waitMs; };
const Step STEPS[] = {
$ino_rows
};
const int STEP_COUNT = sizeof(STEPS) / sizeof(STEPS[0]);

Servo servos[4];
int current[4];
bool gripClosed = false;
unsigned long gripClosedSince = 0;

void setJoint(int joint, int angle) {
  current[joint] = constrain(angle, LOWEST[joint], HIGHEST[joint]);
  servos[joint].write(current[joint]);
}

void watchGrip() {                              // the 5 s grip release
  bool closed = current[3] < GRIP_OPEN - 2;
  if (closed && !gripClosed) { gripClosed = true; gripClosedSince = millis(); }
  if (!closed) gripClosed = false;
  if (gripClosed && millis() - gripClosedSince > MAX_GRIP_HOLD_MS) {
    setJoint(3, GRIP_OPEN);
    gripClosed = false;
    Serial.println("The claw was closed for 5 s: opened it");
  }
}

void waitMs(unsigned long ms) {
  unsigned long start = millis();
  while (millis() - start < ms) { watchGrip(); delay(5); }
}

void moveTo(int base, int shoulder, int grip, unsigned int ms) {   // every joint in straight 20 ms steps
  int from[4] = {current[0], current[1], current[2], current[3]};
  int to[4] = {base, shoulder, ELBOW, grip};
  int ticks = max(1, (int)(ms / 20));
  for (int t = 1; t <= ticks; t++) {
    for (int j = 0; j < 4; j++) setJoint(j, from[j] + (to[j] - from[j]) * t / ticks);
    waitMs(20);
  }
}

void waitForGo() {
  String line = "";
  while (true) {
    while (Serial.available()) {
      char c = Serial.read();
      if (c == '\\n' || c == '\\r') {
        line.trim();
        if (line == "go") return;
        line = "";
      } else {
        line += c;
      }
    }
    delay(10);
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("Mimic arm program $stamp. Pose the arm like the start pose, then type go and press Enter.");
  waitForGo();
  for (int j = 0; j < 4; j++) {                 // staggered start: one servo every 0.4 s, at the start pose
    servos[j].attach(PINS[j], 500, 2400);
    setJoint(j, START_POSE[j]);
    delay(ATTACH_GAP_MS);
  }
  for (int i = 0; i < STEP_COUNT; i++) {
    Serial.printf("step %d of %d\\n", i + 1, STEP_COUNT);
    moveTo(STEPS[i].base, STEPS[i].shoulder, STEPS[i].grip, STEPS[i].ms);
    waitMs(SETTLE_MS + STEPS[i].waitMs);
  }
  Serial.println("Done. Hold the arm: the servos switch off in 2 s.");
  waitMs(2000);
  for (int j = 0; j < 4; j++) servos[j].detach();
}

void loop() {}
''')

README = Template('''# Mimic arm program $stamp

Task: $task. Made by Mimic's demo page from the expected simulation `$folder`, by code (templates filled in from the
plan), not by AI.
$readme_note
| File | What it is |
|---|---|
| `$script` | A standalone Python program: the arm's limits and start pose, every step as a table, a small serial driver and the safety checks. Needs only pyserial. |
| `$sketch_dir/$sketch` | The same plan as an ESP32 Arduino sketch (ESP32Servo), for the board's own pins. Arduino needs it in a folder of the same name, as here. |
| `moves.json` | The expected simulation's move list: every line the simulated arm was sent. |

## Running the Python program

On the computer the arm is plugged into (Windows: a COM port), with `pip install pyserial`:

    python $script --dry-run         every step printed and checked; nothing is sent
    python $script --port COM7       the real arm: a typed yes, then each step once; s or Ctrl+C sends S then D

Before the port opens it checks every angle against the limits, every move against the speed limit, every stretch
with the claw closed against $max_closed_s s, and every base angle against $approved_lo-$approved_hi.

## Uploading the sketch

Open `$sketch_dir/$sketch` in the Arduino IDE (ESP32 board package, ESP32Servo library), upload, open the serial
monitor at 115200 baud, pose the arm like the start pose, type `go` and press Enter. The sketch replaces the Mimic
firmware on the board until the firmware is uploaded again.

No camera is used by either file: each carry runs once, exactly as planned.
''')


def program_files(run_dir) -> dict[str, bytes]:
    """The zip's files: {path in the zip: bytes}."""
    run_dir = Path(run_dir)
    plan = plan_rows(run_dir)
    limits, start, ring, rows = plan["limits"], plan["start"], plan["ring"], plan["rows"]
    stamp = plan["stamp"]
    script, sketch_dir = f"mimic_job_{stamp}.py", f"mimic_arm_{stamp}"
    sketch = f"{sketch_dir}.ino"
    ranges = {j: limits.range_of(j) for j in JOINTS}
    max_closed = limits.max_grip_hold_s - pickdrop.HOLD_MARGIN_S
    width = max(len(r["what"]) for r in rows)
    py_rows = "\n".join(f"    ({r['base']:>3}, {r['shoulder']:>3}, {r['grip']:>3}, {r['ms']:>5}, {r['wait']:>3}),"
                        f"  # {n:>2}. {r['what']}" for n, r in enumerate(rows, start=1))
    ino_rows = "\n".join(f"  {{{r['base']:>3}, {r['shoulder']:>3}, {r['grip']:>3}, {r['ms']:>5}, "
                         f"{round(r['wait'] * 1000):>4}}},  // {n:>2}. {r['what']}" for n, r in enumerate(rows, start=1))
    note = plan["note"]
    fields = dict(stamp=stamp, task=plan["task"], script=script, sketch=sketch, sketch_dir=sketch_dir,
                  folder=run_dir.name, simulation=plan["simulation"],
                  note_block=f"\n{note}\n" if note else "",
                  ino_note="".join(f"// {line}\n" for line in _wrap(note)) if note else "",
                  ino_error=('#error "Simulation angles: calibrate the ring before uploading"\n' if plan["simulation"]
                             else ""),
                  readme_note=f"\n**{note}** The Python program only does `--dry-run` until then, and the sketch "
                              "stops the upload with an #error line.\n" if note else "",
                  limits="{" + ", ".join(f'"{j}": {tuple(ranges[j])}' for j in JOINTS) + "}",
                  approved_lo=pickdrop.BASE_APPROVED[0], approved_hi=pickdrop.BASE_APPROVED[1],
                  min_ms=limits.min_ms_per_degree, max_closed_s=max_closed, hold_s=limits.max_grip_hold_s,
                  hold_ms=round(limits.max_grip_hold_s * 1000),
                  start="{" + ", ".join(f'"{j}": {start[j]}' for j in JOINTS) + "}",
                  start_list=", ".join(str(start[j]) for j in JOINTS), elbow=limits.home_deg["elbow"],
                  grip_open=limits.grip_open_deg, grip_hold=ring.grip_hold, settle_s=pickdrop.SETTLE_S,
                  settle_ms=round(pickdrop.SETTLE_S * 1000), command_s=ring.command_s,
                  pins=", ".join(str(limits.pins[j]) for j in JOINTS),
                  lowest=", ".join(str(ranges[j][0]) for j in JOINTS),
                  highest=", ".join(str(ranges[j][1]) for j in JOINTS), py_rows=py_rows, ino_rows=ino_rows)
    del width
    return {script: PYTHON.substitute(fields).encode(), f"{sketch_dir}/{sketch}": SKETCH.substitute(fields).encode(),
            "moves.json": (run_dir / "simulation_moves.json").read_bytes(),
            "README.md": README.substitute(fields).encode()}


def _wrap(text: str, width: int = 110) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + [line] if line else lines


def program_zip(run_dir) -> tuple[str, bytes]:
    files = program_files(run_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    stamp = Path(run_dir).name.removesuffix("_simulation")
    return f"mimic_arm_program_{stamp}.zip", buffer.getvalue()
