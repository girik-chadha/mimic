"""Simulation mode (run_mimic.py --ring --sim): real perception and the real planner, a simulated arm.

What is REAL: one photo (--photo). SAM finds its 6 dots and the photo calibrates itself; Spark names the dice and
SAM checks each name; SAM measures where each die is. The $0.30 cost cap. The planner and every check of ring mode
(mimic/ring_loop.py: dice spacing, slots kept clear, every carry inside the 4.5 s grip rule, the limits in
calib/arm_limits.json, the camera-check-and-correct loop) and the pick numbers in calib/ring.json.

What is MADE UP: the ring and its slots (MADE_UP_RING; the real ring is not calibrated yet), and the arm. Each die is
placed on the made-up ring in its own direction from the ring's centre ("placed on the made-up ring"). The
simulated arm (SimArm) answers every command like the firmware and keeps a clock instead of waiting. When the claw
closes at table height it picks up the die under it, and when it opens there it puts that die down. The camera
check after each carry reads where the simulation put the dice, not a photo. --sim-miss-deg N puts the first die
down N degrees off, once, so the retry shows (an INJECTED TEST MISS, written on the video).

The target: --instruction "row black, blue, red" (each colour must match exactly one of Spark's names), or the
photo order: the dice left to right on the sheet go to slots 1, 2, 3.

Outputs, in runs/<time>_simulation/, every name starting with "simulation_":
  simulation_moves.json      every line sent to the simulated arm, in order: what the real arm would receive if
                             every die landed where the simulation put it (in a real run each pick follows the
                             camera). Printed too. Each line has its step, and "simulated_claw" where the simulated
                             claw picked up or put down a die (or an INJECTED TEST MISS).
  simulation_summary.json, simulation_attempts.jsonl   like a real run's, with "simulation": true
  simulation_photo.jpg, simulation_photo_dots.jpg      the real photo and its labelled dots
  simulation_video.mp4       the top-down video (mimic/sim_video.py)
"""
import dataclasses
import json
import math
import shutil
import time
from pathlib import Path

from mimic import arm as arm_module
from mimic import brain, config, marks, meta_api, pickdrop, plan, ring_loop, scene, session, sim_video
from mimic import ring as ring_module
from mimic.arm import Arm, FakeTransport

# The made-up ring: centre near the arm's base at the top of the sheet, radius chosen so it passes close to dice
# lying across the middle of the sheet. Base 90 points straight down the sheet (theta = phi - 180). Slots on the
# left of the ring, 16 degrees (5.6 cm) apart and at least 4 cm from every dot. Pick numbers come from ring.json.
MADE_UP_RING = {"centre_cm": (18.5, 27.0), "radius_cm": 20.0, "scale": 1.0, "offset_deg": -180.0,
                "slots_deg": [22, 38, 54]}
PICK_RADIUS_CM = 1.0          # the claw picks up a die whose centre is this close to the claw tip
INSTRUCTION_MODES = ("row", "stack")
DIE_SUFFIX = " die"           # the naming prompt asks for "colour die"; anything else is not a die
LABEL = "placed on the made-up ring"
MARK = "SIMULATION: an output of Mimic's simulation mode (run_mimic.py --ring --sim); the arm was simulated."


class SimError(ValueError):
    pass


# ---------- the made-up world ----------

def made_up_ring(real: ring_module.Ring) -> ring_module.Ring:
    """calib/ring.json's pick numbers (table angle, grip, lift, approach, bridge time, camera_clear) on the
    made-up circle and slots."""
    return dataclasses.replace(real, placeholder=False, points=[], source="made up for the simulation",
                               note="MADE UP for the simulation (mimic/sim.py MADE_UP_RING): not a measured ring.",
                               **MADE_UP_RING)


def place_on_ring(positions: dict, ring) -> dict:
    """Each die moved onto the ring along the line from the ring's centre (its direction is kept)."""
    placed = {}
    for name, (x, y) in positions.items():
        cx, cy = ring.centre_cm
        d = math.dist((x, y), (cx, cy))
        if d == 0:
            raise SimError(f"{name} is at the ring's centre: it has no direction.")
        placed[name] = (cx + (x - cx) / d * ring.radius_cm, cy + (y - cy) / d * ring.radius_cm)
    return placed


class SimArm(FakeTransport):
    """The simulated arm: answers like the firmware (FakeTransport), keeps a clock instead of waiting, records every
    line, and carries the dice. Starts disarmed, like the board."""

    def __init__(self, limits, ring, dice: dict, say=print, miss: tuple | None = None):
        super().__init__(limits)
        self.ring, self.limits, self.out = ring, limits, say
        self.dice = dict(dice)        # name -> (x, y) cm; None while held
        self.held = None
        self.clock = 0.0              # simulated seconds
        self.step = ""                # the step being run, from the last line printed
        self.carrying = ""            # "red die -> slot 1, try 1"
        self.try_of = None            # (name, attempt) of the carry being run
        self.miss = miss              # (name, degrees): that die's first put-down lands this far off
        self.injected = None          # {"name", "degrees", "t_s", "attempt"} once the miss has happened
        self.log = []                 # one entry per line sent

    def sleep(self, seconds: float) -> None:
        self.clock += seconds

    def say(self, text: str) -> None:
        """Print, and remember the step a pick-and-drop line names ("lower to 172: base 72, ...")."""
        self.out(text)
        self.step = text.strip().split(": base ")[0]

    def on_try(self, move, attempt: int) -> None:
        self.carrying, self.try_of = f"{move.name} -> slot {move.slot}, try {attempt}", (move.name, attempt)

    def positions(self, names: list) -> dict:
        return {n: self.dice[n] for n in names if self.dice.get(n) is not None}

    def send(self, line: str) -> str:
        before = list(self.angles)
        reply = super().send(line)
        event = self._claw(before) if line.startswith("J") and reply == "OK" else None
        self.log.append({"t_s": round(self.clock, 3), "send": line, "reply": reply,
                         "step": self.step if line.startswith("J") else "", "carrying": self.carrying,
                         "before": before, "after": list(self.angles), "dice": dict(self.dice), "held": self.held,
                         "event": event})
        return reply

    def _claw(self, before: list) -> str | None:
        """A J at table height that closes the claw picks up the die under it; one that opens it puts it down."""
        base, shoulder, _, grip = self.angles
        if shoulder < self.ring.shoulder_table:
            return None
        open_deg, hold = self.limits.grip_open_deg, self.ring.grip_hold
        tip = self.ring.point_at(base)
        if before[3] == open_deg and grip == hold and self.held is None:
            under = [n for n, p in self.dice.items() if p is not None and math.dist(p, tip) <= PICK_RADIUS_CM]
            if not under:
                return "closed on nothing"
            self.held = min(under, key=lambda n: math.dist(self.dice[n], tip))
            self.dice[self.held] = None
            return f"picked up {self.held}"
        if before[3] == hold and grip == open_deg and self.held is not None:
            name, offset, event = self.held, 0.0, f"put down {self.held}"
            if self.miss and self.miss[0] == name:
                offset, self.miss = self.miss[1], None
                attempt = self.try_of[1] if self.try_of and self.try_of[0] == name else None
                self.injected = {"name": name, "degrees": offset, "t_s": round(self.clock, 3), "attempt": attempt}
                event = f"INJECTED TEST MISS: {name} put down {offset:+g} degrees from the claw"
                self.out(f"  {event} (simulation only)")
            self.dice[name], self.held = self.ring.point_at(base + offset), None
            return event
        return None


# ---------- the target ----------

def parse_instruction(text: str) -> tuple[str, list[str]]:
    """'row black, blue, red' -> ("row", ["black", "blue", "red"])."""
    parts = text.strip().lower().split(None, 1)
    if len(parts) < 2 or parts[0] not in INSTRUCTION_MODES:
        raise SimError(f'An instruction is "row" or "stack" and then colours separated by commas, for example '
                       f'"row black, blue, red". Got {text!r}.')
    colours = [c.strip() for c in parts[1].split(",") if c.strip()]
    return parts[0], colours


def match_colours(colours: list[str], names: list[str]) -> list[str]:
    """Each colour (one or more words) must be in exactly one of Spark's names: 'blue' -> 'blue die'. A die's
    whole name always picks that die ('blue die' when there is also a 'dark blue die')."""
    order = []
    for colour in colours:
        matches = [n for n in names if n == colour] or [
            n for n in names if all(word in n.split() for word in colour.split())]
        if len(matches) != 1:
            hint = " Separate the colours with commas." if len(colour.split()) > 1 and not matches else ""
            raise SimError(f"{colour!r} matches {len(matches)} of the dice Spark named ({', '.join(names)}); it "
                           f"must match exactly one.{hint}")
        if matches[0] in order:
            raise SimError(f"{matches[0]} is named twice in the instruction.")
        order.append(matches[0])
    return order


def photo_order(positions: dict) -> list[str]:
    """Left to right on the sheet (smallest x first)."""
    return sorted(positions, key=lambda n: positions[n][0])


# ---------- the run ----------

def run(photo, instruction: str | None = None, miss_deg: float = 0.0, out_root=config.RUNS_DIR, say=print,
        make_video=None) -> dict:
    """The whole simulated run. Everything that can be refused is refused before any API call or simulated move."""
    make_video = make_video or sim_video.render
    photo = Path(photo)
    if not photo.exists():
        raise SimError(f"No photo at {photo}.")
    layout = scene.load_layout()
    limits = arm_module.load_limits()
    start = arm_module.load_start_pose(say=say)
    the_ring = made_up_ring(ring_module.load_ring())
    ring_loop.preflight(layout, the_ring, limits)
    mode, colours = parse_instruction(instruction) if instruction else ("row", None)
    if mode == "stack":
        raise SimError("Stacking is not built yet (it comes in the next commit).")

    out_dir = Path(out_root) / f"{time.strftime('%Y%m%d-%H%M%S')}_simulation"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    ring_note = {k: list(v) if isinstance(v, tuple) else v for k, v in MADE_UP_RING.items()}
    summary = {"simulation": True, "mode": "ring, simulated", "photo": str(photo), "instruction": instruction,
               "target": None, "made_up_ring": ring_note, "injected_miss": None, "success": False, "moves": [],
               "attempts": 0, "max_error_cm": None, "cost_usd": 0.0, "seconds": 0.0, "sim_seconds": 0.0,
               "error": None, "out_dir": str(out_dir)}
    photo_copy = out_dir / f"simulation_photo{photo.suffix.lower()}"
    shutil.copyfile(photo, photo_copy)
    try:
        # Perception: real calls on the real photo, under the run's cost cap.
        meta_api.reset()
        budget_before = config.BUDGET_USD
        config.BUDGET_USD = min(config.BUDGET_USD, ring_loop.RUN_BUDGET_USD)
        try:
            say(f"Perception on {photo} (real SAM and Spark calls, cap ${ring_loop.RUN_BUDGET_USD:.2f})")
            calib = scene.calibrate(str(photo), layout, out_dir / "simulation_photo_dots.jpg")
            names = brain.name_objects(str(photo))
            dice = [n for n in names if n.endswith(DIE_SUFFIX)]
            if len(dice) < len(names):
                say(f"  not dice, left out: {', '.join(n for n in names if n not in dice)}")
            seen = scene.read_positions(str(photo), dice, calib, layout)
        finally:
            config.BUDGET_USD = budget_before
            summary["cost_usd"] = round(meta_api.cost_so_far(), 4)
        target = match_colours(colours, dice) if colours else photo_order(seen)
        summary["target"] = target
        say(f"Target ({'typed' if colours else 'photo order, left to right'}), slot 1 first: {', '.join(target)}")

        # Planning: the dice placed on the made-up ring, then ring mode's own checks, before any move.
        placed = place_on_ring(seen, the_ring)
        for name in dice:
            say(f"  {name}: photo ({seen[name][0]:.1f}, {seen[name][1]:.1f}) cm -> {LABEL} at base "
                f"{the_ring.angle_for(*placed[name]):.0f}, ({placed[name][0]:.1f}, {placed[name][1]:.1f}) cm, "
                f"moved {math.dist(seen[name], placed[name]):.1f} cm")
        slots_cm = [the_ring.point_at(a) for a in the_ring.slots_deg]
        moves = plan.plan_moves(placed, target, slots_cm)
        ring_loop.check_mess(placed, moves, the_ring, limits)

        # Execution on the simulated arm, exactly as run_mimic.py --ring drives the real one.
        miss = (moves[0].name, miss_deg) if miss_deg else None
        world = SimArm(limits, the_ring, placed, say=say, miss=miss)
        arm = Arm(world, limits)
        attempts_path = out_dir / "simulation_attempts.jsonl"
        attempts = []

        def park() -> None:  # out of the camera's view, as the real run does before every photo
            steps = pickdrop.plan_park(limits, arm.known_pose(), the_ring.camera_clear, the_ring.shoulder_stop)
            pickdrop.run_steps(arm, steps, world.say, world.sleep)

        def look(shot: int):
            park()
            return f"simulation_check_{shot}", world.positions(target)

        def log(attempt: dict) -> None:
            attempt.update(simulation=True, sim_time_s=round(world.clock, 2),
                           cost_so_far_usd=summary["cost_usd"], time_s=round(time.monotonic() - started, 1))
            attempts.append(attempt)
            summary["attempts"] += 1
            with open(attempts_path, "a") as f:
                f.write(json.dumps(attempt) + "\n")

        def body():
            try:
                arm.enable(start, sleep=world.sleep)   # E at the start pose, like confirm_and_enable
                park()                                  # the real run parks at camera_clear for its photos
                ring_loop.carry(arm, moves, placed, the_ring, limits, look, log, world.say, world.sleep,
                                finished=summary["moves"], on_try=world.on_try)
                world.carrying, world.step = "", "back to the start pose"
            except Exception as error:
                summary["error"] = f"{type(error).__name__}: {error}"
                raise

        session.guarded(arm, body, world.say, world.sleep)   # a normal end: ramp to the start pose, then D
        summary["success"] = summary["error"] is None and all(m["result"] == "PASS" for m in summary["moves"])
        errors = [m["final_error_cm"] for m in summary["moves"] if m["final_error_cm"] is not None]
        summary["max_error_cm"] = max(errors) if errors else None
        summary["injected_miss"] = world.injected
        summary["sim_seconds"] = round(world.clock, 2)

        save_moves(out_dir / "simulation_moves.json", world, photo, the_ring, say)
        story = sim_video.Story(log=world.log, end_s=world.clock, ring=the_ring, layout=layout,
                                photo=str(photo_copy), seen=seen, placed=placed, moves=moves,
                                attempts=attempts, injected=world.injected, summary=summary)
        make_video(out_dir / "simulation_video.mp4", story, say=say)
        return summary
    except Exception as error:
        summary["error"] = summary["error"] or f"{type(error).__name__}: {error}"
        raise
    finally:
        summary["seconds"] = round(time.monotonic() - started, 1)
        (out_dir / "simulation_summary.json").write_text(json.dumps(summary, indent=2))
        mark_images(out_dir)
        say(f"Simulation log: {out_dir}  (cost ${summary['cost_usd']:.4f}, cap ${ring_loop.RUN_BUDGET_USD:.2f})")


def mark_images(out_dir: Path) -> None:
    """"simulation" in every picture's own metadata too (a JPEG comment, a PNG text chunk)."""
    for path in sorted(out_dir.glob("simulation_*")):
        if path.suffix.lower() in (".jpg", ".jpeg"):
            marks.mark_jpeg(path, MARK)
        elif path.suffix.lower() == ".png":
            marks.mark_png(path, "Comment", MARK)


def save_moves(path: Path, world: SimArm, photo: Path, the_ring, say=print) -> None:
    """Every line the simulated arm was sent, in order, saved and printed as JSON."""
    commands = [{"t_s": e["t_s"], "send": e["send"], "reply": e["reply"], "step": e["step"],
                 "carrying": e["carrying"], **({"simulated_claw": e["event"]} if e["event"] else {})}
                for e in world.log]
    bases = sorted({int(c["send"].split()[1]) for c in commands if c["send"].startswith("J")})
    outside = [b for b in bases if not pickdrop.BASE_APPROVED[0] <= b <= pickdrop.BASE_APPROVED[1]]
    data = {"simulation": True,
            "what": "SIMULATED EXECUTION: every line sent to the simulated arm, in order. The real arm would receive "
                    "these lines if every die landed where the simulation put it; in a real run each next pick "
                    "follows the camera.",
            "photo": str(photo), "made_up_ring": the_ring.note, "pick_numbers": "calib/ring.json",
            "limits": "calib/arm_limits.json (every angle checked against them)",
            "base_angles_outside_approved": {"approved": list(pickdrop.BASE_APPROVED), "angles": outside},
            "commands": commands}
    path.write_text(json.dumps(data, indent=2))
    say(f"The move list ({len(commands)} lines, SIMULATED EXECUTION):")
    for c in commands:
        say("  " + json.dumps(c))
    if outside:
        say(f"  note: base {', '.join(map(str, outside))} is outside the {pickdrop.BASE_APPROVED[0]}-"
            f"{pickdrop.BASE_APPROVED[1]} you approved (the made-up slots); arm_limits.json allows 0-180.")
    say(f"Saved {path}")
