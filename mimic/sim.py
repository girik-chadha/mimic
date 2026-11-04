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
photo order: the dice left to right on the sheet go to slots 1, 2, 3. --instruction "stack black, blue, red" piles
them on slot 1, black at the bottom: SIMULATION ONLY (STACK_WHY), so those moves are marked "sim_only" and
run_mimic.py refuses a stack without --sim.

Outputs, in runs/<time>_simulation/, every name starting with "simulation_":
  simulation_moves.json      every line sent to the simulated arm, in order: what the real arm would receive if
                             every die landed where the simulation put it (in a real run each pick follows the
                             camera). Printed too. Each line has its step, and "simulated_claw" where the simulated
                             claw picked up or put down a die (or an INJECTED TEST MISS).
  simulation_summary.json, simulation_attempts.jsonl   like a real run's, with "simulation": true
  simulation_photo.jpg, simulation_photo_dots.jpg      the real photo and its labelled dots
  simulation_start.jpg, simulation_check_N.jpg, simulation_end.jpg   the simulated world before the first carry, at
                             each camera check, and at the end (frames of the video, for the report)
  simulation_video.mp4       the top-down video (mimic/sim_video.py)
  simulation_report.html, simulation_summary.png         scripts/report.py, run by run_mimic.py afterwards
"""
import dataclasses
import json
import math
import re
import shutil
import time
from pathlib import Path

import cv2

from mimic import arm as arm_module
from mimic import brain, config, marks, meta_api, pickdrop, plan, ring_loop, scene, session, sim_video
from mimic import ring as ring_module
from mimic.arm import Arm, FakeTransport
from mimic.ring import RingError

# The made-up ring: centre near the arm's base at the top of the sheet, radius chosen so it passes close to dice
# lying across the middle of the sheet. Base 90 points straight down the sheet (theta = phi - 180). Slots on the
# left of the ring, 16 degrees (5.6 cm) apart and at least 4 cm from every dot. Pick numbers come from ring.json.
MADE_UP_RING = {"centre_cm": (18.5, 27.0), "radius_cm": 20.0, "scale": 1.0, "offset_deg": -180.0,
                "slots_deg": [22, 38, 54]}
PICK_RADIUS_CM = 1.0          # the claw picks up a die whose centre is this close to the claw tip
INSTRUCTION_MODES = ("row", "stack")
DIE_SUFFIX = " die"           # the naming prompt asks for "colour die"; anything else is not a die
LABEL = "placed on the made-up ring"
SIM_TAG = "[simulated] "
MARK = "SIMULATION: an output of Mimic's simulation mode (run_mimic.py --ring --sim); the arm was simulated."
STACK_WHY = ("The real arm cannot stack: with the elbow taped, the claw reaches the table only at one shoulder "
             "angle, and lifting it (a smaller shoulder angle) also moves it along the radius, so it cannot let go "
             "of a die right on top of another. These moves lower to the table angle as usual; only the simulation "
             "puts the die on top.")


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
        self.ring, self.limits = ring, limits
        self.out = lambda text: say(SIM_TAG + text)  # every line about the arm says it is the simulated one
        self.dice = dict(dice)        # name -> (x, y) cm; None while held
        self.level = {n: 0 for n in dice}   # 0 on the table, 1 on top of one die, ...
        self.held = None
        self.clock = 0.0              # simulated seconds
        self.step = ""                # the step being run, from the last line printed
        self.carrying = ""            # "red die -> slot 1, try 1"
        self.try_of = None            # (name, attempt) of the carry being run
        self.miss = miss              # (name, degrees): that die's first put-down lands this far off
        self.injected = None          # {"name", "degrees", "t_s", "attempt"} once the miss has happened
        self.log = []                 # one entry per line sent
        self.stack = False            # a stack: the carry label says "the stack", not a slot

    def sleep(self, seconds: float) -> None:
        self.clock += seconds

    def say(self, text: str) -> None:
        """Print, and remember the step a pick-and-drop line names ("lower to 172: base 72, ...")."""
        self.out(text)
        self.step = text.strip().split(": base ")[0]

    def on_try(self, move, attempt: int) -> None:
        where = "the stack" if self.stack else f"slot {move.slot}"
        self.carrying, self.try_of = f"{move.name} -> {where}, try {attempt}", (move.name, attempt)

    def positions(self, names: list) -> dict:
        return {n: self.dice[n] for n in names if self.dice.get(n) is not None}

    def send(self, line: str) -> str:
        before = list(self.angles)
        reply = super().send(line)
        event = self._claw(before) if line.startswith("J") and reply == "OK" else None
        self.log.append({"t_s": round(self.clock, 3), "send": line, "reply": reply,
                         "step": self.step if line.startswith("J") else "", "carrying": self.carrying,
                         "before": before, "after": list(self.angles), "dice": dict(self.dice), "held": self.held,
                         "level": dict(self.level), "event": event})
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
            self.held = max(under, key=lambda n: self.level[n])   # the top one of a stack
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
            spot = self.ring.point_at(base + offset)
            below = [n for n, p in self.dice.items() if p is not None and math.dist(p, spot) <= PICK_RADIUS_CM]
            self.level[name] = max((self.level[n] + 1 for n in below), default=0)
            if below and not event.startswith("INJECTED"):
                event += f" on top of {max(below, key=lambda n: self.level[n])} (stack level {self.level[name] + 1})"
            self.dice[name], self.held = spot, None
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


GOAL_EXAMPLES = ('Goals that work: "put black on blue, then red on black" (a stack, built on slot 1 from the bottom '
                 'die up), "row black, blue, red" (slots 1, 2, 3), "stack black, blue, red" (bottom first), or leave '
                 "it empty for the photo's left-to-right order.")
PAIR = re.compile(r"^(?:(?:and|then|put|place)\s+)*(?:the\s+)?(.+?)\s+on\s+(?:top\s+of\s+)?(?:the\s+)?(.+)$")
NOUN = r"(?:die|dice|block|blocks|cube|cubes|object|objects|one)"
# "... stack the black die on top of the blue die ..." -> ("black", "blue"): the word before "on" (skipping a noun
# like "die") and the word after "on", "on top of" or "on the". Used when the goal is a longer sentence.
PAIR_IN_SENTENCE = re.compile(r"\b([a-z]+)(?:\s+" + NOUN + r")?\s+on\s+(?:top\s+of\s+)?(?:the\s+)?([a-z]+)\b")
MAX_NAME_WORDS = 3   # a clause whose X or Y is longer than this is a sentence: scan it for pairs instead


def parse_goal(text: str) -> tuple[str, list]:
    """The goal, read by code (no API call), before anything else runs:
        ""                                       -> ("photo", [])           the photo's left-to-right order
        "row black, blue, red"                   -> ("row", ["black", ...])
        "stack black, blue, red"                 -> ("stack", ["black", ...])   bottom first
        "put black on blue, then red on black"   -> ("pairs", [("black", "blue"), ("red", "black")])
    Anything else is refused with GOAL_EXAMPLES."""
    text = " ".join(text.strip().lower().split())
    if not text:
        return "photo", []
    first = text.split(None, 1)[0]
    if first in INSTRUCTION_MODES:
        try:
            return parse_instruction(text)
        except SimError:
            raise SimError(f"Not understood: {text!r}. {GOAL_EXAMPLES}") from None
    if " on " not in f" {text} ":
        raise SimError(f"Not understood: {text!r}. {GOAL_EXAMPLES}")
    pairs, sentence = [], False
    for clause in (c.strip() for c in re.split(r",|\bthen\b", text)):
        clause = re.sub(r"^(?:and\s+)+|\s+and$", "", clause).strip()
        if not clause:
            continue
        found = PAIR.match(clause)
        if not found or any(len(part.split()) > MAX_NAME_WORDS for part in found.groups()):
            sentence = True   # not a short "X on Y" clause: read the whole goal as a sentence
            break
        pairs.append((found.group(1).strip(), found.group(2).strip()))
    if sentence:
        pairs = PAIR_IN_SENTENCE.findall(text)
    if not pairs:
        raise SimError(f"No \"X on Y\" found in {text!r}. {GOAL_EXAMPLES}")
    return "pairs", pairs


def resolve_goal(mode: str, spec: list, names: list[str], seen: dict) -> list[str]:
    """The target order (for a stack: bottom first), with each colour matched to exactly one of Spark's names.
    "X on Y" pairs must build one pile: each Y is the die on top so far ("red on black" after "black on blue")."""
    if mode == "photo":
        return photo_order(seen)
    if mode in INSTRUCTION_MODES:
        return match_colours(spec, names)
    order = []
    for x_colour, y_colour in spec:
        x, y = match_colours([x_colour], names)[0], match_colours([y_colour], names)[0]
        if x == y:
            raise SimError(f"{x} cannot go on itself.")
        if not order:
            order = [y, x]
            continue
        if y != order[-1]:
            raise SimError(f"{x_colour} on {y_colour}: {y} is not on top of the pile ({order[-1]} is). Each pair "
                           f"puts a die on the one placed last. {GOAL_EXAMPLES}")
        if x in order:
            raise SimError(f"{x} is already in the pile.")
        order.append(x)
    return order


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


def pile_in_place(placed: dict, order: list[str], ring, limits):
    """For "X on Y" goals: the pile is built where its bottom die already lies, which stays put, if the arm can
    reach it and every carry onto it passes ring mode's checks. Returns (moves, a copy of the ring whose only slot
    is that spot); raises RingLayoutError or RingError if not, and the caller builds on slot 1 instead."""
    bottom = order[0]
    angle = ring.base_angle_for(*placed[bottom])
    lo, hi = limits.range_of("base")
    if not lo <= round(angle) <= hi:
        raise ring_loop.RingLayoutError(f"{bottom} is at base {angle:.0f}, outside the base range {lo}-{hi}")
    spot = dataclasses.replace(ring, slots_deg=[angle])
    moves = [plan.Move(n, tuple(placed[n]), tuple(placed[bottom]), slot=1) for n in order[1:]]
    ring_loop.check_mess(placed, moves, spot, limits, stack_base=bottom)
    return moves, spot


def stack_moves(placed: dict, order: list[str], ring) -> list:
    """Every die to slot 1, the first at the bottom (simulation only: STACK_WHY)."""
    if not order:
        raise SimError("A stack needs at least one die.")
    slot = ring.point_at(ring.slots_deg[0])
    return [plan.Move(name, tuple(placed[name]), tuple(slot), slot=1) for name in order]


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
    mode, spec = parse_goal(instruction or "")   # refused here, before any API call
    stack = mode in ("stack", "pairs")

    out_dir = Path(out_root) / f"{time.strftime('%Y%m%d-%H%M%S')}_simulation"
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    ring_note = {k: list(v) if isinstance(v, tuple) else v for k, v in MADE_UP_RING.items()}
    summary = {"simulation": True, "mode": "ring, simulated" + (", stack" if stack else ""), "stack": stack,
               "sim_only": STACK_WHY if stack else None, "photo": str(photo), "instruction": instruction,
               "target": None, "made_up_ring": ring_note, "injected_miss": None, "success": False, "moves": [],
               "attempts": 0, "max_error_cm": None, "cost_usd": 0.0, "seconds": 0.0, "sim_seconds": 0.0,
               "error": None, "out_dir": str(out_dir)}
    photo_copy = out_dir / f"simulation_photo{photo.suffix.lower()}"
    shutil.copyfile(photo, photo_copy)
    summary["photo_copy"] = photo_copy.name
    try:
        # Perception: real calls on the real photo, under the run's cost cap.
        meta_api.reset()
        budget_before = config.BUDGET_USD
        config.BUDGET_USD = min(config.BUDGET_USD, ring_loop.RUN_BUDGET_USD)
        try:
            say(f"Perception on {photo} (real SAM and Spark calls, cap ${ring_loop.RUN_BUDGET_USD:.2f})")
            calib = scene.calibrate(str(photo), layout, out_dir / "simulation_photo_dots.jpg")
            checks = [f"{role} {calib.check(px, cm):.2f} cm" for role, px, cm in
                      zip(scene.ROLES[4:], calib.pixel_pts[4:], calib.table_pts_cm[4:])]
            say(f"  SAM found the 6 dots; the photo calibrated itself, check dots off by {', '.join(checks)}  "
                f"(${meta_api.cost_so_far():.4f} so far)")
            naming = []
            names = brain.name_objects(str(photo), attempts=naming)
            say(f"  Spark named: {', '.join(names)}; SAM found exactly 1 box for each  "
                f"(${meta_api.cost_so_far():.4f} so far)")
            dice = [n for n in names if n.endswith(DIE_SUFFIX)]
            if len(dice) < len(names):
                say(f"  not dice, left out: {', '.join(n for n in names if n not in dice)}")
            boxes = {}
            seen = scene.read_positions(str(photo), dice, calib, layout, boxes=boxes)
            summary["sam"] = {"check_cm": {role: round(calib.check(px, cm), 2) for role, px, cm in
                                           zip(scene.ROLES[4:], calib.pixel_pts[4:], calib.table_pts_cm[4:])},
                              "boxes": {n: [b.x1, b.y1, b.x2, b.y2] for n, b in boxes.items()},
                              "seen_cm": {n: [round(v, 2) for v in seen[n]] for n in dice}}
            summary["naming"] = naming
            draw_sam(out_dir / "simulation_photo_dots.jpg", boxes, seen, out_dir / "simulation_sam.jpg")
        finally:
            config.BUDGET_USD = budget_before
            summary["cost_usd"] = round(meta_api.cost_so_far(), 4)
        target = resolve_goal(mode, spec, dice, seen)
        summary["target"] = target
        summary["goal_read"] = {"mode": mode, "spec": spec}
        how = "photo order, left to right" if mode == "photo" else "read from the goal by code"
        if stack:
            say(f"Target ({how}): a stack, bottom first: {', '.join(target)}. SIMULATION ONLY: {STACK_WHY}")
        else:
            say(f"Target ({how}), slot 1 first: {', '.join(target)}")

        # Planning: the dice placed on the made-up ring, then ring mode's own checks, before any move.
        placed = place_on_ring(seen, the_ring)
        for name in dice:
            say(f"  {name}: photo ({seen[name][0]:.1f}, {seen[name][1]:.1f}) cm -> {LABEL} at base "
                f"{the_ring.angle_for(*placed[name]):.0f}, ({placed[name][0]:.1f}, {placed[name][1]:.1f}) cm, "
                f"moved {math.dist(seen[name], placed[name]):.1f} cm")
        slots_cm = [the_ring.point_at(a) for a in the_ring.slots_deg]
        carry_ring, in_place = the_ring, False
        if mode == "pairs":   # "X on Y": build where the bottom die lies, if it can be reached there
            try:
                moves, carry_ring = pile_in_place(placed, target, the_ring, limits)
                in_place = True
                say(f"  the pile is built where {target[0]} lies (base {carry_ring.slots_deg[0]:.0f}); it stays put")
            except (ring_loop.RingLayoutError, RingError) as error:
                say(f"  {target[0]} cannot be the base where it lies ({error}); building the pile on slot 1 instead")
        if not in_place:
            moves = stack_moves(placed, target, the_ring) if stack else plan.plan_moves(placed, target, slots_cm)
            ring_loop.check_mess(placed, moves, the_ring, limits)
        summary["stack_in_place"] = in_place
        summary["seen_cm"] = {n: [round(v, 2) for v in seen[n]] for n in dice}       # SAM, in the real photo
        summary["placed_cm"] = {n: [round(v, 2) for v in placed[n]] for n in dice}   # on the made-up ring
        summary["carries"] = [{"name": m.name, "slot": m.slot, "to_cm": [round(v, 2) for v in m.to_cm]} for m in moves]

        # Execution on the simulated arm, exactly as run_mimic.py --ring drives the real one.
        miss = (moves[0].name, miss_deg) if miss_deg else None
        world = SimArm(limits, the_ring, placed, say=say, miss=miss)
        world.stack = stack
        arm = Arm(world, limits)
        attempts_path = out_dir / "simulation_attempts.jsonl"
        attempts = []

        def park() -> None:  # out of the camera's view, as the real run does before every photo
            steps = pickdrop.plan_park(limits, arm.known_pose(), the_ring.camera_clear, the_ring.shoulder_stop)
            pickdrop.run_steps(arm, steps, world.say, world.sleep)

        def look(shot: int):
            park()
            return f"simulation_check_{shot}.jpg", world.positions(target)

        def log(attempt: dict) -> None:
            attempt.update(simulation=True, sim_time_s=round(world.clock, 2),
                           cost_so_far_usd=summary["cost_usd"], time_s=round(time.monotonic() - started, 1))
            attempts.append(attempt)
            summary["attempts"] += 1
            with open(attempts_path, "a") as f:
                f.write(json.dumps(attempt) + "\n")

        first_carry_s = []

        def body():
            try:
                arm.enable(start, sleep=world.sleep)   # E at the start pose, like confirm_and_enable
                park()                                  # the real run parks at camera_clear for its photos
                first_carry_s.append(world.clock)
                ring_loop.carry(arm, moves, placed, carry_ring, limits, look, log, world.say, world.sleep,
                                finished=summary["moves"], on_try=world.on_try, stack=stack,
                                already_stacked=target[:1] if in_place else None)
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
        summary["plan"] = plan_in_words(attempts, the_ring, limits)

        save_moves(out_dir / "simulation_moves.json", world, photo, the_ring, say, stack_bottom=target[0] if stack
                   else None)
        story = sim_video.Story(log=world.log, end_s=world.clock, ring=the_ring, layout=layout,
                                photo=str(photo_copy), seen=seen, placed=placed, moves=moves,
                                attempts=attempts, injected=world.injected, summary=summary, stack=stack)
        sim_video.save_stills(story, out_dir, first_carry_s[0] if first_carry_s else 0.0)
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


def draw_sam(dots_image: Path, boxes: dict, seen: dict, out_path: Path) -> None:
    """What SAM saw: the labelled dots photo, plus each die's box and its colour."""
    image = cv2.imread(str(dots_image))
    if image is None:
        return
    for name, box in boxes.items():
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), (255, 209, 79), 2)
        label = name.removesuffix(DIE_SUFFIX)   # short, so dice side by side do not overlap (positions: the page)
        (width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(image, label, ((box.x1 + box.x2 - width) // 2, box.y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 209, 79), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])


def plan_in_words(attempts: list, ring, limits) -> list[dict]:
    """Each try of each carry with the numbers the page's Plan tab shows."""
    return [{"name": a["name"], "slot": a["move"], "try": a["attempt"], "pick_deg": a["pick_deg"],
             "drop_deg": a["drop_deg"], "above": ring.shoulder_table - ring.approach, "table": ring.shoulder_table,
             "lifted": ring.shoulder_table - ring.lift, "grip_hold": ring.grip_hold, "grip_open": limits.grip_open_deg,
             "claw_closed_s": a["claw_closed_s"], "claw_limit_s": limits.max_grip_hold_s - pickdrop.HOLD_MARGIN_S,
             "result": a["result"], "miss_cm": a["miss_cm"], "next_correction_deg": a["next_correction_deg"]}
            for a in attempts]


def mark_images(out_dir: Path) -> None:
    """"simulation" in every picture's own metadata too (a JPEG comment, a PNG text chunk)."""
    for path in sorted(out_dir.glob("simulation_*")):
        if path.suffix.lower() in (".jpg", ".jpeg"):
            marks.mark_jpeg(path, MARK)
        elif path.suffix.lower() == ".png":
            marks.mark_png(path, "Comment", MARK)


def save_moves(path: Path, world: SimArm, photo: Path, the_ring, say=print, stack_bottom: str | None = None) -> None:
    """Every line the simulated arm was sent, in order, saved and printed as JSON. For a stack, every line of a carry
    onto the stack (every die but the bottom one) is marked "sim_only": true."""
    def onto_stack(e) -> bool:
        return stack_bottom is not None and bool(e["carrying"]) and not e["carrying"].startswith(stack_bottom + " ->")

    commands = [{"t_s": e["t_s"], "send": e["send"], "reply": e["reply"], "step": e["step"],
                 "carrying": e["carrying"], **({"simulated_claw": e["event"]} if e["event"] else {}),
                 **({"sim_only": True} if onto_stack(e) else {})}
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
            "sim_only": STACK_WHY if stack_bottom else None,
            "commands": commands}
    path.write_text(json.dumps(data, indent=2))
    say(f"The move list ({len(commands)} lines, SIMULATED EXECUTION):")
    for c in commands:
        say("  " + json.dumps(c))
    if stack_bottom:
        say(f"  note: the lines marked sim_only (stacking) can never go to the real arm: {STACK_WHY}")
    if outside:
        say(f"  note: base {', '.join(map(str, outside))} is outside the {pickdrop.BASE_APPROVED[0]}-"
            f"{pickdrop.BASE_APPROVED[1]} you approved (the made-up slots); arm_limits.json allows 0-180.")
    say(f"Saved {path}")
