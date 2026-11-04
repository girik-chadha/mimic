"""The simulation's video (mimic/sim.py): a top-down view of the made-up world, drawn with OpenCV, encoded by ffmpeg.

    a 2-second title card, then the run in real (simulated) time, then the last frame held for 2 seconds

Every frame has the "SIMULATED EXECUTION" banner. The view shows the sheet outline and its six dots, the made-up
ring and its numbered slots, every die as a square in its colour (with a hollow grey square where SAM saw it in the
real photo), and the claw: a filled dot when it is down at table height, a hollow ring that grows as it lifts, and
the held die under it. The side panel shows the move list, the step being run and the exact line sent, the last
camera check, and the real photo. "INJECTED TEST MISS" is shown while a deliberate miss is being corrected.

Frames are written as PNGs into the run folder, joined by Windows ffmpeg at 20 frames a second (H.264, with
"SIMULATION" in the video's title and comment), then deleted.
"""
import math
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mimic import camera, config

FPS = 20
TITLE_S = 2.0
END_S = 2.0
W, H = 1280, 720
BANNER_H = 56
VIEW_W, VIEW_H = 820, H - BANNER_H
PANEL_X = 836
VIEW_CM = ((-4.0, -4.0), (41.0, 31.0))      # the part of the table shown: x, y from - to
SCALE = min(VIEW_W / (VIEW_CM[1][0] - VIEW_CM[0][0]), VIEW_H / (VIEW_CM[1][1] - VIEW_CM[0][1]))
DIE_CM, SLOT_CM, DOT_CM = 1.9, 2.6, 0.5
TITLE_TEXT = ("SIMULATED EXECUTION: the same move list the arm would receive. "
              "Perception and planning ran live on a real photo.")
BANNER_TEXT = "SIMULATED EXECUTION"
VIDEO_TITLE = "Mimic SIMULATION (simulated execution, not the real arm)"
VIDEO_COMMENT = ("SIMULATION. The arm in this video is simulated on a made-up ring. Perception (SAM 3.1, Muse Spark "
                 "1.3) and planning ran live on a real photo; the move list is the one the real arm would receive.")
FFMPEG_TIMEOUT_S = 300
FONT = cv2.FONT_HERSHEY_SIMPLEX

BG, SHEET, INK, MUTED = (236, 236, 232), (255, 255, 255), (40, 40, 40), (120, 120, 115)
RED, YELLOW, RING = (40, 40, 200), (60, 215, 245), (200, 160, 120)
COLOURS = {"red": (50, 50, 215), "black": (35, 35, 35), "blue": (200, 120, 30), "green": (70, 150, 40),
           "yellow": (40, 200, 235), "white": (245, 245, 245), "orange": (30, 130, 245), "purple": (150, 60, 130),
           "pink": (170, 120, 240), "brown": (40, 75, 125), "grey": (140, 140, 140), "gray": (140, 140, 140)}


class SimVideoError(RuntimeError):
    pass


@dataclass
class Story:
    """Everything the video shows, from one simulated run."""
    log: list            # SimArm.log: one entry per line sent, with the time, angles and dice
    end_s: float         # the simulated run's length
    ring: object         # the made-up ring (mimic.ring.Ring)
    layout: object       # calib/layout.json (the dots and the sheet)
    photo: str           # the real photo
    seen: dict           # where SAM saw each die in the photo (cm)
    placed: dict         # where it was placed on the made-up ring (cm)
    moves: list          # plan.Move, slot 1 first
    attempts: list       # every camera check, with sim_time_s
    injected: dict | None
    summary: dict


# ---------- what is happening at time t ----------

def state_at(log: list, t: float) -> dict:
    """The angles (a move part-way through), the dice, the held die, and the last line sent, at time t."""
    last = None
    for entry in log:
        if entry["t_s"] > t:
            break
        last = entry
    if last is None:
        first = log[0]
        return {"angles": first["before"], "dice": first["dice"], "held": None, "line": "", "step": "",
                "carrying": ""}
    angles = last["after"]
    if last["send"].startswith("J"):
        ms = int(last["send"].split()[5])
        part = 1.0 if ms == 0 else min(1.0, (t - last["t_s"]) / (ms / 1000))
        angles = [b + (a - b) * part for b, a in zip(last["before"], last["after"])]
    return {"angles": angles, "dice": last["dice"], "held": last["held"], "line": last["send"],
            "step": last["step"], "carrying": last["carrying"]}


def checks_by(attempts: list, t: float) -> list:
    return [a for a in attempts if a["sim_time_s"] <= t]


def overlay_texts(story: Story, t: float) -> list[str]:
    """Big notices on the view: the injected miss (until the die that missed is checked again), and the end."""
    texts = []
    miss = story.injected
    if miss and t >= miss["t_s"]:
        later = [a for a in story.attempts if a["name"] == miss["name"] and miss["attempt"] is not None
                 and a["attempt"] > miss["attempt"]]
        if not later or t <= later[0]["sim_time_s"] + 1.0:
            texts.append(f"INJECTED TEST MISS: {miss['name']} put down {miss['degrees']:+g} degrees off on purpose")
    if t > story.end_s:
        moves = story.summary["moves"]
        placed = sum(1 for m in moves if m["result"] == "PASS")
        texts.append(f"Finished (simulated): {placed} of {len(moves)} dice within 1.5 cm of their slots")
    return texts


# ---------- drawing ----------

def to_px(cm) -> tuple[int, int]:
    (x0, _), (_, y1) = VIEW_CM
    top = BANNER_H + (VIEW_H - (VIEW_CM[1][1] - VIEW_CM[0][1]) * SCALE) / 2
    return (int(round((cm[0] - x0) * SCALE)), int(round(top + (y1 - cm[1]) * SCALE)))


def die_colour(name: str):
    for word in name.split():
        if word in COLOURS:
            return COLOURS[word]
    return (160, 160, 160)


def _text(img, text, xy, scale=0.5, colour=INK, thick=1):
    cv2.putText(img, text, (int(xy[0]), int(xy[1])), FONT, scale, colour, thick, cv2.LINE_AA)


def _square(img, centre_cm, size_cm, colour, thick):
    x, y = to_px(centre_cm)
    half = int(round(size_cm * SCALE / 2))
    cv2.rectangle(img, (x - half, y - half), (x + half, y + half), colour, thick, cv2.LINE_AA)


def _die(img, name, centre_cm):
    _square(img, centre_cm, DIE_CM, die_colour(name), -1)
    _square(img, centre_cm, DIE_CM, (90, 90, 90), 1)


def banner(img) -> None:
    cv2.rectangle(img, (0, 0), (W, BANNER_H), RED, -1)
    (tw, _), _ = cv2.getTextSize(BANNER_TEXT, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
    cv2.putText(img, BANNER_TEXT, ((W - tw) // 2, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)


def _thumbnail(photo: str):
    image = cv2.imread(photo)
    if image is None:
        return None
    width = W - PANEL_X - 10
    return cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])))


def draw_world(img, story: Story, state: dict) -> None:
    ring, layout = story.ring, story.layout
    (sx0, sy0), (sx1, sy1) = layout.sheet_min_cm, layout.sheet_max_cm
    cv2.rectangle(img, to_px((sx0, sy1)), to_px((sx1, sy0)), SHEET, -1)
    cv2.rectangle(img, to_px((sx0, sy1)), to_px((sx1, sy0)), (170, 170, 165), 2)
    for dot in layout.dots_cm.values():
        cv2.circle(img, to_px(dot), int(DOT_CM * SCALE), (95, 95, 95), -1, cv2.LINE_AA)
    centre = to_px(ring.centre_cm)
    cv2.circle(img, centre, int(round(ring.radius_cm * SCALE)), RING, 2, cv2.LINE_AA)
    label_at = to_px(ring.point_at(125))
    _text(img, "made-up ring", (label_at[0] + 10, label_at[1] + 22), 0.5, (150, 110, 70))
    for i, slot_deg in enumerate(ring.slots_deg, start=1):
        slot = ring.point_at(slot_deg)
        _square(img, slot, SLOT_CM, (150, 150, 150), 1)
        out = (ring.centre_cm[0] + (slot[0] - ring.centre_cm[0]) * 1.13,
               ring.centre_cm[1] + (slot[1] - ring.centre_cm[1]) * 1.13)
        _text(img, str(i), (to_px(out)[0] - 6, to_px(out)[1] + 6), 0.6, MUTED, 2)
    for name, seen in story.seen.items():                      # where SAM saw it -> where it was placed
        cv2.line(img, to_px(seen), to_px(story.placed[name]), (190, 190, 190), 1, cv2.LINE_AA)
        _square(img, seen, DIE_CM, (175, 175, 175), 1)
    for name, where in state["dice"].items():
        if where is not None:
            _die(img, name, where)
            _text(img, name, (to_px(where)[0] + 22, to_px(where)[1] + 5), 0.42, MUTED)
    base, shoulder = state["angles"][0], state["angles"][1]
    tip = to_px(ring.point_at(base))
    cv2.line(img, centre, tip, (150, 150, 150), 5, cv2.LINE_AA)
    cv2.circle(img, centre, 16, (110, 110, 110), -1, cv2.LINE_AA)
    _text(img, "arm base (made up)", (centre[0] + 22, centre[1] + 5), 0.45, MUTED)
    if state["held"]:
        _die(img, state["held"], ring.point_at(base))
    height = ring.shoulder_table - shoulder                    # degrees above the table
    if height <= 0.5:
        cv2.circle(img, tip, 7, INK, -1, cv2.LINE_AA)
    else:
        cv2.circle(img, tip, int(10 + 0.3 * height), INK, 2, cv2.LINE_AA)
    _text(img, "Hollow grey: where SAM saw each die in the real photo.  Coloured: " + "placed on the made-up ring.",
          (12, H - 12), 0.45, MUTED)


def draw_panel(img, story: Story, state: dict, t: float, thumb) -> None:
    x = PANEL_X
    _text(img, "MOVE LIST", (x, BANNER_H + 30), 0.65, INK, 2)
    done = checks_by(story.attempts, t)
    y = BANNER_H + 58
    for move in story.moves:
        pick = story.ring.angle_for(*story.placed[move.name])
        slot_deg = story.ring.slots_deg[move.slot - 1]
        mine = [a for a in done if a["name"] == move.name]
        now = state["carrying"].startswith(move.name + " ->")
        status = (f"{mine[-1]['result']}, {mine[-1]['miss_cm']:.1f} cm from the slot" if mine else "waiting")
        if now and (not mine or mine[-1]["result"] == "RETRY"):
            status = "now" + (f" (last try: {status})" if mine else "")
        colour = RED if now else INK
        _text(img, f"{move.slot}. {move.name}: base {pick:.0f} -> slot {move.slot} (base {slot_deg:g})", (x, y),
              0.5, colour, 2 if now else 1)
        _text(img, "   " + status, (x, y + 20), 0.45, colour)
        y += 48
    y += 8
    _text(img, "NOW", (x, y), 0.65, INK, 2)
    lines = [state["carrying"] or "-", f"step: {state['step'] or '-'}", f"sent: {state['line'] or '-'}",
             "base {:.0f}  shoulder {:.0f}  grip {:.0f}".format(state["angles"][0], state["angles"][1],
                                                                   state["angles"][3])]
    for i, line in enumerate(lines):
        _text(img, line[:52], (x, y + 26 + 22 * i), 0.5, INK)
    y += 26 + 22 * len(lines) + 14
    _text(img, "LAST CAMERA CHECK (simulated)", (x, y), 0.55, INK, 2)
    last = done[-1] if done else None
    text = (f"{last['name']} -> slot {last['move']}, try {last['attempt']}: {last['miss_cm']:.1f} cm -> "
            f"{last['result']}" if last else "-")
    _text(img, text[:52], (x, y + 24), 0.5, INK)
    if thumb is not None:
        top = H - thumb.shape[0] - 30
        img[top:top + thumb.shape[0], x:x + thumb.shape[1]] = thumb
        _text(img, "The real photo: SAM 3.1 and Muse Spark 1.3 ran live", (x, H - 10), 0.45, MUTED)
    _text(img, f"t = {min(t, story.end_s):.1f} s (simulated time)", (12, H - 34), 0.5, MUTED)


def draw_overlays(img, texts: list[str]) -> None:
    y = BANNER_H + 40
    for text in texts:
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
        x = max(10, (VIEW_W - tw) // 2)
        cv2.rectangle(img, (x - 12, y - th - 12), (x + tw + 12, y + 12), YELLOW, -1)
        cv2.rectangle(img, (x - 12, y - th - 12), (x + tw + 12, y + 12), INK, 2)
        _text(img, text, (x, y), 0.7, INK, 2)
        y += th + 34


def frame_at(story: Story, t: float, thumb=None):
    """One frame of the run at simulated time t (past the end: the last frame, with the end notice)."""
    img = np.full((H, W, 3), BG, np.uint8)
    state = state_at(story.log, min(t, story.end_s))
    draw_world(img, story, state)
    cv2.rectangle(img, (PANEL_X - 12, BANNER_H), (W, H), (248, 248, 246), -1)
    draw_panel(img, story, state, t, thumb)
    draw_overlays(img, overlay_texts(story, t))
    banner(img)
    return img


def title_card():
    img = np.full((H, W, 3), (45, 40, 38), np.uint8)
    lines = textwrap.wrap(TITLE_TEXT, 42)
    y = H // 2 - (len(lines) * 56) // 2 + 30
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, 1.25, 2)
        cv2.putText(img, line, ((W - tw) // 2, y), cv2.FONT_HERSHEY_DUPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)
        y += 56
    banner(img)
    return img


# ---------- the video file ----------

def frame_times(end_s: float) -> list:
    """None for each title-card frame, then the simulated time of each later frame (the end held for END_S)."""
    title = [None] * round(TITLE_S * FPS)
    return title + [i / FPS for i in range(math.ceil((end_s + END_S) * FPS) + 1)]


def ffmpeg_command(frames_windows_dir: str, out_windows_path: str) -> list[str]:
    return [config.FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(FPS),
            "-i", frames_windows_dir + "\\frame_%05d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-metadata", f"title={VIDEO_TITLE}", "-metadata", f"comment={VIDEO_COMMENT}", out_windows_path]


def render(out_path, story: Story, say=print, run=subprocess.run, to_windows=camera.windows_path) -> Path:
    out_path = Path(out_path)
    frames_dir = out_path.parent / "simulation_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    times = frame_times(story.end_s)
    say(f"Drawing {len(times)} frames ({len(times) / FPS:.1f} s at {FPS} fps)...")
    thumb, card = _thumbnail(story.photo), title_card()
    for i, t in enumerate(times):
        image = card if t is None else frame_at(story, t, thumb)
        cv2.imwrite(str(frames_dir / f"frame_{i:05d}.png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if out_path.exists():
        out_path.unlink()  # so an old file can never be mistaken for the new one
    command = ffmpeg_command(to_windows(frames_dir), to_windows(out_path))
    try:
        result = run(command, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise SimVideoError(f"ffmpeg took more than {FFMPEG_TIMEOUT_S} s; the frames are in {frames_dir}.")
    except FileNotFoundError:
        raise SimVideoError(f"ffmpeg not found at {config.FFMPEG_PATH}: set MIMIC_FFMPEG in .env. The frames are in "
                            f"{frames_dir}.")
    if result.returncode != 0:
        last = (result.stderr or "").strip().splitlines()[-1:] or ["no message"]
        raise SimVideoError(f"ffmpeg could not make the video (exit {result.returncode}): {last[0]}. The frames are "
                            f"in {frames_dir}.")
    if not out_path.exists():
        raise SimVideoError(f"ffmpeg finished but wrote no video. The frames are in {frames_dir}.")
    shutil.rmtree(frames_dir)
    say(f"Saved {out_path} (SIMULATED EXECUTION banner on every frame)")
    return out_path
