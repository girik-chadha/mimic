# Mimic: implementation plan

Companion to [description.md](description.md) (what we are building) and [needs.md](needs.md) (verified facts and open questions). This file is how we build it, in what order, and how we prove each phase works before moving on. [todo.md](todo.md) is the short list of what is left.

Solo build. Claude Code writes code; Girik reviews, runs hardware, and must be able to explain every file.

---

## 1. Repo layout

```
mimic/                     the Python package
  __init__.py
  config.py                settings from .env, model names, prices, budget, paths
  meta_api.py              one client for every Meta call: data URLs, retries,
                           response cache, cost counter, budget guard
  sam.py                   parse SAM output (boxes, video tracks); find(image, name); track(video, name)
  video.py                 (3 Nov) demo video: shrink to 10 fps / 720p, settled frame, arrival order
  brain.py                 Spark calls: name_objects(photo), diagnose(before, after, move)
  scene.py                 read_target(photo) -> ordered names; read_positions(photo) -> px
  calib.py                 pixel <-> table centimetres (homography from 4 corner dots)
  plan.py                  plan_moves(positions_cm, target_order, slots_cm) -> moves
  kinematics.py            table cm -> joint angles (maths or taught-pose grid)
  arm.py                   Arm: move(angles), grip(), home(), status()
  camera.py                capture(path) -> photo
  loop.py                  run(): the whole see-plan-act-check loop, writes runs/<time>/
  web.py                   (Phase 5) local web page: camera, plan, attempt log
firmware/
  mimic_arm/mimic_arm.ino  ESP32: tiny command protocol, smooth moves, safety limits
scripts/
  plan_from_photos.py      Gate 1 demo: two photos in, plan out
  show_demo.py             (3 Nov) the demo as a video: target order and arrival order out (opt-in;
                           run_mimic.py still takes a demo photo)
  arm_jog.py               move one joint at a time from the terminal
  run_mimic.py             the full loop
  benchmark.py             N runs, prints success rate, retries, cost
  spikes/                  day-2 test scripts, kept as a record (sam_test.py, ...)
tests/
  fixtures/meta_responses/ real API replies saved to disk (no network in unit tests)
  fixtures/photos/         test photos
  unit/                    fast, no network, no hardware
  api/                     real Meta API calls, opt-in: pytest -m api
  hw/                      needs the arm or camera, opt-in: pytest -m hw
calib/                     saved calibration + learned offsets (committed)
runs/                      run logs and photos (gitignored, keep the best ones for the video)
docs/                      description.md, plan.md, needs.md, todo.md, LEARN.md
CLAUDE.md                  context Claude Code reads every session
BUILD_LOG.md               dated record of progress (proof it was built during the event)
pyproject.toml             dependencies, pytest markers
.env.example               variable names, no values
```

---

## 2. Contracts

### 2.1 Python interfaces

Keep these names and shapes. Everything downstream depends on them.

```python
# mimic/sam.py
@dataclass
class Box:
    x1: int; y1: int; x2: int; y2: int; w: int; h: int   # w, h = size of the whole image
    id: int = 0                                          # match id (photo) / track id (video)
    @property
    def centre(self) -> tuple[float, float]: ...

def parse_boxes(text: str) -> list[Box]
def parse_tracks(text: str) -> dict[int, list[tuple[int, Box]]]   # track id -> [(frame, box)]
def find(image_path: str, name: str, use_cache=True) -> list[Box]  # cached, cost-counted

# mimic/brain.py
def name_objects(image_path: str, attempts: list | None = None) -> list[str]
    # Spark looks at the demo photo, returns SAM-friendly names: ["red block", "blue block", ...]
    # each name must give exactly 1 SAM box; failures go back to Spark (max 2 retries, $0.03 cap),
    # then NamingFailed. Every check is appended to `attempts` as {round, name, boxes}. (changed 29 Oct)
    # (3 Nov) both prompts (naming and rename) say the objects are small painted dice, named "colour die"
def diagnose(before_path: str, after_path: str, move, error_cm: tuple[float, float]) -> dict   # (2 Nov)
    # one Spark call with both photos and the measured miss
    # {"problem": slipped|dropped|knocked|missed_aim|not_found|other,
    #  "fix": {"dx_cm": float, "dy_cm": float, "retry": bool}, "explanation": str (<= 300 chars)}
    # DiagnosisError on an unusable reply, or if spent + $0.02 would pass the run's cost cap

# mimic/scene.py   (changed 29 Oct: takes a Layout loaded from calib/layout.json)
def load_layout(path=config.LAYOUT_PATH) -> Layout
    # dot positions by role, dot prompt + size range, slots_cm, slot_tolerance_cm (<= half the slot spacing), sheet bounds,
    # height correction (off by default: camera height, point under the lens, point height)
def find_dots(image_path: str, layout: Layout) -> (centres, dropped)   # one SAM call, size filter
def assign_roles(dots_px: list, layout: Layout) -> list                # 6 centres -> ROLES order
def choose_dots(candidates: list, layout: Layout) -> (ordered, leftovers)   # (2 Nov) 6-12 dot-sized boxes:
    # every set of 6 in its 8 orders; keep the set whose check dots land within 0.5 cm; stop if another set
    # fits within 0.1 cm of the best, or if there are more than 12
def calibrate(image_path: str, layout: Layout, labelled_path=None) -> Calib
    # (changed 30 Oct) every photo calibrates itself: exactly 6 dots, roles by position (4 rotations),
    # map from the 4 corners, both check dots within 0.5 cm; saves a labelled photo.
    # Refuses while layout.json has "placeholder": true. No saved camera map; the camera may move.
def read_positions(image_path: str, names: list[str], calib, layout) -> dict[str, tuple[float, float]]  # cm
def read_target(image_path: str, names: list[str], calib, layout) -> list[str]     # names in slot order
    # every name must give exactly 1 SAM box; off the sheet, not in a slot, or two in one slot -> SceneError

# mimic/calib.py
def fit(pixel_pts: list[tuple], table_pts_cm: list[tuple]) -> Calib
    # first 4 points (corner dots) build the map; points 5+ are check points kept out of the fit,
    # and any more than 0.5 cm off raises CalibrationError (changed 29 Oct)
Calib.px_to_cm(pt) -> tuple[float, float];  Calib.cm_to_px(pt) -> tuple[float, float]
Calib.check(pixel_pt, cm_pt) -> float;  Calib.fit_report() -> float;  Calib.save(path) / Calib.load(path)

# mimic/plan.py
@dataclass
class Move:
    name: str; from_cm: tuple[float, float]; to_cm: tuple[float, float]; slot: int
def plan_moves(positions_cm: dict, target_order: list[str], slots_cm: list[tuple]) -> list[Move]
    # raises ValueError if the target is empty, a name is missing or duplicated, there are more names
    # than slots, two targets are under 1 cm apart, or two slots are the same point
def extra_objects(positions_cm: dict, target_order: list[str]) -> list[str]   # left alone, reported

# mimic/kinematics.py   (2 Nov: taught-pose grid, calib/pose_grid.json)
PoseGrid.to_angles(x_cm, y_cm, height) -> dict   # height "hover" | "grab"; base, shoulder, elbow
    # 1/d^2 blend of the 4 nearest taught points, exact at a taught point; refuses outside the hull of the
    # taught points + 1 cm, and while the grid is a placeholder (< 4 complete points, no grip/camera_clear)
PoseGrid.check_against(limits)                   # every taught angle inside arm_limits.json

# mimic/motion.py   (2 Nov)
def plan_pick_and_place(grid, limits, start, from_cm, to_cm) -> (steps, hold_s)
    # hover, lower, close, lift, travel, lower, release, lift; MotionError if the planned grip hold
    # (close -> release) is over max_grip_hold_s - 0.5 s, before anything moves
def pick_and_place(arm, grid, limits, from_cm, to_cm) -> hold_s   # prints the planned hold time
def go_camera_clear(arm, grid, limits)                            # a normal move out of the photo

# mimic/arm.py   (29 Oct: limits from calib/arm_limits.json; transport = anything with send(line) -> reply, is_real)
class Arm:
    def __init__(self, transport, limits: ArmLimits)
    def move(self, angles: dict, ms: int = 800) -> None   # raises ArmError outside the limits / too fast
    def grip(self, closed: bool) -> None
    def home(self) -> None
    def status(self) -> dict                             # angles + "locked" (2 Nov); locked angles are not kept
    def known_pose(self) -> dict                         # refuses if the firmware says it is locked
    def stop(self) -> None                               # always allowed
    # a real transport refuses move/grip/home unless placeholder is false AND limits_checked_by_hand is true
class SerialBridgeTransport:                             # (2 Nov) the real arm via scripts/serial_bridge.py
    def send(self, line: str) -> str                     # ArmError on any timeout; then refuses further commands
    def close(self) -> None                              # the bridge sends S and exits

# mimic/camera.py   (2 Nov: iVCam via Windows ffmpeg, needs.md A9)
def capture(path: str) -> str   # CameraError: ffmpeg fails / > 10 s, no file, nearly black, < 1280x720

# mimic/loop.py   (2 Nov)
def run(arm, capture, layout, grid, limits, wait_for_user=input, out_root=RUNS_DIR) -> dict
    # summary: success, moves, attempts, max_error_cm, cost_usd, seconds, error. $0.15 cap.
    # pick from the latest photo; PASS within 1.5 cm; else aim = slot - total miss (each axis within
    # +-2 cm), pick it up from where it landed, at most 2 retries, then FAIL and go on.
def choose_correction(slot, measured, previous, before_photo, after_photo, move) -> (correction, info)
    # (2 Nov) the numeric correction decides. Spark's fix is logged and printed only, unless
    # MIMIC_USE_SPARK_FIX=1; then it is used only if within 2 cm per axis, not pointing the same way as a
    # miss over 0.5 cm, and consistent with the measurement. Total limited to +-2 cm. At most 2 diagnoses
    # per run; full-run cap $0.30.
def consistency(advice, miss) -> (agrees | None, words)   # "contradicts measurement" if Spark's seen_offset_cm
    # is more than 0.5 cm from the measured miss, or it says not_found while the camera measured the block
```

### 2.2 Firmware command protocol

One line in, one line out, the same over USB serial or Wi-Fi HTTP (`GET /cmd?c=<line>`):

| Command | Meaning | Reply |
|---|---|---|
| `J <base> <shoulder> <elbow> <grip> <ms>` | move all joints smoothly to these angles over `ms` milliseconds | `OK` or `ERR <reason>` |
| `H` | go to the home pose | `OK` |
| `?` | report current angles | `A <base> <shoulder> <elbow> <grip>`, then ` L` while locked or ` D` while disarmed (3 Nov) |
| `S` | stop and hold | `OK` |
| `E [<base> <shoulder> <elbow> <grip>]` | enable (added 3 Nov): switch the servos on, one every 400 ms, each starting at these angles (default 90) | `OK`, `ERR already enabled`, `ERR limits not measured` |
| `D` | disable (added 3 Nov): stop, then switch every servo off (limp) | `OK` |
| `B` | why did the board last start? (added 3 Nov) | `B <reason> <seconds since start-up>`: power-on, brownout, software reset, watchdog, other |

Safety in the firmware, not only in Python: every angle is clamped to the limits measured on Friday (needs.md B4); moves are ramped, never jumps; the gripper never closes past its measured limit (overheating).

Details settled 29 Oct (firmware/mimic_arm/arm_logic.h, tested with g++ on the laptop):
- `J` replies `OK` as soon as the move starts; the ramp runs in the background, so `S` and `?` work mid-move.
  A new `J` replaces a running ramp, starting from where the arm is.
- Moves are slowed to at least `MIN_MS_PER_DEGREE` per degree. A grip held closed for `MAX_GRIP_HOLD_MS` relaxes to open.
- Lines over 64 characters are rejected. With no command, the arm holds; the last move is never repeated.
- `J` and `H` reply `ERR limits not measured` unless the limits are real and checked by hand; `?` and `S` always work.
- (2 Nov) While locked, no servo is attached and no angle is written (`servos_enabled()`), and `?` replies
  `A b s e g L`: the L says these are the firmware's own numbers, not a real pose.
- (3 Nov, Girik's decision) **The board starts disarmed:** nothing is attached at power-on. `E` switches the servos
  on one at a time (`servos_to_attach()`, now counted from E), `D` switches them all off. `J` and `H` reply
  `ERR not enabled` until all four are on; `?`, `S`, `B` and `D` always work. The lock and the ramp are unchanged.
  Python: `arm.enable()` / `arm.disable()`; every script asks before enabling; `s`, Ctrl+C and errors send S then
  D; a normal quit says "Hold the arm" and disables. Grip range 40-140 (`GRIP_RANGE`), closed limit 140 one end of it.
- (3 Nov) `B` is read-only and always allowed, like `?` and `S`. The reason comes from the ESP32's reset code
  (mapping in `arm_logic.h`, tested with g++; the numbers are checked against the ESP32's own names at compile time).
  It shows ESP32 resets only, not a sag in the servo battery that leaves the ESP32 running (needs.md A10b).
- Python (`arm.py`) refuses out-of-range angles; the firmware clamps. The same numbers live in
  `calib/arm_limits.json` and `arm_logic.h`, and a unit test checks they match.

### 2.2b The demo as a video (added 3 Nov, Girik's decision; photos stay the default)

`scripts/show_demo.py demo.mp4`: the human films the demo instead of taking one photo.
- Windows ffmpeg shrinks it first: 10 fps, shorter side at most 720 px, no sound, no metadata. An 8 s clip is
  about 80 frames. Refused before anything is sent if it could cost more than $0.30 (3 blocks: about 40 s).
- A warning (hands, faces and anything in view go to Meta) and a typed `yes` before anything is sent.
- The last frame is calibrated (6 dots) and named by Spark first, so a hidden dot or wrong layout stops for about
  $0.01 before the video calls. The video must end with the blocks placed and hands away for a second.
- One SAM video call per name. Stops if a name has no track, two track ids, or is lost (last seen more than 1 s
  before the end), or if SAM's frame count or picture size does not match the video.
- The frame read: the last one where every block was seen in each of the previous 10 frames (1 s) and stayed
  within 1.5% of the image width of where it is then; up to 3 such frames, 1 s apart, are tried for the 6 dots.
  On it the existing pipeline runs (`read_positions`, `assign_slots`). Stops if a block's photo box and video box
  disagree by more than 1 cm, or if a block ends the video more than the slot tolerance from where it was in that
  frame (the video ended before the last block settled).
- Arrival order (first to settle first) is logged in `demo.json`, not used: the row task fills slots in order.
- `meta_api`: a video call is priced from its frame count before it is sent and refused if it would pass the
  budget; it gets a 300 s timeout and is never retried after a timeout (Meta may still bill it).

### 2.2c Push mode (opt-in, added 3 Nov, Girik's decision): a fallback if the claw cannot hold a block

`motion.plan_push(grid, limits, start, from_cm, to_cm, others_cm)` plans one straight push, before anything moves:
- **Claw angle `grip_push`** (in `calib/pose_grid.json`): where the empty jaw tips just touch, minus 2 degrees, found
  in the grip test. **Unknown (null) until the grip test, and push mode refuses to run without it.** Never the
  140 closed limit: a stalled claw overheats.
- **Direction:** the straight line from the block's centre to its slot. The claw comes down 3 cm behind the block
  (on the far side from the slot), closes to `grip_push` in the air first, then lowers to grab height.
- **Slide** in steps of at most 1.5 cm, all at grab height, until the claw is 2 cm short of the slot (so the
  block's centre ends on it); back off 1 cm, lift, open. Short steps because one long move turns all the joints
  together, and the claw would curve off the line and rise or dip between two grab poses.
- **Grip timer:** the claw-closed time (close to lift) must fit in max_grip_hold_s minus 0.5 s (4.5 s), the same
  rule as pick and place. A longer push is cut at the last step that fits (about 9 cm per push) and the plan says
  where the block should end up; the camera check and the next push carry on.
- **Not knocking other blocks:** every other block's centre must be at least 4 cm from the path (from where the
  claw lands to the slot), or nothing moves. Every waypoint goes through the taught grid, so the path must stay
  inside the taught area plus 1 cm.
- Provisional sizes (a 2 cm block, a claw 1-2 cm across) until measured. Pushes are straight lines only; a block
  is never pulled.

`scripts/run_mimic.py --push` (`loop.run(push=True)`; without the flag nothing changes):
- Refused before anything is opened if `grip_push` is unknown.
- **Order:** the slot furthest from the mess area first (slot 3, 2, 1), so a pushed block never passes through a
  filled slot. If that block's path passes near another block, the next block whose path is clear goes first;
  if none is clear, the run stops ("spread the blocks out").
- **After every push:** camera clear, photo, calibrate, and measure **every** block. Within 1.5 cm of the slot:
  PASS. Otherwise push again from where it is now (the direction is worked out again, so an overshoot is pushed
  back from the other side), at most 4 pushes per block. A push that moved it less than 0.5 cm missed it (the
  claw passed beside it); two in a row is FAIL, and the run goes on with the next block.
- **Knocks:** another block that moved more than 1 cm is logged; a block in the mess area simply carries on from
  its new position, but a block knocked out of its slot stops the run.
- Each check costs about $0.01 with 3 blocks (one dot photo plus one photo per block). No Spark diagnosis in push
  mode. Every push is a line in `attempts.jsonl` with `"mode": "push"`.
- `scripts/teach_plan.py --push` lists the extra points to teach: where the claw lands behind the mess area's
  corners, and beyond the last slot (to push an overshoot back), when they are more than 1 cm outside the normal
  points' area. A sideways correction at a slot may still need a point outside it, and is then refused (FAIL).

### 2.3 Scene layout

- Work area: an A3 sheet taped down. Table coordinates in cm with the origin at the sheet's bottom-left dot.
- Six dots drawn on the sheet (changed 30 Oct): 4 corner dots build each photo's map, 2 check dots test it.
  SAM finds all 6 with one prompt ("small coloured dot"); which dot is which comes from where they are, not
  their colour. Dot positions by role, slots and sheet bounds live in `calib/layout.json`.
- (3 Nov, Girik's decision) Two flags there: `dots_measured` is enough to calibrate (`scene.calibrate`,
  `show_demo.py`, `plan_from_photos.py`, and `repeatability.py`, which only measures with the camera and is still
  guarded by the arm's own lock); `placeholder: false` (the whole file real, which needs `dots_measured` too) is
  still required by everything that uses slots, mess area or reach with the arm: `run_mimic.py`, the loop,
  `teach_plan.py`. The dots were ruler-measured 3 Nov; the rest is still placeholder.
- **Keep every block and slot at least 4 cm from every dot:** a covered dot stops the run, because each photo
  needs all 6 dots to calibrate itself.
- **Mess area** on one side, **target row** of numbered slots on the other. Slots start empty, so a placement never lands on another block.
- Arm base at a fixed, measured spot just outside the **top** edge of the sheet (placeholder (18.5, 30) cm).
  `reach_cm` in `layout.json`: what it can reach (placeholder 6-20 cm from the base: not right under the base,
  not further than 20 cm). `mess_area_cm`: the rectangle where blocks are scattered. Slots and the mess area must
  be inside the reach and at least 4 cm from every dot (changed 2 Nov). `scripts/teach_plan.py` lists the points
  to teach: slots, then the mess area's centre and corners.

### 2.4 Run log (every run, `runs/<YYYYMMDD-HHMMSS>/`)

- `demo.jpg`, `mess.jpg`, `after_<n>.jpg` per attempt, `final.jpg`
- `attempts.jsonl`: one line per attempt: move, target cm, measured cm, error cm, PASS/FAIL, correction used, cost so far
- `summary.json`: success, total attempts, max error, cost, seconds

This log is the evidence for the video, the README numbers and the judges.

---

## 3. Phases, tests and exit criteria

### Phase 0: Setup and first contact (Gate 0, Wed 28 Oct, done)

Done: API key, SAM on photo and video, Spark JSON, public repo, first commit. See BUILD_LOG.md.

### Phase 1: See and plan, no hardware (Gate 1, Thu 29 Oct)

Tasks, in order:

1. **Tidy the repo.** Create the `mimic/` package and folders from section 1. Move the day-2 scripts into `scripts/spikes/`. Add `pyproject.toml` (dependencies and pytest markers `api`, `hw`), `.env.example`, and gitignore entries for `runs/`, `.cache/`.
2. **`config.py` + `meta_api.py`.** One OpenAI client. `image_url(path)` / `video_url(path)` data URLs. `call(model, content)` with retries on 429/5xx, a disk cache keyed on (model, prompt, file hash) in `.cache/`, and a cost counter using the prices in needs.md A4. Raise `BudgetExceeded` when the run passes `MIMIC_BUDGET_USD`. Save real replies to `tests/fixtures/meta_responses/` with a `--save` flag.
3. **`sam.py`.** Move the parsing from the spikes into `parse_boxes` and `parse_tracks`; add `find()`. Unit tests on the saved real replies (the lighter photo and the 317-frame video).
4. **Settle needs.md B1:** can Spark see images? Write `brain.name_objects()`. One `api` test on `desk.jpeg` expecting three names. Record the answer in needs.md.
5. **`plan.py`.** `plan_moves` with the errors from section 2.1. Unit tests: normal case, missing name, duplicate name, too many names.
6. **`calib.py`.** Homography from 4 point pairs (OpenCV `findHomography` or NumPy). Unit tests: synthetic points round-trip px -> cm -> px within 0.01; a known square maps to the right size.
7. **`scene.py`.** `read_target` (sort detected centres along the target row) and `read_positions`.
8. **`scripts/plan_from_photos.py`.** Demo photo + mess photo in; prints names, positions in cm and the move list; saves an annotated image (boxes, labels, arrows) to `runs/`.
9. **Windows toolchain (Girik, evening):** install Arduino CLI or IDE 2, the ESP32 core, `ESP32Servo`, and the CP210x driver. Decide the camera (needs.md B2).

Tests: all `unit` tests green with no network; the `api` test for B1 run once and its result recorded.

**Exit (Gate 1):** from two real photos of the dot sheet, `plan_from_photos.py` prints the right order, uses names chosen by Spark, and the cm positions are within 1 cm of a ruler measurement.

### Phase 2: Arm bring-up (Gate 2, Fri 30 Oct)

1. Assemble the arm from the KEYESTUDIO PDF. Centre the servos during assembly (the kit's centring sketch is fine for this step only).
2. Measure and write into needs.md B4: pins, joint ranges, link lengths, gripper opening.
3. `firmware/mimic_arm/mimic_arm.ino`: the protocol from section 2.2 over USB serial first, with clamps and ramps.
4. Settle needs.md B3 (Wi-Fi hotspot vs Windows bridge) and add that transport to the firmware.
5. `arm.py` for the chosen transport. `scripts/arm_jog.py` to move one joint at a time.
6. `camera.py` (from the B2 decision). Mount the camera.

Tests: `unit` tests for the command strings `arm.py` sends (fake transport). `hw` test: home, move each joint 10 degrees and back, `?` reports the expected angles.

**Exit (Gate 2):** from WSL, every joint and the gripper move on command, and `capture()` saves a photo from code.

### Phase 3: Aim, pick and place one block (Gate 3, Sat 31 Oct)

1. (changed 30 Oct) No separate calibration step or saved map: every photo calibrates itself from its 6 dots
   (`scene.calibrate`). First, the blocker in todo.md: move the check dots off the middle lines.
2. Put a bright sticker on the gripper tip so SAM can find it ("gripper marker").
3. Settle needs.md B6: precision test (same pose 10 times).
4. `kinematics.py`: **taught-pose grid chosen (2 Nov), no arm maths.** Teach 6-8 points with `scripts/teach.py`
   covering every slot and the whole mess area (the arm only goes inside their hull + 1 cm), plus grip_open,
   grip_closed and camera_clear.
5. `pick(cm)` and `place(cm)`: hover, lower, grip or release, lift.
6. Aim check: after moving to hover, the camera measures the tip against the target and corrects once. Save learned offsets in `calib/offsets.json`, so the arm gets better with practice.

Tests: `unit` tests for kinematics (known poses, grid interpolation exact at the taught points). `hw` test: the tip reaches 5 points within 1.5 cm.

**Exit (Gate 3):** a block picked from the mess and placed in slot 1 within 1.5 cm, 4 times out of 5.

### Phase 4: The full loop (Gate 4, Sun 1 Nov)

1. `loop.py::run()`: demo photo, names, target order, mess photo, positions, plan; for each move: pick, place, photo, measure. Within 1.5 cm: next. Missed: correct and retry, at most 2 retries per move.
2. The run log from section 2.4.
3. `brain.diagnose()` when a retry is needed (a block dropped, knocked, or not found); `loop.py` clamps its fix to safe limits.
4. `scripts/run_mimic.py` starts a full run with one command.

Tests: `unit` test of `loop.py` with a fake arm, fake camera and saved SAM replies, including one forced miss that must trigger a retry and be logged.

**Exit (Gate 4, Sun 21:00):** three blocks rebuilt correctly in 3 runs out of 5, no keyboard during the run. If not, shrink: bigger blocks, bigger slots, or two blocks. Do not keep fighting the hardware.

### Phase 5: Demo quality (Gate 5, Mon 2 Nov, freeze 18:00)

1. `web.py`: one local page. Live camera image with boxes, the plan, the attempt log with PASS/FAIL and error in cm, the cost. FastAPI + a single HTML file. The robot is the hero; the page supports it.
2. Settle needs.md B7, then let `brain.diagnose()` run on Muse Spark Open through Baseten, switchable in `.env`. Add the "Built with Muse Spark Open" notice to the page and README.
3. `scripts/benchmark.py`: 10 runs, prints success rate, average retries, average error, cost per run, time per run.
4. Nice-to-have only if all above is green: voice start with Muse Voice Transcribe.

**Exit (Gate 5):** page works during a real run; benchmark numbers saved; features frozen.

### Phase 6: Video and submission (Gate 6, Tue 3 to Wed 4 Nov)

1. README: pitch, the four-job table, architecture diagram, how to run, benchmark numbers, "built during the event" section, licence notices.
2. Record the 3-minute video from the script in description.md. Real runs only; label anything sped up.
3. Post the best clips on X. Submit by **21:00 UK on Wed 4 Nov**.

---

## 4. Test strategy

| Tier | Marker | Needs | Run with |
|---|---|---|---|
| Unit | none | nothing | `pytest` (default; must always be green) |
| API | `api` | network + credits | `pytest -m api` (on purpose, few calls) |
| Hardware | `hw` | arm and/or camera | `pytest -m hw` |

Rules:

- Unit tests never call the network. They use replies saved in `tests/fixtures/meta_responses/`.
- Every new API reply format we rely on gets saved as a fixture and tested.
- A test that passes against broken code is worse than no test: for the key checks (budget guard, clamping, retry on miss), break the code on purpose once and confirm the test fails.
- Hardware tests start from `home`, move slowly, and end at `home`.

---

## 5. Working agreements

- **Small steps.** One task at a time. Code, test, run, commit, then the next task.
- **Commit after every green step** with a clear message, and add a dated line to BUILD_LOG.md each day.
- **LEARN.md.** After each task, a short plain-English note: what was built, why this way, and how to explain it to a judge in two sentences.
- **Ask Girik first** before: spending more than $2 of API in one session; changing a design decision in this file; sending any new angles to the real arm; installing anything on Windows.
- **Never** print, log or commit the API key. Never use `-contributor` models. Never put faces or private things in photos sent to the API.
- **New code only.** Nothing from before 27 Oct 2026, and never the kit's sample code in the demo.
