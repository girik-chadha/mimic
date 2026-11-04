# Build log

Everything in this repo was written during the Meta Global AI Developer Hackathon (Oct 27 - Nov 4, 2026).

## 28 Oct 2026
- Set up project and Meta Model API access.
- SAM 3.1 finds objects in a still photo by colour and shape (one prompt per object).
- SAM 3.1 tracks objects through a video, one track per prompt.
- Muse Spark 1.3 (muse-spark-1.3) returns parseable JSON from a text task.
- Design decision: move order is computed in Python (exact, testable);
  Spark is used for judgement: understanding the demo and diagnosing failures.

## 29 Oct 2026
- Repo tidied into the planned layout: `mimic/` package, `scripts/spikes/` (day-2 scripts, still working),
  `tests/` (unit / api / hw tiers + fixtures), `pyproject.toml`, `.env.example`. Saved the real 317-frame
  SAM video reply as a test fixture.
- `config.py` + `meta_api.py`: one door to Meta's models with a disk cache, retries on 429/5xx,
  a cost counter, a hard per-run budget, and `--save` to keep real replies as test fixtures.
  19 offline tests; budget, retry, cache and contributor guards each broken on purpose to prove the tests catch it.
- `meta_api` hardening: `use_cache=False` for honest benchmark costs, `reset()` / `cost_so_far()` per run,
  API key replaced by *** in every raised error (and its causes), 60 s request timeout. 24 offline tests.
- `sam.py`: `Box` (with match/track id), `parse_boxes`, `parse_tracks`, `find`. Saved 5 real SAM photo replies
  as fixtures ($0.0125); found photo replies share the video format and "no match" is an empty reply.
  14 offline tests, including truncated replies; boxes checked by eye on desk.jpeg.
- B1 settled: Muse Spark 1.3 accepts images. B10 settled: replies include token counts.
- `brain.name_objects`: Spark names the objects, SAM checks each name finds exactly 1 object, and failed names
  go back to Spark with their match counts (max 2 retries, $0.03 retry cap). On desk.jpeg Spark said
  "black box" (SAM: 0 matches), then "black rectangle" (1 match) on the first retry. 23 offline tests.
- `plan.py`: `plan_moves` sends the i-th target name to slot i; raises on an empty target, a missing or repeated
  name, more names than slots, two targets under 1 cm apart, or two slots at the same point. Extra objects are
  left alone and reported. 13 tests. Rename-prompt example changed to a neutral one ("red cup" -> "red mug").
- `calib.py`: homography from the 4 corner stickers; extra stickers are independent check points (> 0.5 cm off
  raises `CalibrationError` with the numbers). Tested: fitting all 5 points, or leaving one out, hides or
  amplifies errors, so the check sticker is kept out of the fit. 14 tests on a made-up tilted camera.
- `scene.py` + `calib/layout.json`: find stickers and calibrate from one photo; object positions in cm (exactly 1
  SAM box each, must be on the sheet); demo order by nearest slot (tolerance <= half the slot spacing); optional
  block-height correction, off by default. 18 tests with a fake SAM. plan.md 2.1 / 2.3 updated.
- `scripts/plan_from_photos.py`: demo + mess photo in; each photo calibrated on its own (check-sticker error
  printed, > 0.5 cm stops), loud warning if the two maps disagree by > 0.5 cm (camera bumped); Spark names,
  slot order, positions, moves; annotated photos + plan.json in runs/. $0.15 cap per run through the budget
  guard; refuses while layout.json is a placeholder. 8 offline end-to-end tests. Gate 1 waits for real photos.
- Phase 2 software ahead of the arm: `firmware/mimic_arm/` (protocol J/H/?/S; clamps, eased ramps, speed limit,
  grip relaxes after 5 s, 64-char lines, hold when idle; safety logic in `arm_logic.h`, tested with g++ in 3 builds),
  `mimic/arm.py` (fake transport, exact command strings, Python range check), `calib/arm_limits.json` (placeholder +
  limits_checked_by_hand: both must be right before a real move), `scripts/arm_jog.py` (dry run). A test keeps the
  firmware and JSON limits identical. The .ino is not compiled for the ESP32 and nothing was uploaded.

## 30 Oct 2026
- Calibration redesign: the sheet has 6 dots found with one SAM prompt ("small coloured dot"), because SAM cannot
  tell similar sticker colours apart. Every photo calibrates itself: 4 corner dots build the map (roles worked
  out from position, trying 4 rotations), 2 check dots test it (0.5 cm). No saved camera map; the camera may move.
  Labelled photo of the dots saved for checking by eye. 2 real SAM replies saved ($0.005): 6a calibrates
  (right check 0.49 cm), 6b is refused (0.56 cm) with the provisional dot positions. Known limit: mirror images
  are read upside down until the check dots move off the middle lines (todo blocker before Gate 3).

## 2 Nov 2026
- Firmware compiles for the ESP32 (`esp32:esp32:esp32`, core 2.0.12, ESP32Servo 1.2.1) with the Windows
  arduino-cli run from WSL: no errors, no warnings, 22% of flash. Not uploaded.
- Pins set from the kit's sample code (base 16, shoulder 17, elbow 2, grip 4), untested; limits still placeholders.
- Laptop-to-arm link: `scripts/serial_bridge.py` (Windows Python + pyserial) started from WSL by
  `SerialBridgeTransport`; finds the CP210x by USB ID, DTR/RTS low, 2 s boot wait, boot text discarded, every reply
  time-limited, S on close. `scripts/arm_check.py` (read-only first test), `scripts/arm_lock.py` (lock/unlock in both
  files at once). Firmware while locked: no servo attached or written, `?` ends with L. 25 new tests, all without
  hardware. Firmware copied to C:\Users\girik\mimic_firmware and compiled there; nothing uploaded.
- Correction: ESP32Servo 1.2.1 fails under `--warnings all` (unused variable in the library); use default warnings.
- Pick-and-place software against fakes: `kinematics.py` (taught-pose grid, 1/d^2 blend, refuses outside the
  taught hull + 1 cm), `scripts/teach.py`, `motion.py` (planned grip hold checked against max_grip_hold_s before
  anything moves), `loop.py` (photo before every check with the arm at camera_clear, PASS within 1.5 cm,
  corrected retries within +-2 cm, FAIL after 2 retries, run log, $0.15 cap, diagnose hook), `camera.py`
  (iVCam via Windows ffmpeg) and `scripts/run_mimic.py`. B2 decided. 36 new tests with a fake world;
  hold-time check broken on purpose once. Nothing run on hardware, no API calls.
- **First test on the real board passed:** locked firmware uploaded; `scripts/arm_check.py` through the Windows
  serial bridge got `A 90 90 90 90 L` for `?`, `OK` for `S`, and `ERR limits not measured` for `J` and `H`.
  B3 settled (needs.md A10). No servo driven yet.
- `scripts/benchmark.py` (N runs, cache off via `meta_api.cache_enabled`, waits for the blocks to be reset,
  prints success rate, attempts per block, worst error, cost per run); shared setup checks in `loop.load_and_check`.
- README drafted: pitch, the four-job table, safety, honest status, benchmark placeholders (no results claimed),
  related work (Instant Policy, Vosylius and Johns, arXiv 2411.12633).
- First-powered-session tools, all tested against fakes (no hardware, nothing uploaded, arm_limits.json still
  locked): robust dot finding (best 6 of up to 12 dot-sized boxes, stop if two sets fit equally),
  `scripts/first_moves.py` (each direction stops at its first n; proposal = approved minus 5 degrees, never
  arm_limits.json), `scripts/grip_test.py` (2-degree steps, opens between asks, lift test within the grip timer),
  `scripts/repeatability.py` (spread, backlash from alternating approaches, verdict vs the 1.5 cm tolerance, $0.10
  cap), `mimic/session.py` (s at any prompt or Ctrl+C sends S; the bridge sends S on Ctrl+C). 36 new tests;
  equal-fit stop broken on purpose once. needs.md C7: the first-power checklist.
- `brain.diagnose`: one Spark call with the before and after photos and the measured miss; JSON problem, fix and
  plain-English explanation; the prompt says the fix is advice only. `loop.choose_correction` uses Spark's fix
  only if it is within 2 cm and does not point the same way as the miss; otherwise (or on a bad reply, an API
  error, or a nearly spent budget) the numeric correction; total limited to +-2 cm. Logged and printed.
- `scripts/report.py` (one self-contained report.html per run + summary.png for the video),
  `scripts/teach_plan.py` (points to teach: slots, mess centre, mess corners; refuses points out of reach).
  layout.json: arm base just outside the top edge, reach 6-20 cm, mess area; placeholders re-derived to sit
  inside the reach and 4 cm from every dot. 26 new tests; +-2 cm clamp broken on purpose once.
- B11 settled with one real call ($0.0207): Spark takes two photos in one call and answers in valid JSON, but on
  photos with no block it still described "the red block ... 1 cm to the right", repeating the miss from the prompt.
- Spark's fix is now logged and printed only; the numeric correction always decides (`MIMIC_USE_SPARK_FIX=1` to
  use it). New consistency check flags a diagnosis that "contradicts measurement" (log, terminal, report). Reserve
  $0.03, at most 2 diagnoses per run, full-run cap $0.30. Prompt rule "If you cannot see the block in photo 2, answer
  not_found. Do not repeat the measured miss." plus `seen_offset_cm`: second real call ($0.0194) on the no-block
  photos now answers not_found. Consistency check broken on purpose once.

## 3 Nov 2026
- Dot check on a real iVCam frame with the arm in place: SAM found exactly the 6 dots (25-33 px). dot_size_frac max
  raised 0.026 -> 0.030 (the orange dot had 0.2 px of margin; a 19 mm die, ~38 px, is now a candidate the dot finder
  leaves out). With the provisional dot numbers this frame is refused on the right check dot (0.57 cm): a third photo
  putting it at ~35.0 cm, not 34.5. Real reply saved as a fixture ($0.0025); `ivcam_arm.jpg` and `ivcam_sheet.jpg`
  committed (checked by Girik: no people or screens).
- First powered session, steps 1-3 (needs.md C7): the kit's ranges (base 0-180, shoulder 80-180, elbow 30-120, grip
  90/120, home 90; from its sample code, untested) in both `arm_limits.json` and `arm_logic.h`; tests green with the
  sync test; `arm_lock.py unlock`; `test_committed_limits_are_locked` updated on purpose, and the lock-guard tests now
  build their own locked limits so they keep testing the guard. Unlocked firmware compiled (300,937 bytes, servo
  code included) and copied to C:\Users\girik\mimic_firmware\mimic_arm. Not uploaded.
- Grip closed limit 120 -> 140 in both `arm_limits.json` and `arm_logic.h` (kit's own limit 150): a 19 mm die needs
  the jaws closer than 120 allows; the gap per degree is unmeasured. Sync test green; firmware recompiled and
  copied to Windows. Not uploaded.
- Staggered start-up: the firmware switches the servos on one at a time, 400 ms apart (base, shoulder, elbow,
  claw), from `servos_to_attach()` in arm_logic.h (g++ tested; locked: never). g++ test that no 20 ms tick moves a
  joint more than 3 degrees. Recompiled (301,081 bytes), Windows copy refreshed. Not uploaded.
- needs.md C7 revised: post-upload `arm_jog.py status` check, pose the arm by hand first, unplug USB before
  connecting the batteries (so the start-up is staggered), GPIO 2 upload note, write down which way lifts the claw.
- `B` command (firmware, Python, bridge, arm_jog): replies `B <reason> <seconds since start-up>` from the ESP32's reset
  code (power-on, brownout, software reset, watchdog, other); read-only, works while locked. Mapping in arm_logic.h,
  g++ tested in all three builds; the .ino checks the numbers against the ESP32's names with static_assert.
  Recompiled (301,273 bytes), Windows copy refreshed. Not uploaded. needs.md A10b, B12, C7 steps 4 and 8.
- Demo as a video (`scripts/show_demo.py`, `mimic/video.py`, `sam.track`): Windows ffmpeg shrinks it to 10 fps,
  at most 720p, no sound or metadata; warning + typed yes; $0.30 cap checked before anything is sent; last frame
  calibrated and named first; one SAM video call per name; stops on a missing, doubled or lost track; reads the
  last frame where every block was still for 1 s (10 frames) and all 6 dots are found; logs the arrival order.
  meta_api: video calls priced before sending, 300 s timeout, no retry after a timeout. Checked locally: SAM's
  317 frame markers match the clip's 317 frames; the clip shrinks to 106 frames / 548 KB. Broke the lost-track
  stop on purpose: 3 tests failed; restored. No API call made.
- Push mode, step 1 (opt-in): `motion.plan_push` and `push`. The claw closes to `grip_push` (tips just touching
  minus 2 degrees, never the 140 limit; null in pose_grid.json until the grip test), comes down 3 cm behind the
  block, slides it straight towards the slot in steps of at most 1.5 cm at grab height, backs off, lifts, opens.
  A push longer than the grip timer allows is cut short (about 9 cm per push). Refused before moving if another
  block is within 4 cm of the path or the path leaves the taught area.
- Push mode, step 2 (opt-in): `run_mimic.py --push` / `loop.run(push=True)`. Refused before anything opens if
  grip_push is unknown. Furthest slot first; a blocked path lets another block go first, none clear stops the run.
  After every push the camera measures every block: PASS within 1.5 cm, else push again (at most 4 per block);
  two pushes that moved it less than 0.5 cm are FAIL; a knocked mess block carries on, a block knocked out of its
  slot stops the run. `teach_plan.py --push` lists the extra points (5 with the placeholder layout, 1 out of
  reach). Fake pushing world in the tests: 23 cm takes 3 pushes; the without-push run is unchanged.
- Naming prompt: added "The objects are small painted dice, so name each one 'colour die' (for example 'red
  die')." Rename loop and exactly-1-box rule unchanged. The saved naming replies predate the line (lighters on
  desk.jpeg) and are kept, labelled as such; a real dice reply is needs.md B14. No API call.
- Ruler-measured dots in `calib/layout.json` (Girik): dots and sheet edges. New flag `dots_measured: true`;
  `placeholder` stays true (slots, mess area, reach still placeholders). `scene.calibrate`, `show_demo.py` and
  `plan_from_photos.py` now need only the dots (`refuse_unmeasured_dots`); run_mimic, the loop, teach_plan and
  repeatability still need the whole file (`refuse_placeholder`). All three real photos calibrate from their saved
  replies: check dots 6a 0.13 / 0.27, 6b 0.15 / 0.19, ivcam_arm 0.20 / 0.19 cm (limit 0.5). No API call.
- Naming prompts: every example is now a die ("red die" / "blue die"); the dice line is in the rename prompt too,
  whose example is now `{"red die": "dark red die"}` (it was "red cup" -> "red mug", which taught changing the
  object word). `repeatability.py` needs only `dots_measured`; tests: it runs with the committed layout, refuses
  unmeasured dots, and still refuses a locked arm (checked by putting the old check back: 3 tests failed).
  `dice_test.jpg` added to .gitignore. No API call.
- Disarmed start (firmware + Python): at power-on nothing is attached and no signal is sent. `E [b s e g]` switches
  the servos on one at a time, 400 ms apart, each starting at those angles (default 90); `D` stops and switches
  them all off (limp). J and H reply `ERR not enabled` until all four are on; `?` ends with ` D` while disarmed;
  `?`, S, B and D always work; lock and ramp unchanged; `servos_to_attach()` still g++-tested (now from E).
  Python: `arm.enable()` (waits 1.3 s, checks `?`), `arm.disable()`, a restart mid-run is reported with the boot
  reason; `session.confirm_and_enable()` (typed yes) in first_moves, grip_test, repeatability, run_mimic and
  benchmark; `enable` / `disable` in arm_jog and teach; s / Ctrl+C send S then D, errors print the reason then
  S and D, a normal quit says "Hold the arm", waits 2 s and disables. Grip range 40-140 in both limit files
  (`GRIP_RANGE`, `grip_range_deg`; open 90 and closed limit 140 unchanged; the limit must be one end of the
  range). Compiled (301,681 bytes), Windows copy refreshed, not uploaded. needs.md C7 rewritten, B15 (which way the
  claw closes, with the list of everything that assumes a larger angle closes it). Also added the bridge
  pass-through test for `B` that the 3 Nov B commit described but had not actually added.

## 4 Nov 2026

- Claw direction flipped (Girik: a lower angle closes it, open 90). The closed limit is the one setting for the
  direction: `GRIP_CLOSED_LIMIT 20` / `grip_closed_limit_deg` 20, PROVISIONAL (the empty jaws do not touch down to
  40); `GRIP_RANGE` / `grip_range_deg` 10-140. The firmware clamps the grip between the closed limit and 140;
  `range_of("grip")` is 20-140. The 5 s relaxation still returns to 90; max_grip_hold_s stays 5. New
  `grip_hold_deg` (null until found): `grip close` uses it, never the limit. `grip_test.py --start 50`: 2-degree
  steps towards the limit, at the first "holding? y" 3 degrees further closed (never past the limit), then the lift
  check; it takes the grip settings from arm_limits.json, not from an older proposal. g++ tests build both ways
  (limit 20 and 140); breaking the clamp on purpose failed 3 checks in the downward build. Compiled (301,681
  bytes), Windows copy refreshed, not uploaded. B15 moved to needs.md A10c. The "claw fixed" plan is cancelled
  (the claw works).
- Grip closed limit 20 -> 3 (PROVISIONAL; not 0, because 0 is the servo's own end stop and may stall even empty),
  grip range 10-140 -> 0-140, in both limit files (sync test passes). With foam pads the empty jaws are still wider
  than a 19 mm die at 20; Girik is measuring the pad-to-pad gap at 90 and 20. `grip_test.py`: the last step is the
  limit itself, with a warning; no hold down to the limit, or a hold whose extra 3 degrees would reach the limit,
  stops with "no hold found down to the limit" (no hold or lift at the limit). `arm_limits.json` refuses a
  `grip_hold_deg` equal to the limit. `arm_jog` and `teach.py` warn on a grip move onto the limit. Relaxation still
  returns to 90. g++ tests pass in all four builds. Compiled (301,681 bytes), Windows copy refreshed, not uploaded.
- First real first_moves session (Girik): stopped after the base, which wrote nothing. Base: minus stopped at 20
  (approved 15), plus stopped at 40 (approved 35); proposed base range 80-120 (needs.md B4). `first_moves.py` now
  saves the proposal after every finished joint, always merges into the one already there (a joint it runs
  replaces its entry, the rest are kept and listed), has `--joints` and `--base-minus/--base-plus` (the base
  result typed in, saved before anything moves). Until all three joints are in, the proposal is INCOMPLETE with
  limits_checked_by_hand false; `grip_test.py` refuses it with what is missing. `session.ask` now throws away
  anything typed before a question appears (termios flush on a real terminal), so an early answer typed while the
  arm moves can no longer answer the next question.
- Elbow servo dead and unplugged, forearm taped (Girik): the claw tip is at table height at shoulder 173 on a ring
  around the base axis, and grip 23 held the die through a 20-degree lift, by hand. New `scripts/pick_once.py`:
  open, base to the pick angle, shoulder to pick - approach, down in 2-degree steps, close to 23, lift 20, swing,
  down in 2-degree steps, open, lift away, then "Hold the arm" and disable (no H). Everything planned and checked
  before the enable question: ranges from arm_limits.json, the shoulder never past `--stop-at` 175, and the
  claw-closed time (each move, its 0.1 s wait and 0.05 s per command; shoulder at 25 ms per degree, base and claw
  at the limits' 10) must fit 4.5 s: 3.87 s for a 20-degree swing, about 80 degrees at most. `--pick-only`,
  `--measure-latency` (20 status calls, suggests `--command-s`), `--dry-run`; a warning for base angles outside
  the approved 80-120. The enable question now starts "Pose the shoulder near 90 by hand first". Breaking the
  4.5 s refusal on purpose failed 3 tests; restored. The elbow_fixed plan is dropped: IO2 sends 90 into an empty
  socket, which is harmless (needs.md A10d).
- Ring mode, part 1. `mimic/ring.py`: the claw tip at table height is on a circle around the base axis; the fit
  takes (base angle, tip sheet cm) points and finds centre, radius, base scale and offset (for a fixed scale it is a
  linear least-squares fit). Scale is 1 unless another (0.85-1.15, needs 4+ points) fits clearly better: building it
  showed that on a 40-degree arc scale and radius trade off, so 3 points hide a short servo as a wrong radius and
  0.2 cm of camera noise could push a free scale to its limit. A refused fit names the point that spoils it (leave
  one out). `base_angle_for` refuses points more than 1.0 cm off the ring. `calib/ring.json` (placeholder,
  camera_clear 90/90, pick_once's numbers). `scripts/ring_calibrate.py`: 5 base angles; at each the claw goes down
  onto the table, a self-calibrated photo, SAM finds the "gripper marker"; fit, show the errors, save on a yes.
  pick_once's planning moved to `mimic/pickdrop.py` (with plan_touch / plan_lift); pick_once unchanged.
- Ring mode, part 2: `mimic/ring_loop.py`, `run_mimic.py --ring`. Needs only the measured dots and a fitted
  ring.json with slots. Demo photo: dice to their nearest ring slot -> target order. Mess photo: every pick and
  drop planned and checked before any move (4.5 s grip rule, base angle on the ring within 1.0 cm, dice 4 cm
  apart and 4 cm from other slots). Each carry uses pick_once's moves; park at camera_clear, photo, measure: the
  angular miss shifts the next drop (total within +-2 degrees), at most 2 retries picked up from where it landed;
  the radial miss is logged, not corrected. Building the tests showed slots need 5.5 cm between them (4 cm + the
  1.5 cm a passed die may be off), or a correctly placed die would block the next slot; the arc check uses that.
  Layouts that do not fit are refused with the numbers. Tested with a fake ring world (pick at table height when
  the claw closes, drop with a chosen miss).
- calib/ring.json: grip_hold 20 (found by hand), command_s 0.01 (measured with pick_once.py --measure-latency: 3.5 ms
  mean, 4.7 ms slowest), stop_at null so the shoulder stop is 3 degrees past shoulder_table whatever it is
  (`Ring.shoulder_stop`). shoulder_table stays 173: the new angle was not given ("[MY NEW ANGLE]" came through);
  `ring_calibrate.py --table-shoulder N` uses N and saves it with the fit. Tests check that ring_calibrate.py and
  ring mode take the table angle, grip, bridge time and stop from the ring, not from defaults.
- Start pose: `calib/start_pose.json` (base 90, shoulder S0, elbow 90, grip 90; S0 was not given, so the shoulder
  is PROVISIONAL at 90). `arm.load_start_pose()` (missing file: 90 everywhere with a warning; a wrong file is
  refused). Every enable sends `E` with it (`Arm.enable` with no angles, `confirm_and_enable`, whose question now
  says "pose the arm like the start pose: upper arm vertical, forearm horizontal, claw open"), so pick_once,
  ring_calibrate, run_mimic --ring, first_moves and grip_test all start there; pick_once plans its first move from
  it. `arm_jog` and `teach` `home` are a slow J to it (no firmware change). Every script's normal end ramps back to
  it (30 ms per degree, at least 1 s) before "Hold the arm", 2 s and D; stops never ramp. ring.json shoulder_table
  180 (approach 10, so from 170).
- Real iVCam frame (Girik): "gripper marker", "yellow square", "yellow sticker", "yellow foam" found nothing;
  "small yellow rectangle" found the claw marker (743, 182) and the control board's yellow pin sockets (578, 160).
  `marker_name` in calib/ring.json (default "small yellow rectangle"), `ring_calibrate.py --marker-name` (saved with
  the fit). Only boxes whose centre is on the calibrated sheet count (a lone box off the sheet too): exactly one must
  remain, or the point is refused listing every box and where it fell. Each point's photo is saved with every box
  drawn and the chosen one marked. Test with the two real boxes: the marker is chosen, the pins dropped.
- Camera brightness (Girik's real iVCam test: 2 frames 247.5, 15 frames 170.5; ring_calibrate photos 247.6).
  camera.capture already read 15 frames and kept the last, so the blown-out calibration photos were not a short
  grab in the code. Now every photo's mean brightness is printed; outside 40-235 it is retaken once after 1 s, then
  refused with the brightness and a hint to lock iVCam's exposure (ring_calibrate skips that point). Frames per grab:
  `camera_frames` in ring.json (15), `--camera-frames` on ring_calibrate.py and run_mimic.py. Tests with fake frames
  (normal, too bright then normal, too bright twice, too dark twice, the 40 and 235 edges). run_mimic.py uses
  camera.capture in both modes; show_demo.py takes no photos (it reads a video file).
- Camera name: `camera_name` in ring.json (default "e2eSoft iVCam"), `MIMIC_CAMERA`, `--camera-name` (on
  ring_calibrate.py and run_mimic.py; that order of priority: option, environment, ring.json, default). Real grabs
  (4 Nov): 2 frames give iVCam 247.5 and Camo 0.0, 15 frames give 170.5 and 152.7, so 15 frames stay the default.
  Every photo prints camera, size and brightness; refusals name the camera. The 1280x720 minimum now works either
  way round (a portrait frame was refused before). Nothing else assumed iVCam or 1280x720. A unit-test conftest
  clears MIMIC_CAMERA so the shell cannot change the tests.
- `scripts/ring_fit_manual.py`: the ring from repeated `--point BASE X Y` (a base angle and the die's centre in
  sheet cm, by ruler). No arm, camera or SAM. Same fit as ring_calibrate.py (mimic/ring.py, 0.5 cm check and
  leave-one-out naming), at least 5 points; saves ring.json with `source: "manual"`, placeholder false, slots and
  camera_clear kept. ring_calibrate.py now saves `source: "camera"`; the fit printout is shared (`ring.describe`).
- Start pose shoulder 90 -> 150 (Girik measured 4 Nov on fresh batteries: the upper arm is vertical at 150); the
  PROVISIONAL wording is gone. Tests updated: every enable now sends `E 90 150 90 90`, every end ramps to
  shoulder 150, pick_once's first move goes from 150. The first_moves and grip_test tests use made-up shoulder
  ranges (85-95, 80-120) that 150 is outside of, so they start at 90; a new test shows a start pose outside the
  shoulder range refuses to enable. A first_moves proposal (45 degrees around home 90) would cap the shoulder at
  135, below the start pose: noted in needs.md.
- Simulation mode, part 1 (`run_mimic.py --ring --sim --photo ...`, `mimic/sim.py`, `mimic/sim_video.py`,
  `mimic/marks.py`): one real photo goes through the real perception (SAM dots, Spark naming, SAM positions, $0.30
  cap) and ring mode's own planner and checks; the moves go to a simulated arm on a MADE-UP ring (centre (18.5, 27)
  cm, radius 20 cm, slots at base 22, 38, 54; pick numbers from ring.json). The camera-check loop was moved out of
  `ring_loop.run` into `ring_loop.carry` so the real run and the simulation run the same code (its 15 tests pass
  unchanged). Outputs in `runs/<time>_simulation/`, every name starting `simulation_`: the move list (printed as
  JSON, every line the arm would be sent), summary, attempts, the photo and its dots (JPEG comment "SIMULATION"),
  and a 20 fps video (2 s title card, SIMULATED EXECUTION banner on every frame, INJECTED TEST MISS while a
  `--sim-miss-deg` miss is corrected, "SIMULATION" in the mp4's title and comment). Real run on dice_sheet.jpg:
  $0.0156, Spark named red, black and blue die with 1 SAM box each (B14 settled), 3 of 3 placed; the injected-miss
  rerun came from the cache ($0) and showed RETRY then PASS at 0.7 cm.
- Simulation mode, part 2: `--instruction "stack black, blue, red"` piles the dice on slot 1, black at the bottom
  (simulation only: `run_mimic.py` refuses a stack without `--sim`; in simulation_moves.json every line of a carry onto
  the stack is marked `"sim_only": true`, with the reason). `ring_loop.carry(stack=True)` lets the dice already on the
  stack sit at the slot; the real run never sets it. The simulated arm keeps stack levels and picks the top die. The
  video draws a stack with each die offset by its level, with a red "stacking is simulation only" note on every
  frame. `scripts/report.py` now reads ring-mode attempts (miss along the ring and the radius) and, in a simulation
  folder, writes `simulation_report.html` (sticky SIMULATED EXECUTION banner, the real photo next to the simulated
  start, every simulated camera check) and `simulation_summary.png` (banner strip, PNG text "SIMULATION");
  `run_mimic.py --ring --sim` runs it at the end. Cached runs on dice_sheet.jpg ($0): row 3/3, injected miss RETRY
  then PASS at 0.7 cm, stack 3/3.
- Demo page, step 1: `web/app.py` (standard library only, 127.0.0.1:8000) and `web/index.html` (blueprint look,
  Archivo and JetBrains Mono bundled with their OFL licences, nothing loaded from the internet). Upload or the sample
  photo, goal, use cache (off by default), Run; the simulation runs in a background thread and every line it prints
  is streamed to the page as it happens (Server-Sent Events); lines about the arm are tagged "[simulated]". The goal
  is read by code (`sim.parse_goal`): "put black on blue, then red on black" (a stack built on slot 1 from the bottom
  die up), "row ...", "stack ...", or empty for the photo order; anything else is refused with examples before any
  call. Photos must really be JPEG or PNG (first bytes and a decode). The run now prints the check-dot errors and
  Spark's names, and the video draws at 15 fps with frame-progress lines. A cached run of "put black on blue, then
  red on black" through the server: 3 of 3 stacked, $0.
