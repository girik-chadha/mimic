# Mimic: what is left (as of Wed 28 Oct 2026, night)

Tick items as they finish (`[x]`) and add the date. Details for every item are in [plan.md](plan.md); facts and open questions in [needs.md](needs.md).

## Done (Gate 0)

- [x] Meta API key works; models listed (28 Oct)
- [x] SAM 3.1 finds three objects in a photo, one prompt each (28 Oct)
- [x] SAM 3.1 tracks three objects through a 317-frame video (28 Oct)
- [x] Muse Spark 1.3 returns parseable JSON (28 Oct)
- [x] Public repo github.com/girik-chadha/mimic, first commit (28 Oct)
- [x] Arm kit and NiMH AA batteries ordered (28 Oct)

## Critical path

The arm arrives Fri. Everything that does not need the arm must be finished by Thu night, so Fri evening and all of Sat go to hardware. The riskiest unknowns are physical: precision (B6) and how WSL talks to the arm (B3). Settle them as early as possible.

## Thu 29 Oct: Phase 1, see and plan (Claude Code + you)

- [x] 1. Tidy the repo into the layout in plan.md section 1 (29 Oct)
- [x] 2. `config.py` + `meta_api.py` (cache, cost counter, budget guard) (29 Oct)
- [x] 3. `sam.py` with unit tests on saved real replies (29 Oct)
- [x] 4. Settle B1: can Spark see images? `brain.name_objects()` (29 Oct: yes; names checked by SAM, renamed by Spark if needed)
- [x] 5. `plan.py` with unit tests (29 Oct)
- [x] 6. `calib.py` with unit tests (29 Oct)
- [x] 7. `scene.py` (29 Oct)
- [x] 8. `scripts/plan_from_photos.py` (29 Oct); **Gate 1** on two real photos still to do (waits for photos + spend approval)
- [x] 8b. Calibration redesign: 6 dots, one SAM prompt, every photo calibrates itself (roles by position, 4 rotations, 2 check dots); real replies on dots_check_6a/6b saved as fixtures (30 Oct)

### Thu, physical jobs (you, in between)

- [ ] Make the work area: A3 sheet taped down, 6 dots (4 corners + 2 check dots), slots at least 4 cm from every dot. Measure every dot's position in cm from the bottom-left one with a ruler and write them, the slot centres and the sheet edges into `calib/layout.json` (provisional numbers now)
- [ ] Draw 3 numbered target slots on one side; leave the other side as the mess area
- [ ] Take two photos (the camera may move between them; all 6 dots visible; objects at least 4 cm from every dot): `gate1_demo.jpg` (objects in your chosen order in the slots) and `gate1_mess.jpg` (scattered in the mess area)
- [ ] Decide the camera (needs.md B2): cheap USB webcam, or a phone webcam app that works on Windows
- [x] Evening: install Arduino CLI or IDE 2, the ESP32 core, `ESP32Servo`, and the CP210x driver on Windows (B5; compile + upload work from PowerShell, by 2 Nov)
- [ ] Post a first clip or photo on X (SAM boxes on your desk)

## Fri 30 Oct: Phase 2, arm bring-up

- [ ] Assemble the arm; centre the servos during assembly
- [ ] Measure pins, joint ranges, link lengths, gripper opening; write them into needs.md B4
- [ ] Buy blocks that fit the gripper with ~1 cm to spare (solid, clearly different colours, not white)
- [x] Firmware with the command protocol over USB serial (code + g++ logic tests 29 Oct; compiled 2 Nov; locked firmware uploaded and answering 2 Nov)
- [x] B3 chosen: Windows serial bridge (`scripts/serial_bridge.py`, `SerialBridgeTransport`), tested without hardware (2 Nov)
- [x] Arm link: locked firmware uploaded, `scripts/arm_check.py` passed on the real board (`?`, `S` OK; `J`, `H` refused); B3 settled (2 Nov, needs.md A10)
- [ ] Fill `calib/arm_limits.json` **and** `firmware/mimic_arm/arm_logic.h` with conservative real numbers, jog gently, then set placeholder false / checked by hand true
- [ ] `camera.py`, camera mounted; **Gate 2**

## Sat 31 Oct: Phase 3, aim and one pick

- [ ] **BLOCKER before Gate 3 (Girik):** move both check dots to the sheet edges, off the middle lines (about
  (34.5, 18) and (3, 8) cm), cover the old ones, and retake the dot photos. Today both check dots are on the
  middle line, so a mirror-image photo would be read upside down. **Then Claude:** add the mirror orders to
  `scene.assign_roles`, an ambiguity stop when a mirror and a non-mirror order both fit, and a mirror test.
- [ ] Keep every block at least 4 cm from every dot (a covered dot stops the run)
- [ ] Gripper tip marker
- [x] First-powered-session tools, tested with fakes (2 Nov): `first_moves.py`, `grip_test.py`, `repeatability.py`, robust dot finding, s / Ctrl+C stop everywhere
- [x] First powered session steps 1-3 (3 Nov): kit ranges in both files, unlocked, firmware compiled and copied to Windows (not uploaded)
- [x] `B` command: why the board last started and how long ago (`B brownout 42`), `arm.boot_reason()`, `boot` in `arm_jog.py`; g++ tested, compiled, copied to Windows (3 Nov, not uploaded)
- [x] Push mode, step 1: `motion.plan_push` (claw at grip_push, straight push in 1.5 cm steps, cut by the grip timer, 4 cm clear of other blocks), tested with fakes (3 Nov)
- [x] Push mode, step 2: `run_mimic.py --push` (opt-in; furthest slot first, every block measured after every push, knocks logged or stop), `teach_plan.py --push` extra points; tested with a fake pushing world (3 Nov)
- [ ] `grip_push`: find it in the grip test (needs.md C7 step 11); push mode refuses to run until then
- [x] Disarmed start: `E` (enable, one servo every 400 ms, starting angles) and `D` (disable, limp) in the firmware; `arm.enable()` / `arm.disable()`; every arm script asks before enabling; s / Ctrl+C / errors send S then D; quit disables. Grip range 40-140. g++ and fakes tested, compiled, copied to Windows (3 Nov, not uploaded)
- [x] Claw direction (4 Nov, Girik): a lower angle closes it. Closed limit 20 (provisional) is the one setting for the direction; range 10-140; `grip_hold_deg` via `grip_test.py --start 50` (+3 degrees closed); everything that assumed a larger angle closes it flipped (needs.md A10c). Compiled, copied to Windows, not uploaded
- [x] Closed limit 3 (provisional; not 0, the servo's end stop), grip range 0-140; grip_test tests the limit step with a warning and stops with "no hold found down to the limit"; grip_hold may not equal the limit (4 Nov, not uploaded)
- [ ] Girik: pad-to-pad gap at 90 and 20, to predict whether 3 is enough; after the upload, step the empty claw down to 3 and raise the limit if the pads touch or the servo strains (needs.md C7 step 9)
- [x] Base range measured (4 Nov, Girik): approved -15 / +35, proposed 80-120. `first_moves.py` now saves after every joint, merges, `--joints`, `--base-minus/--base-plus`; typed-ahead answers are thrown away (4 Nov)
- [ ] Resume: `python scripts/first_moves.py --joints shoulder elbow --base-minus 15 --base-plus 35` (the elbow is now dead: answer n, or `--joints shoulder`)
- [x] Elbow dead and unplugged, forearm taped: table height at shoulder 173 on a ring; grip 23 holds the die through a 20-degree lift (4 Nov, by hand). `scripts/pick_once.py` written and tested with fakes (4 Nov)
- [ ] `pick_once.py --measure-latency`, then `--pick-only`, then a full pick and drop (needs.md C7 step 12b)
- [x] Ring mode, part 1: `mimic/ring.py` (fit from base angles and tip positions, base_angle_for, 1 cm off-ring limit), `calib/ring.json` (placeholder, camera_clear 90/90), `scripts/ring_calibrate.py` (5 points); pick_once's moves moved to `mimic/pickdrop.py` (4 Nov)
- [x] ring.json: grip_hold 20 (by hand), command_s 0.01 (measured 3.5 / 4.7 ms), stop_at null = table + 3; `ring_calibrate.py --table-shoulder` (4 Nov)
- [x] `calib/start_pose.json` (base 90, shoulder 150, elbow 90, grip 90): every enable starts there, every script ramps back there before disabling, `arm_jog`/`teach` home is a slow J to it; ring.json table 180 (4 Nov)
- [x] Start pose shoulder 150: the upper arm is vertical there (Girik, 4 Nov, fresh batteries)
- [x] `scripts/ring_fit_manual.py`: the ring from 5+ ruler-measured points (same fit and checks), saved with source "manual" (4 Nov)
- [ ] Girik: `python scripts/ring_calibrate.py` on the real arm, or `ring_fit_manual.py` from a ruler (needs.md C7 step 12a), then set `slots_deg` in `calib/ring.json`
- [x] Ring mode, part 2: `mimic/ring_loop.py`, `run_mimic.py --ring` (slots on the ring, 5.5 cm apart; every move checked first; angular miss corrected up to 2 degrees, 2 retries); tested with a fake ring world (4 Nov)
- [ ] After calibrating and setting `slots_deg`: a first `run_mimic.py --ring` with 2 or 3 dice (needs.md C7 step 12c)
- [ ] Decide the layout for the ring: slots and the mess area must lie on it (needs.md A10d)
- [ ] First powered session (needs.md C7) from step 3's upload: ranges with `first_moves.py`, copy into both files, upload; `grip_test.py`
- [ ] Precision test B6: `repeatability.py`
- [x] Kinematics: taught-pose grid (`kinematics.py`, `calib/pose_grid.json`, `scripts/teach.py`), tested with fakes (2 Nov)
- [x] `pick` / `place` (`motion.py`), planned grip hold checked against max_grip_hold_s, tested with fakes (2 Nov)
- [ ] Teach the real grid: 6-8 points covering every slot and the whole mess area, plus grip and camera_clear (needs.md C6)
- [ ] **Gate 3** on the real arm

## Sun 1 Nov: Phase 4, the full loop

- [x] `loop.py`, run log, retries (with a hook for `diagnose()`), `camera.py` (iVCam via ffmpeg), `scripts/run_mimic.py`: built and tested with fakes (2 Nov)
- [ ] B2 camera test on the real camera: ten sharp overhead photos in a row
- [x] `brain.diagnose()` in `choose_correction` (Spark's fix as advice, checked by code), `scripts/report.py`, `scripts/teach_plan.py`; tested with fakes (2 Nov)
- [x] B11: one real two-photo Spark call: accepted, $0.0207, but it described a block that was not there (needs.md A5c) (2 Nov)
- [x] Decided (2 Nov): Spark's fix logged only (`MIMIC_USE_SPARK_FIX=1` to use it), consistency check, at most 2 diagnoses per run, $0.30 full-run cap; prompt fixed (not_found on the no-block photos)
- [ ] Test `diagnose` on real before/after photos with a block in them (Girik, 3 Nov)
- [x] Dot check on a real iVCam frame with the arm in place: 6 dots found; dot_size_frac max 0.030 (3 Nov)
- [x] Ruler-measure the dots (3 Nov, Girik): in `layout.json` with `dots_measured: true`; all three real photos now calibrate (check dots 0.13-0.27 cm off, needs.md A2b)
- [ ] Measure the slot centres (and redraw them if needed) so `placeholder` can go false together with the mess area and reach
- [x] Demo as a video: `scripts/show_demo.py` (shrink to 10 fps / 720p, warning + typed yes, $0.30 cap, settled frame, arrival order), `mimic/video.py`, `sam.track`; tested on the saved 317-frame reply and fakes (3 Nov)
- [ ] First real `show_demo.py` run on an 8-10 s demo video (needs.md B13; after the dots are measured)
- [x] Naming prompt: one line of scene context, the objects are small painted dice named "colour die" (3 Nov; no API call)
- [ ] Record Spark's real naming reply on a photo of the dice (needs.md B14)
- [ ] Measure the arm's base position, its reach, and the mess area into `calib/layout.json`
- [ ] **Gate 4 at 21:00**: 3 blocks, 3 runs out of 5. Otherwise shrink the task.

## Mon 2 Nov: Phase 5, demo quality (freeze 18:00)

- [ ] Web page; Spark Open on Baseten (B7); benchmark 10 runs

## Tue 3 to Wed 4 Nov: Phase 6

- [ ] README, video, X posts; **submit by 21:00 UK Wed 4 Nov**

## Rules for every day

- Commit after every green step; one dated line in BUILD_LOG.md per day.
- Keep faces and private things out of every photo sent to the API.
- Watch the cost counter. The spend limit on the Meta dashboard only notifies; it does not stop charges.
