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
- [ ] 5. `plan.py` with unit tests
- [ ] 6. `calib.py` with unit tests
- [ ] 7. `scene.py`
- [ ] 8. `scripts/plan_from_photos.py`, then **Gate 1** on two real photos

### Thu, physical jobs (you, in between)

- [ ] Make the work area: A3 sheet taped down, 4 small coloured stickers at the corners, measure their positions in cm from the bottom-left one
- [ ] Draw 3 numbered target slots on one side; leave the other side as the mess area
- [ ] Take two photos from the same fixed spot: `demo.jpg` (objects in your chosen order in the slots) and `mess.jpg` (scattered in the mess area)
- [ ] Decide the camera (needs.md B2): cheap USB webcam, or a phone webcam app that works on Windows
- [ ] Evening: install Arduino CLI or IDE 2, the ESP32 core, `ESP32Servo`, and the CP210x driver on Windows (B5)
- [ ] Post a first clip or photo on X (SAM boxes on your desk)

## Fri 30 Oct: Phase 2, arm bring-up

- [ ] Assemble the arm; centre the servos during assembly
- [ ] Measure pins, joint ranges, link lengths, gripper opening; write them into needs.md B4
- [ ] Buy blocks that fit the gripper with ~1 cm to spare (solid, clearly different colours, not white)
- [ ] Firmware with the command protocol over USB serial
- [ ] Settle B3 (Wi-Fi hotspot vs Windows bridge), then `arm.py` + `arm_jog.py`
- [ ] `camera.py`, camera mounted; **Gate 2**

## Sat 31 Oct: Phase 3, aim and one pick

- [ ] Calibration from the 4 stickers; gripper tip sticker
- [ ] Precision test B6
- [ ] Kinematics (maths, or taught-pose grid by midday if the maths is off)
- [ ] `pick` / `place` with the aim check and learned offsets; **Gate 3**

## Sun 1 Nov: Phase 4, the full loop

- [ ] `loop.py`, run log, retries, `diagnose()`
- [ ] **Gate 4 at 21:00**: 3 blocks, 3 runs out of 5. Otherwise shrink the task.

## Mon 2 Nov: Phase 5, demo quality (freeze 18:00)

- [ ] Web page; Spark Open on Baseten (B7); benchmark 10 runs

## Tue 3 to Wed 4 Nov: Phase 6

- [ ] README, video, X posts; **submit by 21:00 UK Wed 4 Nov**

## Rules for every day

- Commit after every green step; one dated line in BUILD_LOG.md per day.
- Keep faces and private things out of every photo sent to the API.
- Watch the cost counter. The spend limit on the Meta dashboard only notifies; it does not stop charges.
