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
