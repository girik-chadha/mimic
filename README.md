# Mimic: show a robot once

**Show a robot a task once. It reads the goal from your photo or a few typed words, plans every move with plain code, checks each one, and measures what it did with a camera.**

Built during the event by Girik Chadha, with help from an AI coding assistant (Claude Code). The product uses Meta's SAM 3.1 and Muse Spark 1.3. Execution in the demo video is an expected simulation, except the clips of the real arm.

Built for the Meta Global AI Developer Hackathon (27 Oct - 4 Nov 2026). All code in this repo was written during the event; see [BUILD_LOG.md](BUILD_LOG.md).

## The hardware, honestly

- **The arm's elbow servo failed on 4 Nov.** It is unplugged and the forearm is fixed in place with tape. With the elbow fixed, the claw tip can only meet the table on **one ring of reach** around the base: the base angle picks a point on that ring, and one shoulder angle puts the claw at table height. Everything that moves the real arm now works on that ring ("ring mode").
- **Pick-and-carry on the real arm is proven in one scripted run and one manual stacking run.** The full camera-checked loop (`run_mimic.py --ring`) has not run on the real arm yet: the ring has not been calibrated with the camera.
- **Simulation is always labelled.** The demo page and `run_mimic.py --ring --sim` run the real perception (SAM and Spark on a real photo) and the real planner and safety checks, then send the moves to a simulated arm on a made-up ring. Every video frame carries a "SIMULATED EXECUTION" banner, every output file starts with `simulation_`, and the page calls it the "Expected simulation". Stacking is simulation-only: with the elbow fixed, the claw cannot let go of a die on top of another at a controlled height.

## What it does

1. You arrange a few coloured dice, or type the goal ("put black on blue, then red on black", "row black, blue, red").
2. A photo of the table goes to SAM 3.1 and Muse Spark 1.3: the photo calibrates itself from six dots on the sheet, Spark names each die, and SAM measures where each one is, in centimetres.
3. Plain Python plans every carry and checks it before anything moves: the joint limits, the shoulder stop, the spacing between dice, and how long the claw stays closed (the firmware opens a claw held for 5 s, so the plan must stay under 4.5 s).
4. The arm carries each die. After every carry the camera measures where it landed; a miss of more than 1.5 cm is corrected and retried (in the simulation, the camera check reads the simulated world).

The dice are only the test. The idea is the loop: read the goal once, plan with exact code, act, measure, correct.

## How it works

| Job | Done by | What it does |
|---|---|---|
| **See** | SAM 3.1 (Meta) | Finds each object in a photo from a short name like "blue die", and gives its exact box. Also finds the six calibration dots. |
| **Understand** | Muse Spark 1.3 (Meta) | Looks at the photo and names the objects in words SAM can search for. |
| **Plan** | Plain Python | Reads the typed goal, decides which die goes where and in what order, plans every joint angle on the ring, and checks every move against the limits. |
| **Act and check** | Arm + camera + Python | Moves the arm through a serial bridge, takes a new photo, measures the placement in centimetres, retries with a correction if it missed. |

Design rule: **code for exact logic, AI for judgement.**

- **Every photo calibrates itself.** Six dots on the sheet are found with one SAM prompt ("small coloured dot"). Four corner dots give the photo's own pixel-to-centimetre map, and two check dots must land within 0.5 cm, or the photo is refused. On the sample photo they land 0.19 and 0.14 cm off.
- **Spark and SAM check each other.** Spark names each die ("blue die"), and a name only counts if SAM finds exactly one object with it. Otherwise Spark is told the match count and renames it.
- **The goal is read by code, not AI.** Simple goals only; anything else is refused with examples before any API call.

## The demo page

`python web/app.py`, then open http://127.0.0.1:8000. Upload a photo (JPEG or PNG only) or use the sample, type a goal, and Run. The photo is sent to Meta's SAM 3.1 and Muse Spark 1.3, so keep faces out of it. The server runs the real perception and planner and streams every line to the page as it happens, with the cost against the $0.30 cap; lines about the arm are tagged "simulated". Then the "Expected simulation" video plays, with tabs for what SAM saw, what Spark named, and the plan for every carry. At the end, "Download arm program" gives a script for the real arm: `--dry-run` replays the simulation on a fake arm; on the real arm it works out every angle again from the real ring calibration (or, for a stack, from `calib/stack.json`, measured by hand) and refuses until those exist, and refuses any base angle outside 80-120.

## Safety

- The arm's limits live in two places, Python and the ESP32 firmware, and a test keeps them identical. Python refuses an out-of-range move; the firmware clamps anyway.
- The firmware in this repo starts with the servos off. Every script asks for a typed "yes" before switching them on at a known start pose; `s` or Ctrl+C sends stop, then switches the servos off.
- Every move is speed-limited and ramped, the shoulder is never sent past its stop, and the planned time the claw stays closed is checked before moving.
- Every API call goes through one function, with a cache, a cost counter and a hard budget per run ($0.30). Meta's own spend limit only notifies.

## Status (4 Nov 2026)

- **Tested against the real Meta API:** SAM finds the dice and the calibration dots; Spark names the dice ("red die", "black die", "blue die", one SAM box each); a full run through the demo page with the cache off cost $0.023.
- **On the real arm:** the serial link and the joint limits, measured and checked by hand; one scripted pick-and-carry run and one manual stacking run, on the ring, with the elbow fixed.
- **Simulation only (always labelled):** the full task on a made-up ring, with the real perception, planner and checks; stacking.
- **Not done yet:** calibrating the real ring with the camera, the camera-checked loop on the real arm, the benchmark below, Muse Spark Open.

## Benchmark

**No results yet.** These would come from `scripts/benchmark.py` runs on the real arm.

| Measure | Result |
|---|---|
| Runs | _to be measured_ |
| Success rate (all dice placed within 1.5 cm) | _to be measured_ |
| Average attempts per die | _to be measured_ |
| Worst final placement error | _to be measured_ |
| Cost per run (cache off) | _to be measured_ |

## Running it

```
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env          # add META_API_KEY
.venv/bin/pytest              # unit tests: no network, no hardware
```

| Command | What it does |
|---|---|
| `python web/app.py` | The demo page on http://127.0.0.1:8000 (simulated arm; real SAM and Spark calls). |
| `python scripts/run_mimic.py --ring --sim --photo tests/fixtures/photos/dice_sheet.jpg` | The same simulation from the terminal; add `--instruction "row black, blue, red"`. |
| `python scripts/arm_check.py` | Read-only check that the arm answers. |
| `python scripts/pick_once.py --base-pick 100 --base-drop 120 --dry-run` | One pick and drop on the ring (fake arm with `--dry-run`). |
| `python scripts/ring_calibrate.py` | Fit the ring the claw reaches, from 5 camera-measured points. |
| `python scripts/run_mimic.py --ring` | The full camera-checked run on the real arm (needs a calibrated ring). |

Hardware: KEYESTUDIO ESP32 4-DOF arm kit (elbow servo failed, forearm fixed), a phone as the camera (iVCam or Camo on Windows, read with ffmpeg), WSL2 on a Windows 11 ARM laptop. Setup details are in [docs/needs.md](docs/needs.md).

## Related work

Instant Policy (Vosylius and Johns, arXiv [2411.12633](https://arxiv.org/abs/2411.12633)) treats imitation as in-context learning: given one or two demonstrations, a graph-diffusion model trained on simulated pseudo-demonstrations generates the robot's actions for a new task, without further training. Mimic is much narrower. It does not learn a skill or a policy. It reads only the goal (which die ends up where), plans the moves with plain code, and closes the loop by measuring every placement with the camera and correcting misses.

## Built with

- Meta SAM 3.1 and Muse Spark 1.3, through the Meta Model API.
- Open-source libraries, used as-is: OpenAI Python SDK, NumPy, OpenCV, python-dotenv, pyserial, pytest; ESP32 Arduino core 2.0.12 and ESP32Servo 1.2.1 for the firmware; FFmpeg (run as a separate program) for the camera and the simulation video; Python's standard-library web server for the demo page.
- Fonts on the demo page: Archivo, Instrument Serif and JetBrains Mono, under the SIL Open Font License 1.1 (licences in `web/fonts/`).
- KEYESTUDIO ESP32 4-DOF arm kit hardware. The servo pin numbers were read from the kit's sample code; none of the kit's sample code is used.
- An AI coding assistant (Claude Code) helped write the code and docs; see the first lines of this file.

## License

The code in this repository is under the MIT License: see [LICENSE](LICENSE). The bundled fonts in `web/fonts/` keep their own SIL Open Font License 1.1.
