# Mimic: show a robot once

**Show a robot a task once. It reads the goal from your demonstration, does it, and then uses a camera to measure and fix its own mistakes.**

Built for the Meta Global AI Developer Hackathon (27 Oct - 4 Nov 2026). All code in this repo was written during the event; see [BUILD_LOG.md](BUILD_LOG.md).

## What it does

1. You put a few coloured blocks into numbered slots on a sheet, in the order you choose. Mimic takes a photo.
2. You scatter the blocks. Mimic takes another photo.
3. A small robot arm rebuilds your arrangement, one block at a time. After every placement the camera measures where the block landed, in centimetres. A miss of more than 1.5 cm is corrected and retried.

The blocks are only the test. The idea is the loop: read the goal from one demonstration, act, measure, correct.

## How it works

| Job | Done by | What it does |
|---|---|---|
| **See** | SAM 3.1 (Meta) | Finds each object in a photo or video from a short name like "blue block", and gives its exact box and outline. |
| **Understand** | Muse Spark 1.3 (Meta) | Looks at the demo photo and names the objects in words SAM can search for. Later, looks at a failed placement and explains what went wrong. |
| **Plan** | Plain Python | Works out which block goes to which slot and in what order. Exact logic, tested, explainable line by line. |
| **Act and check** | Arm + camera + Python | Moves the arm, takes a new photo, measures the placement in centimetres, retries with a correction if it missed. |

Design rule: **code for exact logic, AI for judgement.**

```
demo photo ──► Spark names the objects (each name checked by SAM) ──► target order
mess photo ──► SAM finds them ──► positions in cm
plan (Python) ──► for each block:
     arm picks and places ──► camera photo ──► SAM measures where it landed
          ├─ within 1.5 cm ──► next block
          └─ missed ──► aim the other way (at most 2 cm) ──► retry (up to 2 times)
```

- **Every photo calibrates itself.** Six dots on the sheet are found with one SAM prompt ("small coloured dot"). Four corner dots give the photo's own pixel-to-centimetre map, and two check dots must land within 0.5 cm, or the photo is refused. The camera may move between photos.
- **Spark and SAM check each other.** Spark names each object ("blue block"), and a name only counts if SAM finds exactly one object with it. Otherwise Spark is told the match count and renames it.
- **No arm maths.** The arm is taught 6-8 spots on the sheet; positions in between are blended from the nearest taught poses, and it never goes outside the taught area.

## Safety

- The arm's limits live in two places, Python and the ESP32 firmware, and a test keeps them identical. Python refuses an out-of-range move; the firmware clamps anyway.
- Until the limits are measured and checked by hand, the firmware does not even switch the servos on. Lifting that lock is one deliberate command (`scripts/arm_lock.py unlock`).
- Every move is speed-limited and ramped. The planned time the gripper stays closed is checked before moving, because the grip servo overheats.
- Every API call goes through one function, with a cache, a cost counter and a hard budget per run ($0.15). Meta's own spend limit only notifies.

## Status (2 Nov 2026)

- **Tested against the real Meta API:** SAM finds objects and the calibration dots; Spark names objects from a photo, and SAM-checked renaming fixed a name SAM could not find; dot calibration on two real photos (one calibrated, one correctly refused while the sheet's dot positions are still provisional).
- **Tested on the real board:** the locked firmware runs on the ESP32, and the laptop-to-arm link (a Windows-side serial bridge) passed a read-only test. No servo has been driven yet.
- **Built and tested against fakes only:** the taught-pose kinematics, pick and place, the full loop with retries, the camera capture, and the benchmark (unit tests with a fake arm, camera and SAM).
- **Not built yet:** Spark diagnosing failed placements, the web page, Muse Spark Open.

## Benchmark

**No results yet.** These will be filled in from `scripts/benchmark.py` runs on the real arm.

| Measure | Result |
|---|---|
| Runs | _to be measured_ |
| Success rate (all blocks placed within 1.5 cm) | _to be measured_ |
| Average attempts per block | _to be measured_ |
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
| `python scripts/plan_from_photos.py demo.jpg mess.jpg` | Two photos in, the move list out (no arm). |
| `python scripts/arm_check.py` | Read-only check that the arm answers and is locked. |
| `python scripts/teach.py --dry-run` | Practise teaching the pose grid without an arm. |
| `python scripts/run_mimic.py` | The full run with the real arm and camera. |
| `python scripts/benchmark.py --runs 5` | Several runs, cache off, with the numbers above. |

Hardware: KEYESTUDIO ESP32 4-DOF arm, an iPhone as the camera (iVCam on Windows, read with ffmpeg), WSL2 on a Windows 11 ARM laptop. Setup details are in [docs/needs.md](docs/needs.md).

## Related work

Instant Policy (Vosylius and Johns, arXiv [2411.12633](https://arxiv.org/abs/2411.12633)) treats imitation as in-context learning: given one or two demonstrations, a graph-diffusion model trained on simulated pseudo-demonstrations generates the robot's actions for a new task, without further training. Mimic is much narrower. It does not learn a skill or a policy. From one photo of your arrangement it reads only the goal (which block ends up in which slot), plans the moves with plain code, and closes the loop by measuring every placement with the camera and correcting misses. It is built for rearranging objects into slots, not for general manipulation.

## Built with

- Meta SAM 3.1 and Muse Spark 1.3, through the Meta Model API.
- Open-source libraries, used as-is: OpenAI Python SDK, NumPy, OpenCV, python-dotenv, pyserial, pytest; ESP32 Arduino core 2.0.12 and ESP32Servo 1.2.1 for the firmware; FFmpeg (run as a separate program) for the camera.
- KEYESTUDIO arm hardware. The servo pin numbers were read from the kit's sample code; no kit code is used.
