# Mimic: show a robot once

Companion files: [plan.md](plan.md) (how we build it, phase by phase), [needs.md](needs.md) (what is verified, what is not, what to set up), [todo.md](todo.md) (what is left, in order).

Solo entry, Meta Global AI Developer Hackathon (Cerebral Valley). Build period 27 Oct to 4 Nov 2026. Track: Indie Developer, problem statement "Push the Frontier".

---

## 1. What Mimic is

### The one-sentence pitch

**Show a robot a task once. It learns it, does it, and fixes its own mistakes using just a camera. No programming needed.**

### The problem

Robots are hard to teach. Getting an arm to do a new job usually means an engineer writes code for every movement, then watches it, then tunes it, again and again. That is slow and expensive, so small businesses, labs and workshops rarely use robots at all.

Two things make it hard:

1. **Telling the robot what to do.** Someone has to turn "put these in this order" into code.
2. **Checking it worked.** Someone has to watch every attempt and adjust. The robot cannot see its own mistakes.

### What Mimic does about it

Mimic removes both:

- **You show, you don't code.** You arrange some objects once in front of the camera. Mimic works out what the arrangement is.
- **The robot checks itself.** After every move, the camera looks at what actually happened. If a block landed in the wrong place, Mimic sees it, works out why, and fixes it.

### The demo task

Coloured blocks on a white table, a small robot arm, a camera looking down.

1. You line up three blocks in an order you choose. Mimic takes a photo.
2. You scatter the blocks.
3. Mimic finds every block, plans the moves, and the arm rebuilds your row, one block at a time.
4. After each placement the camera checks it. A miss is spotted, explained, and corrected on screen.
5. The final row is shown next to your original. It matches.

The blocks are the test. The point is the system: learn from one demonstration, act, check with a camera, correct.

---

## 2. Why this matters (Impact, 30% of the score)

- **Anyone can teach a robot.** Small workshops, labs, people with assistive arms: nobody needs to write movement code.
- **Cheap hardware becomes useful.** A £43 arm is imprecise. A camera loop that measures and corrects every move makes it do the job anyway. Precision comes from checking, not from expensive motors.
- **It is where robotics is heading.** Meta's own robot research (V-JEPA 2, RoboJEPA) plans robot actions by comparing what the camera sees with a goal. Mimic is a student-sized version of that idea, built with Meta's models, on hardware anyone can buy.

Be honest in the pitch: this is a working proof of the idea on a toy-sized task, not a factory product.

---

## 3. How it works

Four jobs, each done by the right tool.

| Job | Done by | What it does |
|---|---|---|
| **See** | SAM 3.1 (Meta) | Finds each object in a photo or video from a short name like "blue block", and gives its exact box and outline. |
| **Understand** | Muse Spark 1.3 (Meta) | Looks at the demo photo and names the objects in words SAM can search for. Later, looks at a failed placement and explains what went wrong. |
| **Plan** | Plain Python | Works out which block goes to which slot and in what order. Exact logic, tested, explainable line by line. |
| **Act and check** | Arm + camera + Python | Moves the arm, takes a new photo, measures the placement in centimetres, retries with a correction if it missed. |

Design rule: **code for exact logic, AI for judgement.** We learned this on day 2: asked to plan moves, Spark gave a valid answer that a real arm cannot carry out (it "inserted" a block into an occupied spot). Counting slots is Python's job. Understanding a scene and explaining a failure is the model's job.

### The loop

```
demo photo ──► Spark names the objects ──► SAM finds them ──► target order
mess photo ──► SAM finds them ──► positions in cm
plan (Python) ──► for each move:
     arm picks and places ──► camera photo ──► SAM measures where it landed
          ├─ within 1.5 cm ──► next move
          └─ missed ──► correction (offset, or Spark's diagnosis) ──► retry
final photo ──► compare with demo ──► PASS / FAIL on screen
```

---

## 4. Why each Meta tool is load-bearing (Use of Meta AI Tooling, 25%)

- **SAM 3.1** is used as a **measuring instrument**, not to draw boxes for show. Every placement is checked by SAM in centimetres. Without it the robot is blind.
- **Muse Spark 1.3** reads the demo scene and diagnoses failures from before/after photos. Without it a human has to name every object and explain every miss.
- **Muse Spark Open** (should-have): runs the diagnosis step through Baseten, so an open-weights Meta model is part of the loop. Targets the open-weights bonus. Requires the "Built with Muse Spark Open" notice.
- **Muse Voice Transcribe** (nice-to-have): "Mimic, copy this" starts a run by voice.

What should surprise Meta: SAM closing a physical control loop, and Spark Open doing real agent work (it is known to score low on agent benchmarks).

---

## 5. Demo script (3 minutes)

| Time | On screen |
|---|---|
| 0:00-0:15 | Hands line up three blocks. Cut to the arm rebuilding them from a mess. Caption: "Nobody programmed this." |
| 0:15-0:35 | The problem: robots need an engineer for every task, and they cannot see their own mistakes. |
| 0:35-1:45 | One full real run: demo photo, Spark names the blocks, SAM boxes appear, the plan, the arm moves. One placement misses, the screen shows "missed by 2.1 cm", the correction, the retry, "PASS". |
| 1:45-2:15 | A harder run (4 blocks or a different order), sped up and labelled as sped up. |
| 2:15-2:40 | How it works: the four-job table, which Meta model does what. |
| 2:40-3:00 | Numbers from the benchmark runs (success rate, average retries, cost per run), who it is for, repo link. |

Rule: only show work built during the event. Never show the kit's sample code or its record-and-replay feature.

---

## 6. Build plan

### Must-have (the pitch works with only these)

- Spark names objects from the demo photo; SAM finds them; Python plans the moves.
- The arm picks and places three blocks into a target row.
- After each placement the camera measures it, and a miss is retried with a correction.
- A run log: every attempt with photo, measured error and PASS/FAIL.

### Should-have

- A simple local web page showing the live camera image with boxes, the plan, and the attempt log.
- Spark diagnosis of failures from before/after photos.
- Spark Open on Baseten for the diagnosis step (open-weights bonus).
- Benchmark: 10 runs, with success rate, retries and cost.

### Nice-to-have (cut first)

- Voice start with Muse Voice Transcribe.
- Stacking instead of a row (needs video to know the order of moves).
- Four or more blocks.

---

## 7. Phase gates

| Gate | When | Exit |
|---|---|---|
| 0 | Wed 28 Oct (done) | API key works; SAM finds objects in a photo and tracks them in a video; Spark returns JSON; public repo with first commit. |
| 1 | Thu 29 Oct | From two real photos (demo and mess), the plan is printed with object names chosen by Spark and positions in cm. No hardware needed. |
| 2 | Fri 30 Oct | From WSL, each arm joint and the gripper move on command; the camera saves a photo from code. |
| 3 | Sat 31 Oct | One block picked from the mess and placed in slot 1 within 1.5 cm, 4 times out of 5. |
| 4 | Sun 1 Nov, 21:00 | Three blocks rebuilt correctly, 3 runs out of 5, without touching the keyboard. If not: shrink the task (bigger blocks, fewer blocks). |
| 5 | Mon 2 Nov, 18:00 | Feature freeze. Web page, benchmark numbers, Spark diagnosis. |
| 6 | Wed 4 Nov, 21:00 UK | Video, README and submission done. The official cutoff is 02:00 UK on Thu 5 Nov; do not use it. |

---

## 7b. What was built during the event, and what was not

- **Built during the event (all in the repo):** every line of Python and firmware, the calibration, the planner, the loop, the web page.
- **Not ours, used as-is:** the KEYESTUDIO arm hardware; open-source libraries (OpenAI Python SDK, NumPy, OpenCV, ESP32 Arduino core, ESP32Servo). The kit's sample sketches are used only for the servo-centring step during assembly, never in the demo.

---

## 8. Known risks

| Risk | Fallback |
|---|---|
| The cheap arm is not precise enough to place blocks reliably. | Bigger blocks, bigger slots, the camera correction loop, a taught-pose grid instead of maths-only kinematics. |
| The arm arrives late or broken. | Return it and use the Arduino plus SG90 servo sorter (same software, simpler hardware). |
| Gripper servo overheats (reported in a review at 9 V). | NiMH batteries (7.2 V), never hold a grip longer than needed, stop if it smells. |
| Uploading to the ESP32 fails on the ARM laptop. | Run the upload from Windows, not WSL; check the CP210x driver; hold BOOT while uploading. |
| Camera moves between calibration and the run. | Mount it rigidly; recalibrate at the start of every run (takes one photo). |
| Similar colours get confused. | Pick blocks with very different colours; never use white or pale blocks on the white table. |
| API cost runs away in a retry loop. | Response cache, a cost counter, and a hard per-run budget in code. |

---

## 9. Why this can win

- It matches Meta's own wording for the track: "give a machine a new capability", "control a physical system".
- It shows a pattern judges trust: the system acts, measures, and corrects itself, visibly.
- Two Meta models do real work that cannot be swapped for a hard-coded rule.
- The demo is instantly understandable and very shareable on X, which matters for Most Viral ($20K) and Community Vote ($10K).
- Bonus targets: Best Agentic Use-Case, Best Build on Meta Open Weight Models (if Spark Open is in the loop), Most Viral, Community Vote.

Judging: Impact 30%, Use of Meta AI Tooling 25%, Technical Depth 20%, Demo 25% according to the official rules as reported; one participant guide says equal weights. Build for both: the demo cannot be weak.
