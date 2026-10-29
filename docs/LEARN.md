# What I built and how to explain it

One entry per task. Plain English. Each ends with a two-sentence version for judges.

## 28 Oct: talking to Meta's models

- A Python script sends a photo plus a short name ("blue lighter") to SAM 3.1 and gets back a box around that object in pixels.
- The same works on a video: SAM follows each object through every frame and keeps the same id for it.
- Muse Spark 1.3 answers in JSON when told exactly what shape to use, but it takes shortcuts a real arm cannot follow, so exact planning stays in Python.

Two sentences: "SAM 3.1 is our robot's eyes: give it a photo and a word like 'blue block' and it tells us exactly where that block is. Spark is the judgement layer, while anything that must be exact, like the order of moves, is plain tested code."

## 29 Oct, item 1: tidying the repo

- **What:** Folders for each kind of thing: `mimic/` is the real code, `scripts/` are commands you run,
  `scripts/spikes/` keeps the day-2 experiments as a record, `tests/` has three tiers (unit: free and
  offline; api: real Meta calls, only on purpose; hw: needs the arm or camera).
- **pyproject.toml** lists the libraries we depend on and installs `mimic` into the venv, so any script
  or test can just `import mimic`. Plain `pytest` skips the `api` and `hw` tiers automatically, so the
  everyday test run never costs money or needs the arm.
- **Fixtures:** the real SAM reply for the 317-frame video is saved to disk, so tests can check our
  parser against real Meta output without calling the network.
- **Why this way:** it is the standard Python project shape, and it separates "costs money / needs
  hardware" from "free and instant", which keeps tests fast and the API bill safe.

Two sentences: "The repo is split into the robot's code, the commands you run, and tests in three tiers:
free offline tests, real Meta API tests, and hardware tests. Every test that touches Meta's API uses a
real reply saved to disk, so we can test constantly without spending credits."

## 29 Oct, item 2: config.py and meta_api.py

- **What:** `config.py` holds every setting in one place (model names, prices, budget, folders).
  `meta_api.call(model, content)` is the only function in the project that talks to Meta.
- **Cache:** every reply is saved in `.cache/`, named by a hash of the model, the prompt and the file's
  bytes. Asking the same question twice costs nothing and works offline.
- **Cost and budget:** each call's price is worked out from Meta's price list (photos per image, videos per
  frame with a 50-frame minimum, Spark per token) and added up. Once the run has spent `MIMIC_BUDGET_USD`,
  the next call is refused before it is sent. Every paid call is also written to `.cache/spend_log.jsonl`.
- **Retries:** if Meta says "busy" (429) or has a server error (5xx), wait 1, 2, then 4 seconds and try
  again. Mistakes on our side (400) fail at once, because retrying would not help.
- **Safety:** `-contributor` models and unknown models are refused before anything is sent.
- **Why this way:** one door means the cache, the budget and the safety rules cannot be skipped by
  accident. Meta's own spend limit only sends an email, so the hard stop has to be in our code.
- **Tests:** a fake client plays Meta's part, so the tests are free and instant. I broke each guard on
  purpose and checked that a test failed; one test was too loose and got tightened.

Two sentences: "Every call to Meta goes through one function that caches answers, retries when the server
is busy, and keeps a running bill. If a run hits its budget the next request is never sent, because Meta's
own spend limit only notifies and does not stop charges."

### 29 Oct, item 2 follow-up: making meta_api safe to show and honest to measure

- `use_cache=False` makes a call skip the cache completely, so the benchmark measures real cost and time.
- `reset()` sets the bill back to $0 at the start of each run; `cost_so_far()` is what goes in `summary.json`.
- Any error is cleaned before it is raised: the API key becomes `***`, also in the error that caused it,
  so a traceback on screen during the demo can never leak the key.
- Every request gives up after 60 seconds, so a stuck call cannot freeze the robot.

Two sentences: "Each run starts its bill at zero and records the true total, and the benchmark can switch the
cache off so its numbers are real. If anything goes wrong, the error text is scrubbed of our API key before
it can appear on screen."

## 29 Oct, item 3: sam.py, reading SAM's answers

- **What:** SAM replies with text like `<0f>0<|box;x1=255;y1=363;x2=320;y2=583;w=1280;h=960|><|mask;...|>`.
  `parse_boxes` turns that into `Box` objects (corners, image size, match id, and a `centre`).
  `parse_tracks` does the same for videos, grouping boxes by track id, frame by frame.
  `find(photo, "blue block")` sends the photo through `meta_api` and returns the boxes.
- **What we learned from real replies:** photos and videos use the same format (a photo is frame 0);
  extra matches are numbered `,1`, `,2`; no match is an empty reply.
- **Why it is safe:** the outline (mask) text is long and random-looking, so first we checked that it never
  contains `<` or `>`. Then we cut masks out before reading boxes, so mask text can never be mistaken for
  a box. A reply cut off half way still gives every box that arrived complete, instead of crashing.
- **Tests** run on real replies saved to disk, and the boxes were drawn on the photo and checked by eye.

Two sentences: "SAM answers in a compact text format, and `sam.py` turns that into boxes with a centre point,
which is where the robot will aim. We tested the parser on real SAM replies saved to disk, including
broken and empty ones, so it is checked against what Meta actually sends."

## 29 Oct, item 4: brain.name_objects, Spark names, SAM checks

- **What:** Spark looks at the demo photo and names each object as "colour + object type" ("blue lighter").
  Then SAM searches the photo for each name. A name is only accepted if SAM finds exactly 1 object.
  If not (0 or 2+), Spark gets the photo again with a note like `"black box": 0 matches` and gives a new name.
  At most 2 retries per object; if it still fails, the run stops with a message listing every name tried.
- **Why:** Spark and SAM do not always use the same words. Spark called a black tray "black box", and SAM
  found nothing, because to SAM a "box" is something else. Instead of hard-coding names, the two models
  check each other, and the code decides when a name is good enough (exactly 1 match).
- **Safety:** the retries have their own cost cap ($0.03 per call), on top of the run budget. Every
  attempt (round, name, box count) is recorded for the run log.
- **Facts settled today:** Spark accepts images (B1), and Meta's replies include token counts (B10).
  Spark bills hidden reasoning tokens: a 45-character answer cost 704 output tokens.

Two sentences: "Spark suggests a name for each object and SAM checks it by searching the photo; a name only
counts if it finds exactly one object. When it doesn't, Spark is told what went wrong and tries again, which is
how 'black box' became 'black rectangle' in our test."

## 29 Oct, item 5: plan.py, the move list

- **What:** `plan_moves(positions, target_order, slots)` gives one `Move` per object: the first name in the
  target goes to slot 1, the second to slot 2, and so on. Each move says where the object is now and where
  it goes, in cm.
- **Why so simple:** the target row starts empty, so every move goes from the mess to an empty slot and can
  never land on another block. No clever ordering is needed, and it is exactly right every time.
  On day 2, Spark planned a move a real arm cannot do; this is why planning is plain code.
- **Refuses bad input with a clear message:** empty target, a name not found on the table, a name used twice,
  more names than slots, two targets less than 1 cm apart (SAM probably found one object twice), or two
  slots at the same point. Objects not in the target are left alone and reported so the screen can warn.
- **Tests** check the exact list of moves, not just how many there are.

Two sentences: "The planner is a few lines of plain Python: the first object in your demo goes to slot 1,
the second to slot 2, and so on, and slots start empty so moves never collide. Anything suspicious, like two
names pointing at the same spot, stops the run with a clear message instead of moving the arm."

## 29 Oct, item 6: calib.py, from pixels to centimetres

### What a homography is (plain English)

The camera looks at the A3 sheet from above, but never perfectly straight down. So the sheet appears as a
slightly lopsided four-sided shape in the photo: the far edge looks shorter than the near edge, and a 1 cm
step is a different number of pixels in different places. Pixels are not centimetres.

A **homography** is a 3x3 table of numbers that undoes this, for anything lying flat on the sheet. To use it,
take a pixel (x, y), write it as three numbers [x, y, 1], multiply by the table, and you get three numbers back.
**Divide the first two by the third**, and the result is the position in centimetres. That last division is the
"perspective" part: it is what makes the far side of the sheet get stretched back to its real size.
The reverse (centimetres to pixels) uses the inverse table.

### Why 4 points are enough

The 3x3 table has 9 numbers, but multiplying all of them by the same amount gives the same answer (it cancels
out in the division), so only 8 of them really matter. Each sticker we can see gives 2 facts: where it is
across (x) and where it is up (y). 4 stickers x 2 facts = 8 facts, which is exactly enough to work out the 8
unknowns. The one condition: no 3 of the 4 stickers may be in a straight line, or the facts repeat each other.
It only works for things lying on the flat sheet; something sticking up (like the top of a block) sits
slightly off the map.

### Why a 5th "check" sticker, kept out of the fit

With exactly 4 stickers the map always passes perfectly through all 4, so a sticker measured in the wrong
place gives a wrong map that *looks* perfect. So there is a 5th sticker that is not used to build the map: we
ask the map where the 5th sticker should be and compare with the ruler. More than 0.5 cm off, and the
calibration is refused with the numbers in the message.
We tested two other ideas first and both were worse: fitting all 5 together let the map bend to hide a 2 cm
mistake (it showed as 0.13 cm), and checking each sticker against the other four was so sensitive that 1 pixel
of noise looked like several cm. The 5th sticker must not be in line with two others (the exact centre is on a
diagonal, which broke one of our tests).

- **Tests** use a made-up tilted camera: a tilted A3 sheet measures 42 x 29.7 cm, a 10 cm square measures
  10 cm on every side, and a check sticker written down 2 cm wrong is refused with the numbers.

Two sentences: "The camera sees the table at an angle, so we use a homography: a 3x3 table of numbers, worked
out from 4 stickers at known positions, that turns any pixel into centimetres on the table. A fifth sticker
that the map never saw checks it, and if the map puts that sticker more than half a centimetre off, Mimic
refuses to run."
