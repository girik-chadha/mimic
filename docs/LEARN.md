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
of noise looked like several cm. (Correction, 29 Oct item 7: the check sticker can go anywhere, even the
centre. Only the 4 corner stickers must not have 3 in a line. The "not in line" problem only affected the
checking-against-the-other-four idea we threw away.)

- **Tests** use a made-up tilted camera: a tilted A3 sheet measures 42 x 29.7 cm, a 10 cm square measures
  10 cm on every side, and a check sticker written down 2 cm wrong is refused with the numbers.

Two sentences: "The camera sees the table at an angle, so we use a homography: a 3x3 table of numbers, worked
out from 4 stickers at known positions, that turns any pixel into centimetres on the table. A fifth sticker
that the map never saw checks it, and if the map puts that sticker more than half a centimetre off, Mimic
refuses to run."

## 29 Oct, item 7: scene.py, reading the photo

- **What:** `calib/layout.json` holds the fixed facts about the sheet: each sticker's name and position, the
  slot centres, the sheet edges. From one photo, `calibrate()` finds the stickers with SAM and builds the
  pixel-to-cm map. `read_positions()` finds each object and gives its position in cm. `read_target()` puts
  each object of the demo photo into its nearest slot and returns the names in slot order: that is the
  arrangement the robot must copy.
- **Strict on purpose:** SAM must find each name exactly once. An object off the sheet, more than the
  tolerance away from every slot, or two objects in one slot each stop the run with a clear message.
  The tolerance may not be more than half the gap between slots, so an object can never count as being in two.
- **Block height:** the map is exact only for things lying flat. The middle of a block is a little above the
  table, and from a camera above, raised things look further from the centre of the view than they are.
  The fix pulls each point towards the spot under the lens by (camera height - point height) / camera height.
  For a point 20 cm from that spot, 1.25 cm up, with the lens 45 cm up, that is about 0.56 cm. It is off until
  Gate 1 shows we need it; a test traces the camera ray to prove the formula.

Two sentences: "One photo of the sheet gives us the stickers, which give the pixel-to-centimetre map, and then
the position of every block in centimetres. The demo order is read by putting each block in its nearest slot,
and anything ambiguous, like a block between two slots, stops the run instead of guessing."

## 29 Oct, item 8: plan_from_photos.py, the Gate 1 demo

- **What:** one command, two photos. It calibrates each photo from its stickers, asks Spark to name the
  objects in the demo photo (checked by SAM), reads which object is in which slot, finds the objects in the
  mess photo in cm, and prints the list of moves. It saves both photos with boxes, cm labels, numbered slots
  and arrows, plus a `plan.json` with every number, in `runs/<date-time>/`.
- **Safety checks you can point at:**
  - each photo's check sticker error is printed; over 0.5 cm, the run stops;
  - if the demo and mess maps disagree by more than 0.5 cm at the sheet centre, the camera moved: a loud
    warning (the plan is still right, because each photo uses its own map, but the arm would not be);
  - a hard $0.15 cap per run through the same budget guard as everything else;
  - it refuses to run at all while `calib/layout.json` still holds placeholder numbers.
- **Tests** run the whole script offline: a fake Spark, a fake SAM that "sees" a tilted sheet, and blank photos.
  They check the printed moves, the saved files, the camera-bump warning, a bad check sticker, and the cap.

Two sentences: "Give Mimic a photo of your arrangement and a photo of the mess, and it prints exactly which
block goes where, in centimetres, with a picture to prove it. Before it trusts anything it checks its own
calibration against a spare sticker and refuses to run if the numbers do not add up."

## 29 Oct, Phase 2 software: firmware, arm.py, arm_jog.py (before the arm arrives)

- **What:** the arm understands four one-line commands: `J` (move all joints smoothly, e.g. `J 95 90 87 90 400`),
  `H` (home), `?` (where are you?) and `S` (stop). `arm.py` sends them from Python; `arm_jog.py` lets you type
  `base +5` to move one joint; the firmware on the ESP32 carries them out.
- **Safety in two layers:** Python refuses any angle outside the limits (that would be a bug), and the firmware
  clamps to the same limits anyway, as the last line of defence. A test checks both layers use identical numbers.
  The firmware also: slows any move to a speed limit, eases in and out (no jerks), relaxes a closed grip after
  5 s (the grip servo overheats), rejects over-long lines, and holds still when no command arrives.
- **Nothing moves by accident:** `calib/arm_limits.json` starts as a placeholder. A real arm only moves when
  it says `placeholder: false` *and* `limits_checked_by_hand: true`. Stop and status always work.
- **How firmware is tested without a board:** all the rules live in `arm_logic.h`, plain C++ with no Arduino
  calls. The laptop compiles it with g++ and runs 38 checks: clamps, ramps, timers, refusals. The `.ino` file
  only connects those rules to the servos and the serial port.
- **Known limit:** at power-on the servos jump to home, because a hobby servo cannot say where it is.

Two sentences: "The arm speaks four one-line commands, and every safety rule (joint limits, speed limit, grip
timeout) lives in a small C++ file that we test on the laptop, so the rules are proven before the arm is even
plugged in. Python checks the same limits first, and a test makes sure the two copies of the numbers never differ."

## 30 Oct: every photo calibrates itself from 6 dots

- **What changed:** coloured stickers with names ("red sticker") failed, because SAM cannot tell red from orange.
  Now the sheet has 6 plain dots, and one prompt, "small coloured dot", finds them all. Which dot is which comes
  from *where* they are: the 4 outside dots are the corners, and we try each of the 4 ways the photo could be
  turned, keeping the one where the 2 check dots land where they should.
- **Why every photo calibrates itself:** the camera is allowed to move between photos. Each photo finds its own
  dots and builds its own map, so a bumped camera cannot quietly spoil the numbers.
- **Safety checks:** exactly 6 dots (after dropping anything too small, like a speck, or too big, like a block);
  both check dots within 0.5 cm; if two different turns both fit, stop. A labelled photo with every dot named
  is saved, so the roles can be checked by eye.
- **Real result:** on two real photos from different heights, one calibrated (right check dot 0.49 cm off) and one
  was refused (0.56 cm). Both put that dot about 0.5 cm right of the provisional number, so the number is what is
  wrong; a ruler measurement fixes it. We did not loosen the limit to make it pass.
- **Known limit:** both check dots sit on the middle line, so a mirror-image photo would be read upside down.
  Moving them off the middle lines (todo, before Gate 3) fixes it.

Two sentences: "Six dots drawn on the sheet let every photo work out its own pixel-to-centimetre map, so the camera
can move freely; four dots build the map and two more check it to within half a centimetre. When the check
fails, Mimic refuses the photo instead of guessing, which is exactly what happened on one of our real test photos."
