# Mimic: show a robot once

Solo hackathon project (Meta Global AI Developer Hackathon, 27 Oct to 4 Nov 2026).
A robot arm watches a human arrange coloured blocks once, then rebuilds that
arrangement from a messy table, checking and fixing every move with a camera.

Read these before any task:
- docs/description.md  what we are building and why
- docs/plan.md         repo layout, interfaces, phases, tests, working rules
- docs/needs.md        verified facts (Part A) and open questions (Part B)
- docs/todo.md         what is left, in order

## How to work with Girik

- Girik is a CS student who must explain every file to judges. Plain English, short.
- One task at a time from docs/todo.md. Before coding, say the plan in 3-5 lines.
- After each task: run the tests, show the result, tick docs/todo.md, add a line to
  BUILD_LOG.md, add an entry to docs/LEARN.md (what was built, why this way, how to
  explain it in two sentences), commit with a clear message. Then stop and wait for "go".
- If a fact in docs/needs.md Part B gets settled, move it to Part A with the date.

## Hard rules

- Never print, log or commit the API key (.env is gitignored). Never use -contributor models.
- Unit tests never call the network; use saved replies in tests/fixtures/meta_responses/.
- Ask before: spending more than $2 of API in a session; changing a design decision in
  docs/plan.md; sending new angles to the real arm; installing anything on Windows.
- New code only. Never copy the arm kit's sample code into the project.
- Keep code simple and explainable. No clever tricks, no unnecessary dependencies.

## Environment

- WSL2 Ubuntu (ARM64) on a Windows 11 ARM64 laptop. Repo at ~/mimic, venv .venv.
- USB devices and webcams are not visible inside WSL. Arm uploads run from Windows.
- Repo: https://github.com/girik-chadha/mimic (public).

## Meta Model API (verified 28 Oct, details in docs/needs.md A1-A5)

- OpenAI Python SDK, base_url https://api.meta.ai/v1, client.responses.create(...), resp.output_text.
- Models: sam-3.1 (segmentation), muse-spark-1.3 (reasoning).
- SAM image: content [{"type":"input_text","text":"blue block"}, {"type":"input_image","image_url":"data:image/jpeg;base64,..."}]
- SAM video: same with {"type":"input_video","video_url":"data:video/mp4;base64,..."}
- SAM box: <|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|> then <|mask;...|> (ignore masks).
  Video: <Nf> marks frame N; the number right before <|box is the track id.
- One short name per object ("blue block"). Prices: SAM $2.50/1k images,
  $0.20/1k video frames (min 50); Spark $1.25/$4.25 per 1M tokens in/out.

## Design decisions (do not change without asking)

- Code for exact logic (move order, geometry, safety limits). AI for judgement
  (naming objects from the demo photo, diagnosing failed placements).
- Scene: A3 sheet with 4 calibration stickers, a mess area, and an empty target row
  of numbered slots, so placements never collide.
- Safety limits live in the firmware as well as in Python.
