# What I built and how to explain it

One entry per task. Plain English. Each ends with a two-sentence version for judges.

## 28 Oct: talking to Meta's models

- A Python script sends a photo plus a short name ("blue lighter") to SAM 3.1 and gets back a box around that object in pixels.
- The same works on a video: SAM follows each object through every frame and keeps the same id for it.
- Muse Spark 1.3 answers in JSON when told exactly what shape to use, but it takes shortcuts a real arm cannot follow, so exact planning stays in Python.

Two sentences: "SAM 3.1 is our robot's eyes: give it a photo and a word like 'blue block' and it tells us exactly where that block is. Spark is the judgement layer, while anything that must be exact, like the order of moves, is plain tested code."
