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
