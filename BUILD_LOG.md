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
