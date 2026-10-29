# Mimic: show a robot once

Mimic is a robot arm that learns a task by watching it done once, then does it
itself and fixes its own mistakes using only a camera.

Built for the Meta Global AI Developer Hackathon (Oct 27 - Nov 4, 2026).
All code in this repo was written during the event. See BUILD_LOG.md.

## Built with
- Meta SAM 3.1 (finds and tracks objects from a text prompt)
- Meta Muse Spark 1.3 (names objects in the demo photo, diagnoses failed moves)

## Setup
```
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env    # then add your META_API_KEY
.venv/bin/pytest        # unit tests only: no network, no hardware
```

## Layout
- `mimic/` the code; `scripts/` commands to run; `scripts/spikes/` day-2 experiments
- `tests/unit` (default), `tests/api` (`pytest -m api`), `tests/hw` (`pytest -m hw`)
- `docs/` description, plan, open questions, todo, learning notes

## Status
Day 3: repo laid out; building the see-and-plan pipeline (no hardware yet).
