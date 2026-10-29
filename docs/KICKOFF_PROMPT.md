Paste this into Claude Code (run `claude` inside ~/mimic with the venv active):

---

Read CLAUDE.md, then docs/description.md, docs/plan.md, docs/needs.md and docs/todo.md.

We are starting Phase 1 (Thu 29 Oct): see and plan, no hardware. Work through
docs/todo.md items 1 to 8 in order, one at a time, exactly as plan.md Phase 1 describes.

For each item:
1. Tell me your plan in 3-5 plain lines and wait for me to say "go".
2. Write the code and its tests.
3. Run the tests and show me the output.
4. Tick the item in docs/todo.md, add a line to BUILD_LOG.md, and add an entry to
   docs/LEARN.md: what you built, why this way, and how I'd explain it to a judge in
   two sentences.
5. Commit with a clear message and push. Then stop and wait for me.

Rules for this session:
- Unit tests must not call the network. Real replies go in tests/fixtures/meta_responses/.
- Ask me before any API call that is not part of the current item, and stop if the
  session's estimated spend passes $2.
- Keep the day-2 scripts working by moving them into scripts/spikes/, not deleting them.
- If anything in docs/plan.md looks wrong once you see the real code or data, tell me
  instead of quietly changing the design.

Start with item 1: tidy the repo into the layout in plan.md section 1.
