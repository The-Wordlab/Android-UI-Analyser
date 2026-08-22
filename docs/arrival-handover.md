# Handover: arrival verdict, Phases 2 and 3

Phases 0 and 1 are shipped. This is what a fresh session needs to pick up Phase 2 or 3 without
re-deriving the reasoning. The design lives in `docs/arrival-verdict.md`; this file is the
*state of play* — what is true now, what changed while building, and what to do first.

**Read `docs/arrival-verdict.md` first.** It has the state machine, the evidence tiers, the
loading detectors, the latency ledger, and — importantly — the list of things *not* to do and
why. This file does not repeat it.

## Where it stands

| Phase | State |
|---|---|
| 0 — honest warning | shipped (`Engine._unready_destination_risk`) |
| 1 — content-wait + `arrival` | shipped (`Engine._await_rendered_destination`, `_arrival_report`) |
| the sibling problem — the world moving by itself | shipped (`meta.screen_moved`, gated `capture_hint`, errors carry their read) |
| 2 — remaining signals | **not built** |
| 3 — frame classifier | **not built, and deliberately gated** |

Knobs that exist now: `perf.arrival_extension_ms` (1200 default, `AUA_ARRIVAL_EXTENSION_MS` to
sweep, `0` reverts to Phase 0 behaviour), `output.next_actions` (off), `output.observation_meta`
(`changed` preset), `output.observation_fields`.

## Do this first, before writing any Phase 2 code

**Measure what Phase 1 still misses.** Phase 2 and 3 are both evidence-gated on purpose, and the
instrument already exists: run real flows and count how often each `arrival` state and each named
evidence string appears in the journal.

Concretely, the questions worth answering with data:

* how often does `arrival` appear at all? If it is rare, Phase 2 buys little.
* of those, how many end `settled` after waiting (the content-wait paid off) versus `loading` or
  `unconfirmed` (it did not)?
* which detector names appear in `arrival.evidence`, and which never fire?
* what do the `unconfirmed` cases have in common? That set *is* the Phase 2 backlog, ranked.

Anything built before that is a guess. The whole series' record is that guesses about this code
were wrong more often than right — see below.

## What building it taught us, that the design got wrong

Recorded because each one cost a live-device round to discover and would be re-derived expensively.

1. **`activity_changed` is nearly useless as arrival evidence on modern apps.** Single-Activity
   Compose apps never change Activity for in-app navigation, so a detector keyed on it silently
   passes the exact case it was built for. `shell_only_tree` — planned for Phase 2 — was load-bearing
   for Phase 1 and had to be pulled forward. **Expect more of Phase 2 to be load-bearing than the
   doc assumes.**
2. **`change.text_added` counts every window.** The status-bar clock ticking over reads as
   "content arrived" and vetoes a correct verdict. Any additive-arrival rule must be scoped to
   app-window elements. This will bite Phase 2's OCR-legibility signal in the same way: OCR reads
   the whole frame, including the clock.
3. **A loading shell's own chrome looks like content.** Its nav-back button is a "new actionable
   control"; a page-indicator dot is "text". Bareness has to outrank additive evidence, and a
   label only counts if it contains an alphanumeric character.
4. **The pixel check was never the problem.** The frame in the original failure was *genuinely
   pixel-quiet* — the destination was drawn, blank, and waiting on the network. Physically settled,
   semantically loading. Do not reach for settle-loop tuning; this is a classification problem.
5. **`previous_screen_gone` is the wrong comparison for an action** (it compares against the
   fingerprint just emitted, so any navigation sets it). The world-moved verdict is taken on the
   *pre-action* read, comparing actionable stable-key sets.
6. **Fingerprints move on their own.** Three consecutive analyzes of one untouched screen gave
   node counts 43, 43, 44 and two different fingerprints, with the same nine actionable ids. Any
   Phase 2 signal keyed on a fingerprint or a node count will cry wolf; key on the actionable set.

## Phase 2 — notes beyond the design doc

The doc lists the signals. These are the practical warnings.

* **OCR legibility** is the most promising and the most macOS-dependent (`_start_hierarchy_ocr`
  selects apple_vision only). It must be tri-state: no OCR available is *not* evidence of
  blankness. Use the raw box count *before* `drop_redundant`, and pin it to the settle loop's own
  final frame — the parallel augmenter may hand back a frame up to 250ms old.
* **The mask census** is free (cell indices are already computed) and is the honest fix for
  `GridSettle`'s vacuous-idle hole: a whole-screen crossfade masks every cell and "idle" becomes
  trivially true. Report the masked fraction and shape; do not un-mask animation, because masking
  is why spinners do not hang the loop.
* **One shared loading lexicon.** There are currently three divergent copies (memory's
  `_infer_state`, `_observation_is_loading`, and goto's own checks). Unify before extending, and
  extend by locale *as data*. Never per-app strings — this repo is public and
  `tests/test_no_app_specific_refs.py` enforces it.
* **Logs stay annotation, never a gate.** The digest's tag filters are agent-writable and persist
  (`logcat prefs`), so a gate an agent can starve with `--ignore-tag` is a footgun. Corroborate a
  `loading` verdict; never promote to `settled`.
* **False-positive guard**: `loading_text` alone on a content-rich additive screen (a settings row
  named "Loading behavior") must not classify as loading.

## Phase 3 — the bar to clear before building

* **Needle 2 is closed.** Tried, measured, dropped — `experiments/needle/FINDINGS.md`, 2026-08-18:
  "positional, not semantic", 5/5 correct became 0/7 by moving the answer in the list. Do not
  re-run it.
* The other local models are wrong for this: the small selectors are selectors under the policy
  guard, `gemma4` cannot see a frame, and the grounding VLM is 0.5–2s per call.
* If a classifier is built, its constituency is Compose/Flutter/canvas screens with no a11y and no
  legible text — which today already end in an honest `unconfirmed` rather than a wrong answer. So
  **the job description is only non-empty if the measurement in the first section says so.**
* Non-negotiable shape if it happens: a perception provider (`providers/` subclass + registry
  decorator + `models.<name>` config block), local weights by absolute path, never downloaded,
  absence is the normal case, runs only on ambiguous verdicts, and **may only confirm `loading` or
  demote toward `unconfirmed` — never promote to `settled`.** That way a bad model costs latency,
  never a mis-navigation.
* Training data is free: the rolling capture buffer bursts at 10fps for ~1.5s after every action
  mark, and the journal records the outcome. Auto-label by what the screen *became*.

## Open items, not part of Phase 2 or 3

* **The app-specific guard cannot see untracked files.** `tests/test_no_app_specific_refs.py`
  scans tracked files, so a violation in a new file cannot fail until the commit introducing it
  has already landed. A real resource-id from a tested app reached a public commit this way. A
  staged-file check in `.githooks/pre-commit` would close it. **Highest-value small fix on this
  list.**
* **The on-device helper settles by its own rule** ("three screenshots agree"), so a mid-stretch
  step can still act on a loading frame. The final handoff observation goes through host `_observe`
  and gets a verdict, and the offload is gated to mapped UI-only stretches and aborts on selector
  mismatch — so the damage is bounded, but it deserves its own pass.
* **`SKILL.md` has ~6 bytes of headroom** against a 5120-byte cap enforced by `tests/test_guide.py`.
  Anything new taught there requires demoting something. The non-brief manual is unconstrained.
* **Phase 1's live reproduction was not deterministic.** The mid-fetch bare frame could not be
  forced to reappear on demand, so its regressions are unit fixtures built from observed live
  shapes. A scripted device repro (network throttling around a known network-gated screen) would
  be worth having before Phase 2 changes the same classifier.

## Working here

* **Use a worktree** under `.worktree/`; this checkout is shared with other sessions. A linked
  worktree has no `.venv` — use the main checkout's binaries with `PYTHONPATH=<worktree>/src`.
* **Never `git stash`.** It removed another session's uncommitted work twice in one day.
* **Stage explicit paths.** Other sessions leave uncommitted work in this tree.
* **Restart the warm daemon** (`aua daemon stop`) after source edits, or you will test stale code.
* Test first, proven red. Build fixtures from recorded shapes, not invented ones.
* Full suite, `ruff check .`, `mypy` — all three, every time. The pre-commit hook runs the last two
  and regenerates the SKILL copies.
* **Verify on a device.** The single most reliable finding of this series is that the device
  falsifies designs the code review passes. Three of Phase 1's four corrections came from one hour
  on an emulator; none were findable by reading.
