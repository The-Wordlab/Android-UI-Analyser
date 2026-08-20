# Needle 2 for AUA — ship-now plan and value benchmark

Status: proposed, nothing built.
Target: extract real value from a 14 MB local model **this week**, then prove it added value,
answers correctly, and costs zero in-test latency.

Companion: [../functiongemma/EXPERIMENT_LOG.md](../functiongemma/EXPERIMENT_LOG.md).
That experiment is a different job (navigation / candidate selection) and is not in scope here.

## Why this job, and not navigation

Needle 2 is a 45M-parameter, 14 MB, ~28 MB-RAM model for tool calling and **structured
extraction**, with a 256-token sliding window and a calibrated confidence head. Apache-2.0.
`pip install cactus-needle`.

Two properties decide the scope:

- **256-token window** rules out stateful navigation (no room for history, loop detection,
  progress tracking). Navigation stays with a 32K-context model.
- **Calibrated confidence** is exactly what an optional housekeeping task needs: answer when
  sure, escalate when not. Fine-tuning **disables** it, so this job uses the **base model,
  untuned**.

The chosen job is AUA's own question backlog — `meta.ask` / `research_tasks`, which the guide
already describes as "unresolved map questions ready for an external agent". Today an expensive
agent answers those. They are:

- **stateless** — one screen, one answer
- **latency-insensitive** — nothing waits for a screen name
- **near-zero risk** — a bad name cannot fail a test, it only makes the map worse
- **machine-checkable** — a name can be validated deterministically
- **compounding** — better names improve `goto` / `map --find` for every later run

## Both ends of the pipeline already exist

```
research_tasks          →      [ this plan ]      →      --answers TASK_ID="name"
  memory.py (built)              (missing)                    cli.py (built)
```

- `memory.py`: `research_tasks`, `refresh_research_tasks(...)`, `_research_prompts(...)`
- `cli.py`: `--answers TASK_ID="value"`, `map --audit`, `explore plan`
- `providers/`: pluggable local backends already normal here (`ocr/apple_vision`,
  `grounding/local_vllm`, `policy/functiongemma`, registered by decorator)
- Ground truth: every screen the map already named carries `meta.known_screen`

Only the middle is new.

## Phase 0 — build the ruler first (blocking)

Nothing else in this plan is interpretable without these. The FunctionGemma experiment failed
eight times partly because it was graded on synthetic accuracy plus a handful of real rows.

0.1 **Baseline snapshot.** Before any model runs, record per app map:
- count of open `research_tasks`
- count of named screens (`known_screen`)
- `goto` coverage: how many named screens have a verified route
- map size / edge count

0.2 **Turn counter.** Agent turns per scenario, from the journal, alongside the existing
authoritative AUA call count (`session review`). This is the scoreboard for every later
decision, including navigation. Today a scenario is roughly 20 agent turns and $0.88.

0.3 **Deterministic naming baseline.** Implement the dumb version *first*, in plain code:
take the toolbar title, else the first/largest heading, else top-of-screen text. Score it on
the ground truth from 1.1. **This number is the bar Needle must beat.** Skipping this step is
the single repeated mistake of the FunctionGemma rounds.

## Phase 1 — free offline probe (about one hour, $0, no repo code)

Nothing touches a test. No training. Throwaway script outside `src/`.

1.1 **Mine ground truth.** Pull screens where the map already holds a `known_screen` name,
with their element text. Target ≥50 examples; note the actual count. Hold out a split that
nothing tunes against. Prefer stock-app maps where available so the probe is publishable.

1.2 **Pre-filter candidates in code.** For each screen, reduce visible text to plausible title
candidates (toolbar, headings, largest text, top-of-screen). This keeps the prompt inside 256
tokens and turns the task into extraction rather than generation.

1.3 **Ask base Needle** with a tool schema over those candidates, e.g.
`name_screen(candidate_id)`. Grammar-constrained output means it can only return one of the
strings supplied — it cannot invent or malform a name.

1.4 **Score, always against Phase 0.3:**
- exact-match and fuzzy-match agreement with the recorded name
- **Needle vs deterministic baseline** — the only comparison that matters
- confidence vs correctness: find the threshold where precision ≥ 95%
- per-shape breakdown: screens *with* an obvious title vs screens needing judgment

**Gate.** Beats the deterministic baseline by a clear margin → continue. Ties or loses → stop,
ship the deterministic baseline instead, and record it here. Below ~30% → drop Needle for this
job entirely.

## Phase 2 — the provider (only past the Phase 1 gate)

2.1 New provider category `src/android_ui_analyser/providers/naming/needle.py`, registered by
decorator, mirroring `providers/policy`. Contract: batch in, answers + confidence out.

2.2 **Off by default**, lazy import, no model load unless explicitly enabled — same discipline
as the existing policy provider, with the same style of test asserting no import while off.

2.3 **Deterministic validation gate on every answer.** The model proposes, code vetoes:
- non-empty, sane length
- unique within the app map
- not a generic word (`Screen`, `Page`, `Activity`, `Fragment`, …)
- actually present in the screen's visible text
- confidence ≥ the Phase 1.4 threshold

Confident **and** valid → commit via the existing `--answers` path. Anything else → leave the
question in the queue for an agent. Log every decision: question, candidates, choice,
confidence, validation result, committed yes/no.

## Phase 3 — the batch drainer

3.1 One offline command that loads the model once, drains the open backlog, and exits — e.g.
`aua map --audit --answer-with needle`. Runs after a session or nightly. **No agent involved.**

3.2 Batching amortises model load across the whole queue, so startup cost approaches zero per
answer.

3.3 **Hard architectural rule: the drainer never runs inside a test session.** This is what
makes in-test latency structurally zero rather than merely small.

## Phase 4 — the three questions asked of this work

### 4a. Did it add value?

- `research_tasks` drained with **zero agent turns** (absolute count, and % of backlog)
- `goto` coverage before vs after draining (Phase 0.1 snapshot is the before)
- **agent turns per scenario** before vs after, on the real benchmark, using the Phase 0.2 counter
- three-way arm comparison on the same scenarios: names from Needle / names from an agent /
  no names at all

### 4b. Is it doing a correct job?

- held-out agreement from Phase 1.1 (trivially clean — there is no training)
- **still against the deterministic baseline**, not in isolation
- precision at the committed-confidence threshold; target ≥95% on what it accepts
- human spot-check of 30 committed names — do they read as names a person would choose
- **regression check: did any previously working `goto` route break** after renaming? This is
  the only real damage a bad name can do, and it is directly checkable

### 4c. Did it add latency?

The structural argument comes first, the measurement confirms it:

- a test asserting the naming provider is **never loaded during a test session**
- test-session wall time with naming enabled vs disabled — expect identical within noise
- the drainer measured separately and reported honestly: questions/second, total batch wall
  time, peak RSS. It is a real cost, but off the critical path.

## Kill gates, decided in advance

| Trigger | Action |
| --- | --- |
| Probe below ~30% agreement | drop Needle for this job |
| Needle ≤ deterministic baseline | ship the baseline, drop the model |
| Precision < 95% at any usable confidence threshold | do not auto-commit; propose-only |
| Any `goto` route regression from a committed name | fix or revert; block Phase 3 |
| Any measurable in-test latency | architecture bug — fix before continuing |

## Explicit non-goals

- **No fine-tuning.** It disables the calibrated confidence head, which is the reason this job
  is safe. If tuning ever looks necessary, the abstain gate must be rebuilt first.
- **No on-device deployment.** Needle has no documented Android/ARM runtime — the published
  README is Python-only and the engine binary is fetched per platform. CLI-only until Cactus
  confirms otherwise (`doc/apis.md`, or `founders@cactuscompute.com`).
- **No navigation, no test driving, no verdicts.** AUA's deterministic assertions keep the
  verdict. A model must never be able to report PASS.
- **No coupling to the FunctionGemma experiment.** Different job, different context budget,
  separate scoreboard.

## Rough effort

| Phase | Work |
| --- | --- |
| 0 — ruler + deterministic baseline | ~0.5–1 day |
| 1 — probe | ~2–3 hours incl. data mining |
| 2 — provider + validation gate | ~1 day |
| 3 — drainer | ~0.5 day |
| 4 — benchmarks | ~1 day |

Phases 0 and 1 are worth doing regardless of the outcome: the turn counter and the
deterministic naming baseline are permanent assets, and both are prerequisites for judging any
future navigation model.
