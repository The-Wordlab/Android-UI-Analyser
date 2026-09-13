# Real-application mode: claim, compact, judge, name

Written 2026-09-13 after driving the hosted controller against a real application instead
of the public fixtures. The model was fine. The harness was the bottleneck, in four measured
ways, and this document records the fixes and the design behind them.

## What broke on a real application

| Symptom | Measured | Cause |
|---|---|---|
| Correct run never registers as done | 6 rejected `session_finish`, 13 wasted steps after the goal was met | Completion was fixture-contract-only. A real `session_start` has no machine contract, so `session_finish` returns `finished: false` forever and the model hunts for evidence. |
| Prompt bloat | 11k → 100k prompt tokens in 18 steps; 27.5 KB per frame | Every frame carried ~20 fields per element (bounds, center, depth, false flags) and ~20 metadata keys that inform no decision. |
| Cost from looping, not thinking | $0.023 per run where a clean run is ~$0.002 | Same two causes: no accepted finish, ever-growing history. |
| Pinned route stopped working | 404 "Filter by Parameters" one day after the pilot | `validate_request_config` hard-required `require_parameters: true`; OpenRouter's parameter filter excluded a provider that does serve the request. |

## The primitive: one bounded decision

`judgement.py` adds `Decider.decide()`: a fresh two-message conversation, a forced native
tool call whose parameters are the answer schema, its own `max_tokens` (default 1024), its
own reported-cost stop, one repair attempt. It reuses the controller's model and routing
(`hosted.configure_payload` / `validate_request_config`) and `run_live.completion`, but never
the controller's history. Same weights, separate window, separate budget.

Everything a decision returns carries `oracle: "model_judgement_v1"` and `verified: false`.
It is an opinion from a cheap model. Where AUA has a contract, AUA's verdict wins; where it
has none, a labelled opinion beats an unlabelled infinite loop.

Roles built on it:

- **Outcome judge** (`judge_outcome_votes`). Two stances, neutral and skeptical, each see the
  goal, the compacted final frame, up to three earlier frames, and a terse action log with
  element ids removed. Neither sees the controller's reasoning or history. Verdict enum is
  `pass | pass_with_warning | fail | blocked | unverified`, aligned with the QA suite's
  PASS / PASS_WITH_WARNING / FAIL_CRITICAL / BLOCKED. Disagreement is `unverified`, never a
  coin flip; pass plus pass-with-warning is pass-with-warning; blocked plus fail is blocked.
- **Screen namer** (`ScreenNamer`). Given one compacted frame, returns a stable snake_case
  `logical_name`, a `kind`, one-sentence `purpose`, and up to five `landmarks`. Cached by
  fingerprint, so every distinct screen costs one call and repeats cost nothing. AUA's
  heuristic `known_screen` label is passed in as context so the two can be reconciled later.
- **Route summariser** (`summarize_route`). Screens plus transitions in, a short factual map
  and memory entry out: how to get there, what confirms each hop, what to watch for.

## Compaction (`compaction.py`)

Applied after the hosted privacy projection, before the model reads a result. Per element it
keeps `id`, `text`, `desc`, `resource_id` and positive state flags; it drops bounds, centres,
depth, class, false flags, empty containers and status-bar chrome. Frames are capped at 60
elements with interactive ones preferred and an `elided_elements` count. Text is trimmed to
120 characters. When a frame's fingerprint equals the previous one, only interactive handles
are repeated under `unchanged: true`. Goal progress keeps status and objectives, not ledgers.

Measured on 12 real frames captured earlier: 342 KB projected → 59 KB compact, 17% of the
size, 5.8× smaller. Raw evidence under `controller/evidence/` is untouched; only the model's
view shrinks.

## Loop breakers (`agent_loop.py`, additive)

- `terminal_claim_limit`: after this many rejected terminal-tool calls the loop stops with
  `stop_reason: "terminal_claimed"` and records the claim as `terminal_submission.accepted:
  false`. Real-app mode uses 1: the claim is the signal, the judge is the check.
- `no_progress_limit`: this many consecutive executed calls returning the same fingerprint
  stops with `"no_progress"`. Real-app mode uses 4.

Both default to off, so fixture runs are unchanged. An accepted contract still stops with
`"terminal_tool"` first.

## Routing (`hosted.py`)

`require_parameters` is now an optional boolean. One pinned provider, `allow_fallbacks:
false` and `max_price` remain mandatory, so a request never reroutes silently. The manifest's
DeepInfra entry is repinned to plain `deepinfra` with `require_parameters: false`, the route
that serves native tools today.

## The runner (`run_realapp.py`)

```
python -m experiments.aua_controller.run_realapp \
  --goal "Open settings and switch the theme to Dark" \
  --package <app.package> --launch [--activity <cls>] [--setup-flow login.yaml] \
  --model or-deepseek-v4-flash-0731-low-deepinfra --map \
  --output <private-dir>
```

Flow: `session_start` → optional launch / setup flow → `analyze_screen` → `run_agent` with
compact-v1 tools, hosted-v1 projection, compaction and both breakers → fresh `analyze_screen`
→ judge (skipped when AUA accepted a contract) → optional screen naming and route summary
→ `result.json`, `verdict.md`, `screens.json`, `route.json` → `session_finish
allow_incomplete: true`.

`session_finish` is offered to the model with two fields of its own, `outcome`
(`achieved | already_satisfied | blocked | not_achievable`) and a short `note`. The runner
records them as the claim and calls AUA with the harness-owned arguments only. The claim
reaches the judge labelled `controller_claim_untrusted`.

Cost is surfaced per tier (controller, judge, map) with model and provider, and totalled.
Every model call is a paid OpenRouter call and needs `OPEN_ROUTER_API_KEY` in the
environment; nothing runs without `--model` naming a manifest candidate.

## Live result on a real application (2026-09-13)

Same goal ("open settings, switch the theme to Dark"), same model and route, same emulator.
Before and after are from `result.json` files; the earlier baseline is the morning's
fixture-runner driver on the same app.

| | Baseline (fixture finish gate) | Run 1: Light → Dark | Run 3: already Dark |
|---|---|---|---|
| Stop reason | never finished (6 rejected finishes, step budget) | `terminal_claimed` after 1 claim | `terminal_claimed` after 1 claim |
| Steps | 18–19 (13 wasted) | 6 | 5 (verified, no redundant tap) |
| Prompt tokens, first → last request | 11k → 100k | 3.1k → 7.0k | 3.1k → 6.9k |
| Controller cost | $0.023 | $0.00093 | $0.00132 |
| Judge cost (2 votes) | — | $0.00057 | $0.00047 (re-judged offline) |
| Map cost (names + route) | — | $0.00062 | $0.00040 |
| Verdict | none | PASS, both stances agree | PASS, both stances agree |
| Wall clock | ~4 min | 68 s | ~75 s |

Two judge-prompt defects surfaced live and were fixed the same day. The skeptical stance
first failed a run because the theme was *already* Dark ("no proof the agent changed it");
a goal names an end state, so that is now `pass_with_warning`, and AUA's contract-less
`goal_progress` (always 0/1) is no longer shown to judges. It then refused proof that was
not in the *final* frame, although the controller is told to return home after the change;
proof is now any observed frame that shows the end state with nothing later contradicting
it. Run 3 was re-judged from its saved evidence with the final prompts, no device involved,
which is the point of keeping the judge separate from the loop.

The screen namer first produced six names for three screens because a fingerprint changes
when a row's value changes. It is now keyed on AUA's own `known_screen` label and sees the
names already assigned; run 3 produced exactly three (`home_empty`, `settings_main`,
`theme_option`) and a five-hop route summary with pitfalls a tester would recognise.

Evidence for these runs is in the session's private scratch directory, not in the repository.

## Tests

`tests/test_aua_controller_compaction.py`, `tests/test_aua_controller_judgement.py`,
`tests/test_aua_controller_realapp.py` cover the filter, the decider (fresh window, forced
tool, repair, spend stop, id stripping, two-vote agreement, namer cache) and the runner end
to end with a fake AUA and a fake model (claim stops the loop, contract acceptance skips the
judge, disagreement is unverified, stall downgrades to warning, map is opt-in, setup failure
still cleans up). The hosted routing test now asserts `require_parameters` is optional.

## What this means for core AUA

The primitive generalises. The suggested path, not yet built:

1. A `judge` provider kind next to perception providers, configured under `models.judge`,
   off by default, with its own token and cost budget. Same opt-in rule as every other paid
   tier: nothing pays unless the user turns it on, and every result names the tier that ran.
2. `session_finish` without a contract returns `oracle: "model_judgement_v1"` verdicts when
   the judge is enabled, instead of `finished: false` forever.
3. `known_screen` uses the namer when the heuristic label is a generated hash, and stores the
   logical name in the app map with the fingerprint, so maps and memories read like
   `settings_theme` instead of `screen_3_ab12`.
4. Route summaries become the memory entry written at `session_finish` for a new path.

Each of those needs a fake-provider test and lands as its own small change.
