# Real-application mode: claim, compact, judge, name

Written 2026-09-13 after driving the hosted controller against a real application instead
of the public fixtures. The model was fine. The harness was the bottleneck, in four measured
ways, and this document records the fixes and the design behind them.

## Private setup proof from an app database

For builds without diagnostic log bodies, use `--setup-proof-query proof.json` together with
`--setup-proof-name setup_tier --setup-proof-value premium`. This is an alternative to the
log substring/regex source, not a fallback that accepts stale proof. The operator-authored JSON
contains exactly `database` and `sql`:

```json
{"database":"proof.db","sql":"SELECT tier AS actual, updated_ms AS observed_at_ms FROM state_events WHERE updated_ms >= :since_unix_ms ORDER BY updated_ms DESC LIMIT 1"}
```

The read-only query must project `actual` and `observed_at_ms` (device Unix milliseconds), use
the host-bound `:since_unix_ms`, and have no trailing semicolon. Select the latest relevant event,
including negative/error events; do not filter only the expected state. The host wraps the query
with its own time boundary and exact expected-label allowlist, so only that short enum or null
leaves SQLite. It uses the existing session's `database_query` with `live=true`, `limit=1`; it
does not stop the app, mutate its database, or expose this capability to the controller.

The boundary is a freshly measured device-clock mark before setup. Missing device-clock evidence
means unavailable proof, never a host-time guess. Capture polls briefly for asynchronous results
and runs again before cleanup; failed/latest negative evidence revokes earlier proof. Only
`verified`, allowlisted `actual`, and `source: database` enter the report. SQL arguments, rows,
and database errors are excluded from setup logs. The existing AUA private-database capability
must be supported by the target app/platform; unsupported access leaves the prerequisite unproved.

## What broke on a real application

The judge's native tool schema uses compact entries such as
`{"criterion_index":0,"result":"verified","evidence":"Selected option is visible"}`.
Indexes refer to the authored markdown bullets in zero-based source order. The tool schema never
enumerates the long criterion strings, and the prompt asks the model not to repeat them. Before
combining votes or writing output, the host validates unique in-range integer identities, orders
them by source, and restores the exact authored `criterion` labels. Missing entries become
`not_verified`, never fabricated passes; duplicate, ambiguous or out-of-range identities enter
the bounded repair/fallback path. Legacy exact-label replies remain readable internally, but are
not advertised as the model's output format.

When every applicable provider refuses forced tool choice and the host has relaxed it to `auto`,
a response without a native call may supply exactly one JSON object in `message.content`, either
plain or inside one JSON code fence. The host validates that object against the compact wire
schema before restoring labels. It never extracts JSON from prose or reasoning, accepts multiple
objects/duplicate keys, or overrides a native call with content. Native calls remain preferred;
this recovery is disabled while forced tool choice is active. Sanitized `schema_repair` entries
in `judge-events.jsonl` identify the failing field/constraint without recording the answer text.
An explicit forced-tool capability rejection is a probe, not an answer: the subsequent relaxed
request receives a renewed allowance of up to 45 seconds, capped by the same absolute 90-second
vote deadline. `tool_choice_relaxed` diagnostics record both that allowance and the remaining
vote time. Ordinary schema repairs and transport backoffs do not renew the absolute deadline.

Judge latency is bounded independently of the controller: `Decider` defaults to 45 seconds per
provider/model route (including every transport retry, backoff and schema repair), and 90 seconds
per vote. Each route gets at most its fair share of remaining time divided by remaining routes,
so early stalls cannot starve the last fallback. Responses whose reasoning tokens consume every
completion token (without a native answer) advance immediately, even with `finish_reason: stop`.
Two-vote judging therefore takes at most roughly three minutes of model requests, not
120-second HTTP timeouts multiplied by four transport attempts and repair rounds. A timeout
cancels that transport coroutine, emits a `route_timeout` diagnostic immediately, and advances
the configured ladder while the decision deadline remains. Controller request limits are unchanged.

Reported charges are retained after failure. A timed-out or cancelled request has unknown usage,
not proven zero cost: `cost_complete: false` and `unreported_cost_requests` in the decider report
make that gap explicit, and the run's cost summary is marked incomplete. Graceful cancellation
still writes an unverified result and performs session cleanup; it cannot produce a passing vote.

Judge payloads with a sufficiently large output budget request `reasoning.max_tokens: 2048` and
`exclude: false`, replacing effort on the outgoing judge payload only. Controller profiles and
the comparison manifest remain unchanged. This is a requested budget, not a verified provider
guarantee: [OpenRouter's reasoning documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
states that effort-only models may map a token budget to effort. Wall deadlines and usage-based
fallback therefore remain mandatory. Nonzero `usage.cost_details.upstream_inference_cost` is
counted once when `usage.cost` is zero, including in the spend guard; absent timeout usage remains
unknown rather than a fabricated zero.

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
  --package <app.package> --fresh --apk <build.apk> --record [--activity <cls>] \
  [--prelaunch-setup-flow environment.yaml] \
  [--setup-flow login.yaml] \
  --model or-deepseek-v4-flash-0731-low-deepinfra --map \
  --output <private-dir>
```

Flow: `session_start` deterministically leases a free target or provisions a new one, installs the
APK fresh, and defers product launch → optional recording starts → prelaunch setup flows →
feature flags → `app_launch_and_analyze` with the pinned activity → persona/setup flows →
`run_agent` with compact-v1 tools, hosted-v1 projection,
compaction and both breakers → fresh `analyze_screen` → two independent judge votes with one
evidence-backed result per authored criterion → optional screen naming and route summary →
recording stops → `session_finish allow_incomplete: true` releases the target → `result.json`,
`verdict.md`, `screens.json`, and `route.json`. `--grant-permissions` is explicit rather than a
harness default so guest and limited-access scenarios do not silently receive capabilities their
persona withholds.

`--save-primary-flow` requests a replay candidate, not another acceptance criterion.
If a complete clean controller journal cannot be exported with exact replay scope, the
runner emits a sanitized `primary_flow_export: {status: unavailable, reason:
proof_unavailable}` warning and saves no candidate; the evidence verdict is unchanged.
Missing, corrupt, incomplete or failed execution journals remain execution errors, and
real recording/session cleanup failures still invalidate the run independently.

An `element_not_found` refusal is recoverable only when AUA explicitly says no action was
sent, attaches its current observation, reports zero unknown outcomes and later records a
matching terminal submission. A caller-owned `session_finish` with `claim_recorded=true`
is a claim for the independent judge, not session cleanup. A recovered journal counts only
successful actions and emits an optional-export warning; it never creates a replay candidate.
Possibly dispatched actions, missing terminal claims and failed real cleanup remain strict.

Judge validation may truncate only oversized optional `satisfied`/`unsatisfied` narrative
arrays to their advertised item count, after validating every item (including the discarded
tail). Diagnostics contain field names and counts, never narrative content. Required
`reasons`, verdict/confidence and criterion identities/results/evidence are never clipped;
the per-criterion evidence remains the acceptance authority.

Judges receive the requested compact observation sample (hard cap 32), without the former
silent eight-frame truncation. Selection preserves observed screen families, changed selection
states, checkpoint boundaries and lifecycle epochs. Up to four images prioritize a same-screen
state pair and exclude exact/near duplicate captures only when their observed element state
also agrees. A changed checkmark is never discarded merely because its pixels are similar.
Image-to-evidence-ref/action-step metadata distinguishes post-restart observations from earlier
states. Criteria text never controls the selection algorithm; unavailable evidence stays unknown.

`or-gpt5p6-luna-open` is an independent vision/tool fallback with reasoning disabled, open
throughput routing, `data_collection: deny`, compression off and $0.22/$1.32 per-million
prompt/completion caps. Unexpected native reasoning remains visible (`exclude=false`). The
2026-09-16 saved-evidence audit attached four images and thirteen positioned text observations:
two native votes agreed on ten verified criteria and one unavailable independent system fact
(`not_verified`), in 6.844s/$0.00302065 and 6.179s/$0.00296860 through OpenAI. This validates
transport and evidence handling, not device acceptance. The 90-second vote cap, dynamic fair
route slices and 45-second request ceiling are unchanged.
A second audit of the final selection/provenance prompt repeated that exact criterion agreement
in 6.270s/$0.00301560 and 7.501s/$0.00301755, again with zero repairs/reasoning tokens.

The real-app compact surface adds two navigation actions without widening the fixed fixture
comparison profile. `long_press_and_analyze` accepts only a fresh element id from the current AUA
observation, so pin, unpin and rename menus retain AUA's stale-selector refusal. The separate
`back_gesture_and_analyze` action accepts no arguments: Android derives the left-edge swipe inside
its platform adapter instead of giving the controller arbitrary coordinates.

Long asynchronous UI work is opt-in through `--controller-capability async-ui-wait`. It adds one
small `wait_for_ui_condition` tool: the model supplies exactly one positive semantic `rid:`,
`text:` or `desc:` anchor, temporary pending text that must disappear, and a 1–900 second timeout.
The harness starts `job_start(operation="await")` on the already leased AUA session and polls its
durable job id with `job_status`; polling yields the event loop and makes no model request. The
terminal job result, including its fresh observation and capture evidence, is returned as the one
tool result. A controller timeout or cancellation requests `job_cancel` before normal session
cleanup. Only this virtual tool receives the longer 960-second tool-call ceiling; ordinary model
requests and UI actions retain their configured request timeout, and the whole-run `--time-limit`
still applies.

If provisioning fails before a session exists (for example, host capacity is exhausted), the
runner switches to a bounded wait for an existing lease; the MCP transport timeout expands to
cover that wait. No model chooses a serial, installs or starts the app, controls recording, or
releases the target.

The ordered controller ladder treats a response as usable only after its finish reason and native
tool-call envelope parse successfully. A malformed response advances to the next configured model
before any action from it is dispatched. Valid multiple-call or schema-invalid responses retain
the assistant response plus matched `executed:false` tool feedback for bounded repair; if one model
exhausts that repair budget, the same conversation continues on the next model. A timeout after a
device dispatch remains an unknown outcome and ends the controller instead of falling back or
replaying the action.

`session_finish` is offered to the model with two fields of its own, `outcome`
(`achieved | already_satisfied | blocked | not_achievable`) and a short `note`. The runner
records it as an untrusted claim without releasing the lease. Only the harness's `finally` cleanup
calls AUA `session_finish`; that keeps the device owned through fresh evidence, judgement, and
recording finalization. The claim reaches the judge labelled `controller_claim_untrusted`.

The first user message carries the goal and then whatever `session_start` returned as
`relevant_knowledge`: the accepted knowledge items whose aliases match the goal, at most five,
rendered as `- [kind name] text` and labelled as advice with provenance that may be stale. A
live run once started with three accurate facts in the store (where a setting lives, that a
fresh install overwrites it, the route to it) and re-derived all of them by hand, because the
store was pull-only. Ids stay host-side; `result.json` lists them under `knowledge_shown`. The
judges never see this text: they decide from frames alone.

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
tool, repair, spend stop, id stripping, two-vote agreement, exact per-criterion evidence, namer
cache) and the runner end to end with a fake AUA and a fake model (claim stops the loop without
releasing the device, independent judgement, cleanup failures fail closed, disagreement is
unverified, stall downgrades to warning, map is opt-in, and setup failure still cleans up).
The hosted routing test now asserts `require_parameters` is optional.

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
