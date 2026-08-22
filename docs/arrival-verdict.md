# An honest arrival verdict for the folded observation

Status: **Phases 0–1 implemented** (`Engine._unready_destination_risk`,
`Engine._await_rendered_destination`, `Engine._arrival_report`). Phases 2–3 designed and not
built. This document is the plan; pick it up here.

Every state-changing action folds the post-action screen into `observation` so the caller does
not need a second `analyze`. That promise is the justification for the whole design — measured:
37 taps once produced 73 separate `analyze` calls, because one unfilterable read cost more than
two targeted ones. **A tap that returns a transitional frame recreates exactly that**, and adds
a wrong action on top: the caller picks a target that no longer exists.

## The goal

Remove blind waits and extra function calls from the agent, and make the agent **wait for the
response** instead of composing waits and re-reads itself. Three patterns, worst to best:

```
sleep 2 && analyze                            <- blind: guesses, wrong in both directions
analyze; analyze; analyze                     <- polling: n calls to learn one fact
tap --until 'rid:resultCard,!text:Loading'    <- one call, waits on evidence
```

AUA already has the third **when the caller can name the destination**. This design is for when
it cannot — you tap a login control you have never tapped before, and there is no label to wait
on. The content-wait is what replaces the guess: the action itself holds on, bounded, and
returns either the rendered destination or a verdict that says exactly what is still missing.

## The two wait instruments (why `perf.max_wait_ms` is not a contradiction)

The same codebase caps a caller's 25s wait at 5s while an action patiently holds on for
content. Both are right, because they are different instruments:

- **`perf.max_wait_ms` caps waits the *caller* sizes.** They are open-ended and can be waiting
  for a change that already happened — the measured failure was 42s blocked on a screen that
  had been ready the whole time. No in-call logic can rule that out for a caller-specified
  predicate; returning lets the caller re-decide from a fresh read. The old hint text justified
  the cap with "another function call is cheaper than a blocked session", which the repo's own
  arithmetic contradicts (an LLM re-invoke costs 6–39s of think time); the hint now gives the
  real reason. The cap is anti-*stall*, not economics.
- **`perf.arrival_extension_ms` bounds a wait AUA sizes**, spent only when the classification
  proves the answer in hand is wrong (an unready destination), waiting on *specific missing
  evidence* — content that provably was not there — and exiting on the first sign of it. It is
  exempt from the ceiling for the same reason launch's 5s content poll and provisioning waits
  are: it is not an observation wait the caller composed, it is the action finishing its own
  answer.

Budget: **1200ms default**, sharing the extended confirmation's ceiling (whatever the
confirmation spent comes off the extension), so the worst post-action wall time does not grow
at the default. The owner's 5s figure is the *backstop*, reachable per app via config for
auth-heavy screens — launch already spends up to 5s, so the precedent exists.
`AUA_ARRIVAL_EXTENSION_MS` sweeps it; 0 disables the wait and keeps the honest verdict.

## The failure this exists to fix

A tap started an auth activity that waits on a network round trip before rendering. Recorded:

```
settle            {"ms": 199, "via": "hierarchy"}
stable_delay_ms   250
stale_risk        None                     <- claimed full confidence
change.changed    True
activity_changed  True                     (launch activity -> auth activity)
node_count        43 -> 43                 (identical)
text_added        []                       (the new activity rendered nothing)
text_removed      ["<one label from the old screen>"]
observation       7 rows, all source=hierarchy
```

The caller chose its next target from that frame, tapped a control that no longer existed, and
had to `analyze` to recover: six calls for three calls' work. On the recovery call the CLI
printed its "redundant analyze" warning — **the tool scolded the caller for the call its own
premature settle made necessary.**

## Four things the first diagnosis got wrong

Recorded because each one redirects the fix.

1. **`_tap_settle_needs_confirmation` reads the other way round.** It lists exits that *need*
   extra confirmation (`hierarchy-fast`, `pixels`); plain `hierarchy` is deliberately exempt
   ("slower double-sampled hierarchy settles keep their existing path"). The grade is not being
   ignored — the grade is miscalibrated. `via="hierarchy"` fires on two identical tree dumps
   ~60ms apart, and a frozen transitional window passes that trivially: uiautomator kept serving
   the old activity's 43 nodes while the new one waited on the network.

2. **The pixel check ran and passed truthfully.** `settle["anim"]` is set whenever any grid cell
   was masked; the record has no `anim` key, so nothing was masked — the frame was *genuinely
   pixel-quiet*. The destination was drawn, blank, and waiting. **The screen was physically
   settled and semantically loading.** No settle-loop tuning fixes that; only classification
   does.

3. **`semantic_confirmation` would not have caught it either.**
   `_change_has_semantic_effect` returns True on `activity_changed` **or** `text_removed` alone,
   so this scored as a confirmed semantic arrival. The missing discriminator is asymmetry:
   **removal-only change is departure evidence; additive change is arrival evidence.**
   `_change_summary` already computes `text_added`, `text_removed`, `node_count_delta` and
   `activity_changed` separately — and `text_added` has exactly one consumer, that symmetric
   predicate. The conjunction the case needed was computed, serialised, and never consulted.

4. **`providers_used: null` proved nothing about OCR.** The `changed` observation-meta preset
   strips that key. OCR almost certainly ran. The undiluted legibility signal is the raw OCR box
   count *before* `drop_redundant`, which is currently discarded.

## The machinery already exists, wired to the wrong paths

- `Engine._observation_is_loading` — ProgressBar class, a `loading|please wait` lexicon, mapped
  screen state. Called by `goto`, `back_until`, and action-bound `--until`. **Never by a plain
  tap** — the hot path is the one path that does not check.
- App launch already has a shell-only detector plus a multi-second content poll that swaps in the
  later, better observation. Taps get none of it.
- Memory already models screen state ∈ `loading|error|empty|ready`.
- The rolling capture buffer already bursts at 10fps for ~1.5s after every action mark.

Most of this plan is routing, not invention.

## Phase 0 — shipped

Run `_observation_is_loading` plus the departure-without-arrival conjunction on every folded
observation, but **only where the settle machinery cleared every other caveat**. On a hit: set
`stale_risk`, an honest note recommending an evidence-based wait, and withhold `next_actions`.
Silence the CLI redundant-analyze warning when the previous response admitted uncertainty.

Measured: same-screen tap unchanged within noise; the classifier itself ~26µs.

**Found while implementing, and still open:** `destination_confirmed` is miscalibrated on cold
sessions. It means "recognised name ≠ before name", and with async memory the pre-action
`known_screen` is unstamped — so recognising *anything*, including the origin's own map entry,
counts as a confirmed destination. Phase 0 works around it locally (suppression requires a known
origin). `_stale_observation_risk` trusts the same weak form and should be revisited in Phase 1.

## Phase 1 — the arrival state machine (shipped)

The part that turns an honest warning into a correct answer. This is where the latency budget is
meant to be spent.

**Implemented, with these deviations from the plan below — each learned by building it:**

- **`arrival` is emitted only when there is something to say** — a non-settled state, or a
  settled arrival that had to be waited for. A silently settled action carries no field at all.
  The plan implied an always-present field; the payload-trimming series this landed in teaches
  the opposite lesson: a key that appears only when something is wrong is the cheapest warning.
- **The content-wait skips `app-launch`** (it already has its own, longer content poll) **and
  adopted-baseline observes** (an action-bound `--until` owns its evidence wait; racing it with
  a second wait would double-spend the budget).
- **Poll reads are internal** (`record_ids=False`): no candidate is ever published, so the ids a
  caller holds always match the observation it was handed. The accepted candidate is only a
  trigger — the caller-facing re-read goes through the same `_analyze_post_action` path as the
  first readback (image, vision gate, published ids), and the verdict is re-derived from
  scratch, because the wait itself proves nothing.
- **`destination_confirmed` is fixed globally, not just in the classifier**: it now requires a
  known origin (`Engine._destination_confirmed`). The weak form cleared stale caveats on cold
  sessions by recognising the origin's own map entry.
- **`screen_moved` gained a sixth silence**: a frame already carrying `stale_risk` predicted its
  own replacement, so the replacement is the caller's action landing late — warning "nothing
  you sent caused that" would be a false attribution (see the sibling section below).
- **`via=hierarchy` keeps its confirmation exemption only when the tree grew**
  (`ready.tree_added`): two agreeing dumps of a tree that only lost parts are the recorded
  transitional shape. A settle dict without the count (older tests, monkeypatched settles)
  keeps the old exemption rather than inventing evidence.
- The expiry caveat and note say **how long the call already waited**, or the caller's obvious
  next move is the very wait that just expired.
- **`shell_only_tree` is promoted from Phase 2, because Phase 1 is nearly inert without it.**
  Measured live on a single-Activity (Compose) app — the dominant modern shape — an in-app
  navigation re-entered mid-fetch handed back a destination with nothing readable on it, and
  the verdict stayed silent: `departure_without_arrival` requires an Activity change that
  single-Activity apps never make. A strong subtractive transition landing on a content-bare
  tree is now the same verdict without the Activity evidence.
- **Additive arrival is measured on the app's own elements, never `change.text_added`.** That
  field counts every window; live, the status-bar clock ticking over ("8:15") and the Wi-Fi
  icon returning counted as "content arrived" and vetoed the verdict on exactly the frame that
  needed it.
- **A shell's own chrome is not arrival.** The loading shell's new nav-back control is a new
  actionable control by the additive rule, and it suppressed the verdict live. Bareness is
  therefore checked before additive evidence, in the classifier and in the poll's acceptance.
- **Punctuation is not content.** A page-indicator dot ("•") was a live shell's only text and
  counted as readable content. A label counts only with at least one alphanumeric character
  (`_readable_label`).

### States

| State | Meaning | Caller gets |
|---|---|---|
| `settled` | arrival proven by **content-positive** evidence | full observation, no caveat |
| `no_change` | confirmed same-screen no-op | observation + the existing repeat-mutation caveat, **verbatim** |
| `loading` | stable or animation-masked frame a detector identifies | observation honestly labelled, derived wait call |
| `transitioning` | budget expired while motion continued | observation + stale caveat |
| `unconfirmed` | departure proven, arrival unclassifiable in budget | observation + stale caveat |

### Fields

- `ActionResult.arrival` — `{"state", "evidence": [detector names], "loading": {...}, "waited_ms"}`.
  **Named evidence, never a confidence score** — a number invites trusting the figure over the
  evidence.
- `Meta.arrival_state` on the observation; register it in the `changed` meta preset, the journal
  slim allowlist, and the key list pinned by the slim-payload test.
- `stale_risk` and `note` stay, derived from `arrival.state` — they are the compatibility surface.

### Transitions

```
stable_delay (per-kind, unchanged)
  -> settle scan (existing exits preserved)
       unchanged | hierarchy-same        -> no_change
       pixels | hierarchy-fast | hierarchy -> classify
       timeout                          -> classify, leaning transitioning
  -> classify (free: arithmetic over evidence already in hand)
       content-positive change           -> settled
       loading indicator fired           -> content-wait
       departure-without-arrival,
         shell-only tree, ocr-blank      -> content-wait
       ambiguous                         -> existing confirm window -> reclassify
  -> content-wait (generalise the launch content poll)
       hierarchy-only re-reads, no_cache, ~120-150ms cadence
       additive content appears -> one re-analyze, swap observation -> settled
       budget expires + indicator        -> loading
       budget expires, nothing           -> unconfirmed
```

Two rules keep the budget safe:

- **content-wait and the confirm window share one extension budget**
  (`max(confirmation_cap, arrival_extension_ms)`). Without this a `pixels` exit that then shows
  departure-without-arrival stacks both and blows the ledger.
- **content-wait durations never feed the learned settle profiles** — report as `content_ms`,
  preserving the EMA-poisoning guard.

Also: `via="hierarchy"` loses its confirmation exemption **when the change is not additive**.

## Phase 2 — the remaining signals

Ordered by cost. Every signal is tri-state (*supports / contradicts / unavailable*): absence of a
signal must never masquerade as evidence.

**Free, already collected:** settle exit fields; **mask census** (cell indices are already
computed — derive masked fraction and shape: small centred cluster = spinner-like, full grid =
vacuous idle, distrust `visually_idle`); additive-vs-subtractive change; `known_screen` +
mapped screen state; `_observation_is_loading` on the analyzed tree; app-log digest presence.

**Nearly free (the frame is already decoded):** **OCR legibility count** — raw box count before
`drop_redundant`, pinned to the settle loop's own final frame rather than a buffer frame up to
250ms old. Rule: tree claims N labelled elements, OCR reads ~0 boxes → the pixels do not show
the tree's content → distrust the tree whatever `via` says. macOS-only, hence tri-state.
**Blank-frame statistic** — variance/edge density over the grid signatures already computed.

**Costs latency, spent only on suspicion:** the content-wait poll, and one re-analyze after
content lands.

### Loading detectors

Each contributes one line to `arrival.evidence`:

- `progress_widget` — ProgressBar/Spinner/CircularProgressIndicator class tails. Compose without
  semantics never shows this, which is why no single detector is load-bearing.
- `loading_text` — **one shared lexicon module** replacing three divergent copies today, extended
  across locales *as data*. Matched against tree labels **and** OCR boxes, so canvas/Compose
  loading text the tree cannot see still counts.
- `mapped_loading_screen` — memory state == `loading`.
- `departure_without_arrival` — `activity_changed ∧ no new app-window label ∧ node_count_delta
  ≤ ε ∧ no new clickables`. **This one alone catches the recorded case, at zero cost.**
  (Shipped measuring additions on app elements only — see the deviations above.)
- `shell_only_tree` — **shipped early, in Phase 1** (see the deviations above): a strong
  subtractive transition landing on a content-bare tree, one unlabelled affordance allowed.
- `animated_region` — mask census; gated by conjunction with activity-change/no-content so a
  video on a content screen is not called loading.
- `blank_frame` — near-zero variance + activity change.
- `skeleton_rows` — honestly weak: ≥3 same-class unlabelled non-clickable siblings of uniform
  height under an activity change. Shimmer is the primary residual for Phase 3.

**False-positive guard:** `loading_text` alone on a content-rich additive screen (a settings row
literally named "Loading behavior") must not classify as loading — require conjunction with
departure/shell/blank/widget/mask evidence there.

**Logs are annotation, never a gate.** Three reasons: the digest's tag filters are agent-writable
and persist, so a gate an agent can silently starve with `--ignore-tag` is a footgun; the window
opens pre-action and correlating "request started, no completion" to *this frame* is per-app
guesswork; and a log-driven gate pushes platform semantics into the core against the adapter
boundary. When the verdict is already `loading`/`unconfirmed`, scan the fetched-anyway digest for
generic in-flight patterns and append `logs_show_inflight_request` to the evidence. Logs may
corroborate loading; they may never promote to settled.

### Fusion rule — a decision list, not a score

1. `--until` predicate met → `settled`.
2. Recognised `known_screen` ≠ before ∧ mapped state not `loading` → `settled`.
3. Additive semantic change ∧ no loading indicator → `settled`.
4. Any loading detector fired → content-wait → `loading` if unresolved.
5. Departure-without-arrival / shell-only / ocr-blank → content-wait → `loading` or `unconfirmed`.
6. Confirmed no change → `no_change` with the existing caveat.
7. Cap expiry mid-motion → `transitioning`.

## Phase 3 — a model, only if measurement demands one

**Existing local models: no.** The small selectors are candidate *selectors* under the policy
guard with a recorded history of positional aggregate-accuracy collapse — wrong I/O, wrong risk.
The text-only reviewer cannot see a frame. The grounding VLM is 0.5–2s per call, which busts the
ledger.

**Needle 2: already tried, measured, and dropped** — see `experiments/needle/FINDINGS.md`
(2026-08-18): "positional, not semantic", 5/5 correct became 0/7 by moving the answer in the
list; fine-tuning kills the calibrated confidence head; 256-token window. Nothing here changes
that. Do not re-run it.

**The one candidate that fits** is a purpose-trained tiny frame classifier: final settle frame →
`{content, loading, blank/splash, transition}`. Input the already-decoded frame downscaled
(~224px), optionally a diff-vs-pre-action second channel. MobileNetV3-small class, <10M params,
estimated 10–40ms on CPU/MPS — **must be measured, not assumed.**

- Integration: a perception provider (`providers/` subclass + registry decorator +
  `models.<name>` config block), local weights by absolute path, **never downloaded**. Absence is
  the normal case and the deterministic tiers must be complete without it.
- It runs **only** on ambiguous verdicts, and may only **confirm `loading` or demote toward
  `unconfirmed` — never promote to `settled`.** So a bad model costs latency-honesty, never a
  mis-navigation.
- Training data comes free: the rolling capture buffer plus the journal. Auto-label by what the
  screen *became* — frames between an action mark and the first frame showing additive semantic
  change are `transition/loading`; stable frames after are `content`; near-uniform are `blank`.
- **Evidence-gated.** Ship Phases 1–2, instrument `arrival.evidence` in the journal, and
  commission the model only if the measured residue is material. Its real constituency is
  Compose/Flutter/canvas screens with no a11y and no legible text — which under this design
  already end in an honest `unconfirmed` rather than a wrong answer. That is a job description
  which has not yet been proven non-empty.

## Latency ledger

Baseline today: same-screen tap ≈ 330–370ms. Worst case ≈ 3250ms + analyze.

| Scenario | Today | Designed | Added |
|---|---|---|---|
| already-settled tap | ~350ms | Tier-0/1 arithmetic on collected data | **+0–5ms** |
| the recorded case | ~450ms **wrong**, then a 6-call recovery | content lands → settled; else ~1.65–1.8s `loading` + a ready-made wait call | +0.2–1.35s, only where it was wrong before |
| permanently-animated screen | normal cost, sometimes confident-wrong | detector fires → content-wait expires → `loading` | bounded ≤ ~3050ms, **under today's worst** |

Invariant to enforce and test: post-action observation wall time
≤ `stable_delay + settle_total_max + max(confirmation_cap, arrival_extension_ms)`. With a 1200ms
default extension the worst case does not grow at all. Precedent that this is conservative:
launch's existing content poll already spends up to 5s. And the economics: an LLM caller's
re-invoke costs 6–39s of think time, so ~1.2s spent in-call to prevent one wasted round trip
wins by an order of magnitude.

## Migration

- **Phase 1** (shipped): `arrival` field + state machine; the launch content poll generalised to
  tap-driven transitions; `via="hierarchy"` confirmation for shrink-only changes; budgets in
  config (`perf.arrival_extension_ms` + `AUA_ARRIVAL_EXTENSION_MS`); meta key registered in
  projection, journal and the slim-payload test; `destination_confirmed` made strict.
- **Phase 2**: the remaining signals; unified lexicon; log annotation.
- **Phase 3**: the classifier, evidence-gated.

Tests encoding today's behaviour — change deliberately, not accidentally:
`test_observe_confirms_transitional_taps.py`, `test_settle_partial_render.py`,
`test_post_action_observation_can_be_stale.py`, `test_action_arrival_mismatch.py`,
`test_app_launch_reports_where_it_landed.py`, `test_open_link_reports_unconfirmed_arrival.py`,
`test_observation_payload_is_slim.py`, and the timing-envelope tests.

**Boundary compliance**: every signal flows through surfaces the settle loop already reaches via
the adapter. No new adb anywhere; the classifier is a host-side provider. CLI/MCP/daemon share
`_observe`, so one implementation by construction.

**Risks**: a false `loading` verdict freezes a cautious agent — mitigated because the result never
blocks, always returns the observation plus a *bounded* recommended wait, and content-wait exits
on the first additive change. Content-wait re-analyzes must pass `no_cache`, or the
unchanged-analyze shortcut serves the very payload the caveat warns about.

**Open, not solved here**: the on-device helper settles by its own "three screenshots agree" rule,
so a mid-stretch step could still act on a loading frame. The offload is gated to mapped UI-only
stretches and aborts to host on selector mismatch, which bounds the damage, but it deserves its
own pass.

## What not to do

- **Raise the tap `stable_delay` or the blind settle caps.** The hazard keys on the *outcome* (did
  this gesture start an activity), not the gesture. Taxing every same-screen tap for the few that
  navigate is a documented field regression.
- **Block until loading resolves.** A long ceiling does not buy patience, it buys a stall. Return
  honestly and recommend, as `goto` already does.
- **Emit a confidence score.** Named evidence instead.
- **Use logs as a settle gate.** Agent-writable filters, timing skew, per-app variance.
- **Put any VLM or grounding call on the settle path.** 0.5–6s per call.
- **Revive Needle 2.** Tried, measured, dropped, reasons written down.
- **"Fix" the grid settle by unmasking animation.** Masking is why spinners do not hang the loop.
  Add the mask census as a *signal* instead.
- **Weaken the `no_change` caveat.** The repeat-a-mutation hazard (a second submit, a second
  purchase) outranks arrival optimism.
- **Put per-app loading knowledge in this repo.** Locales as data, yes; app strings never.

## Flagged uncertainties

Classifier and OCR latencies are estimates pending measurement. The claim that uiautomator serves
frozen old-window trees mid-transition is inferred from one record plus code comments, not
re-measured. Compose spinner accessibility exposure varies, which is exactly why no single
detector is load-bearing. The 1200ms extension default is a judgement call, built to be swept.

---

# Sibling problem: the world moving on its own

Everything above answers *"did **my** action arrive?"*. A separate class of failure answers
*"did something arrive that I did not ask for?"* — and settle logic cannot see it, because
nothing the caller did caused it.

Recorded case: a tap landed on a tab, and then a **promotional/onboarding interstitial appeared
by itself**, covering the bottom bar. The caller's next tap named a control the interstitial had
covered. It read like a stale-id mistake and was not: the screen moved between two calls.

The same shape covers push notifications, permission dialogs, session-expiry sheets, and
rate-limit interstitials — none of them caused by the caller, all of them invalidating the
observation the caller is holding.

## 1. The warning already exists and is not delivered

`caller.previous_screen_gone` (+ `previous_screen_age_ms`) is computed from the fingerprint the
adapter stamps at the end of each turn (`Engine.close_caller_turn`, `_previous_screen_gone`), and
`guide.py` already instructs agents to read it. But `caller` is **not** in the `changed`
observation-meta preset, so a folded observation never carries it.

**Delivered — but not by restoring `caller`.** The whole block measured 182-199 B on every
action, most of it the telemetry this series removed on purpose, and `previous_screen_gone`
itself is the wrong comparison for an *action*: it compares the fingerprint of the screen just
emitted, so a tap that navigates always sets it. Measured live: an ordinary back-to-back tap
that changed nothing but one label reported `previous_screen_gone: true`, and a tap taken while
a real overlay had arrived on its own reported `false`.

So `Meta.screen_moved` carries the signal instead (`projection.OBSERVATION_META_PRESETS`
`changed`), decided by `Engine._screen_moved_verdict` on the **pre-action resolution read** —
the only read that happens before the device is touched, which is what makes "nothing you sent
caused this" true. It fires only when the set of *actionable* stable keys differs between the
screen published to the caller and the live one: three consecutive `analyze` calls on one live
screen with nobody touching the device gave node counts 43, 43, 44 and two different
fingerprints with the same nine actionable ids, so a fingerprint-keyed warning would have fired
on almost every call. It is also silent on the first action of a session and on a gap
`caller_latency` classifies as `idle`. `None`-stripped, so an unmoved screen pays nothing; when
it fires the reason is also prepended to the action `note` (and folded into the miss `hint`),
because a field a caller must remember to read is weaker than a sentence where it is already
reading.

## 2. An error should carry the observation it already took

The stable-key miss reads:

```
{"ok": false, "error": {"code": "element_not_found",
  "hint": "No action was sent. … re-analyze and use an id from that fresh observation …"}}
```

`_resolve_action_key` **already analyzed the current screen** to discover the key was absent, then
discards that observation and tells the caller to go and fetch it. That is one guaranteed wasted
round trip per miss, on the exact path where the caller is already confused.

The mechanism exists: `SelectorNotFoundError` takes an `observation` and attaches it.
`ElementNotFoundError` does not. Give it the same treatment — the resolution read is already
paid for. Same for the stale-id path, which also re-reads before raising.

While there: the hint should say *what the screen is now*, not just "re-analyze". A miss plus a
fresh observation plus "the screen changed under you" is a recoverable answer; a miss plus an
instruction is a round trip.

## 3. Point at the frame history

The rolling capture buffer already keeps deduped frames with diff summaries
(`capture_sidecar.py`, `aua capture last --since last-action`), and `meta.capture_hint` already
names the exact command. `capture_hint` is also **not** in the `changed` preset.

On an error, or on any non-settled arrival verdict, the frame history is the one artefact that
shows *what happened in between* — an interstitial sliding in, a dialog appearing, a screen
replaced twice. Surface `capture_hint` on those responses specifically, rather than on every
successful action where it was pure cost.

**Delivered**, and the gate turned out to be needed in the other direction too: the top-level
`ActionResult.capture_hint` was still being attached to *every* observed action whenever the
buffer was live — 98 B for a pointer to a buffer nothing was wrong with.
`Engine._frame_history_matters` now gates it on the three verdicts that ask the question the
frames answer (`not ok`, `stale_risk`/`settled_unmet`, an empty observation), and
`Engine._miss_observation` keeps `capture_hint` through the shared observation budget that
drops it on the healthy path — a miss has no `ActionResult` to hang it on.

## Why these three belong together

They are one story: **the caller's picture of the screen can be wrong for reasons that are not
its fault, and AUA usually already knows.** Each fix delivers information the tool has already
computed, so all three are close to free:

| | already computed | delivered |
|---|---|---|
| the screen moved on its own | yes, per turn | yes — `meta.screen_moved` + the action `note` |
| observation on a key miss | yes, to detect the miss | yes — `errors._CarriesObservation` |
| `capture_hint` / frame history | yes, always running | yes — on a miss / an unsettled arrival only |

Two of the three were removed by the payload-trimming work in this same series, which is the
lesson worth keeping: **trimming by "is this usually empty" removed fields whose whole value is
that they are usually empty.** A key that appears only when something went wrong is the cheapest
possible warning, and the easiest to mistake for noise.
