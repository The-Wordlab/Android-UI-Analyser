# Changelog

All notable, user-facing changes to `aua` (android-ui-analyser) are recorded in this file.

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Every release is a git tag `vX.Y.Z` with a matching GitHub Release carrying that version's
notes, so you can check for a newer version — and read what changed — without pulling `main`.
`aua --version` prints the version you have installed.

## [Unreleased]

### Added

- Model calls take the cheapest route they have a key for. Every hosted model request (controller,
  judge, icon names, grounding) goes through `llm_route`: an OpenAI model is called on OpenAI
  directly when `OPENAI_API_KEY` is set, and through OpenRouter otherwise. A direct answer is priced
  into `usage.cost`, so spend stops and totals keep working. See `docs/models.md`, which also lists
  the one setting that switches each model role.
- `aua config exec --optional NAME` passes a credential when it is set or in the .env file, and
  never asks for it.

- `icon_names` (off by default): a clickable control the app never named gets a short name read
  off its pixels by a hosted vision model, once per distinct icon, kept in one SQLite
  database shared by every AUA run on the machine (`icon_names.db`, default
  `~/.android-ui-analyser/icon-names.db`, never a per-run cache). It lands in `content_desc`
  marked `named_by`, so every reader sees
  "hamburger menu button, opens the side drawer" instead of "unlabelled control, top left".
  Measured on one app: 26% of offered controls were unnamed; the default model named a
  hamburger correctly three times out of three in 0.6-2.6 s for about $0.00005, so the whole
  app costs a few thousandths of a dollar once. A crop the model sees nothing drawn in is
  remembered too, so no screen read waits on the same question twice. Needs
  `OPEN_ROUTER_API_KEY`.
- The app map now remembers each screen's layout tree: what is where, top to bottom, with
  tap/scroll/input/selected marks, repeated rows collapsed and system chrome dropped.
  `aua map --screen <name>` prints it (and `--json` carries it as `layout`); a logical name
  prints one tree per feature-flag context. The dashboard's App map shows the same tree per
  screen.

- Opt-in `connection: existing-chrome` drives one explicitly approved, already-open Chrome tab
  through AUA's semantic UI, screenshot, diagnostics, map and flow surfaces. A bundled Manifest
  V3 extension/native host detaches on disconnect and cannot read browser cookies, history, or
  unapproved tabs; isolated Playwright remains the default.

- Opt-in `aua mcp --tool-profile web` (or `AUA_MCP_TOOL_PROFILE=web`) exposes a focused web
  session/UI/map/flow/browser catalogue and matching agent instructions. The default `full`
  catalogue and shared engine dispatch are unchanged.

- `run_realapp.py --judge-engine typesafe` judges a finished run against its authored contract
  with a TypeSafe System One model (`jev-latest`) instead of the chat-model judge ladder.
  Each contract bullet is scored on a four-level evidence ladder and two
  yes/no questions ask whether something outside the feature stopped the run and whether the
  run ever arrived; the five-way verdict is then composed in ordinary code from those numbers,
  against thresholds a caller can read and change. One run costs one request. Install with the
  new `typesafe` extra and set `TYPESAFE_API_KEY`.

  This judge returns no written justification: a System One model generates no text, so its
  per-criterion evidence names the level a criterion landed on rather than quoting a screen.
  Runs that need a written rationale still want the chat-model judge.

- The System One navigator's journey now records every step the run took, not only the steps the
  navigator itself won. In the default configuration it wins a minority of them, so the model was
  being told it was on step 6 of a run that was on step 12 — and the measurement that justified
  sending a journey at all had been taken on a journey built from every step, which is not what
  shipped. Turns also close before any early return, so a step the navigator could not read no
  longer closes the previous turn against a screen it never saw.

- What an action did is now reported as counts rather than as a guess about their meaning. The
  previous wording asserted a screen was "still working on the last action", which is wrong on a
  toggled switch, where one control changing is the completed action, and wrong on a relabel,
  which AUA reports as `changed` with nothing added or removed — a field that was not being read.

- Element digests, screen fingerprints and pixel bounds no longer reach the model: the numbered
  menu kept them out of the questions while the state body carried them on every element.

- `wait` is now an action the System One navigator can actually take, mapped to
  `wait_and_analyze`. It was offered as an answer and never honoured: choosing it handed the step
  to the controller model, which was then paid to answer the same question about the same screen.
  The repeat guard bounds it without a counter — a wait that leaves the screen identical is the
  same (fingerprint, action) pair, so the second one escalates. `blocked` stays refused, because
  acting on it means ending the run.

- Controls the app never named are described by where they sit ("unlabelled control, top right of
  the screen") instead of falling back to their own 32-character id. Measured over two recorded
  runs, 19 of 176 options handed over — 11% — were raw digests, the exact opaque value the
  numbered menu exists to keep out of the request. Dropping them is not safe: this class of app
  leaves many genuinely pressable controls unnamed.

- Scrolling is now two actions, `scroll_down` and `scroll_up`, instead of one `scroll` action
  plus a separate direction question — the shape the public Jev browser harnesses use. "Which way
  should this screen be scrolled" was a hop of indirection, and gating on the minimum of two
  separate questions' confidences is exactly what jev-1.13's notes warn against. Replayed over 11
  saved screens the merged form chose the same action 11 times out of 11 and asked 2% fewer
  tokens, so the extra question was buying nothing.

- Every System One call's transcript entry now carries the harness's verdict on it: whether the
  proposal was taken, and when it was not, which rule refused it and what the gate needed. A step
  handed to the controller model used to appear in the log as the controller model simply acting,
  with no trace of the refusal that put it there.

- The System One navigator's journey now says what an action actually did instead of whether the
  screen fingerprint moved. "screen changed" was a boolean, so a button losing its label while a
  login was in flight read exactly like arriving somewhere new — and the navigator, told only
  that, pressed the same button again. The frame already carried the difference (same activity,
  2 of 32 controls redrawn, against a wholesale replacement) and compaction was dropping it before
  anyone read it. Replayed on the real step, confidence fell from 0.86 to 0.63 — below the default
  gate, so the step goes to the controller model — while the steps that were genuine progress
  stayed at 0.96 and 0.99.

- Finishing is one of the System One navigator's moves (`achieved`, `already_satisfied`)
  beside the presses, scrolls and `wait`, instead of a separate `outcome` question answered on
  every step. The separate question had no true answer mid-run, so the model was made to pick one
  regardless and its `in_progress` veto then blocked correct finishes; one question, one answer,
  and a finish carries its own outcome and is gated on its own confidence.

- A System One navigator run now writes `controller/system-one-turns.jsonl` beside the chat
  model's own `model-turns.jsonl`: one object per call holding the state sent, the questions
  asked, the raw answers with their probability distributions, the latency, the input tokens and
  the dollar cost. The navigator report gains `usd` alongside `input_tokens`. Without this a run
  could be summarised but never audited.

- `run_realapp.py --nav-action-space full` widens what the System One navigator may answer from
  presses alone to presses, scrolls, back and finishing with an outcome — the shape the public
  Jev browser harnesses use. One request returns the action and every operand it might need
  (which control, which direction, which outcome), so the operands belonging to actions that
  lose cost nothing: a System One request prices its state once and answers in parallel.
  Typing still goes to the controller model in both spaces, because a System One model returns
  a choice and never a string. The default stays `taps`.

  Finishing carries the outcome only and never a note: the note is free text, and a fabricated
  one would reach the judge as evidence of something no screen showed.

- `run_realapp.py --nav-engine typesafe` lets a TypeSafe System One model answer the narrow
  "which control moves toward the goal" steps while the controller model keeps the rest. It is
  gated on the model's own confidence (`--nav-min-confidence`, default 0.80) and proposes taps
  only: stopping, going back, scrolling and typing all stay with the controller model, because
  a wrong stop corrupts a verdict where a wrong tap costs a step. `--nav-shadow` records what it
  would have chosen without letting it act. It also never repeats a tap on a screen that has
  not changed, because a System One model reads each screen from scratch and would otherwise
  loop. Navigation is where a run spends its requests — tens per run against the judge's
  two — so this is the half worth making cheap.

- An observation names what the app asked its own backend since the last one, and what came
  back: `meta.network_calls` carries
  them as `["PUT /v1/profile -> 200", "POST /v1/send -> no answer yet"]`. A
  screen mid-load, a screen whose tap failed, and a screen whose tap did nothing at all are the
  same hierarchy; this says which. Reporting only the *unanswered* calls measured to nothing on a
  real run — that backend answers in 55 ms and AUA settles the screen before handing an
  observation back, so 25 calls happened and none were in the air when anyone looked, while those
  same windows held a `401` and its retry and the `PUT /v1/profile -> 200` that was the change
  under test. Needs
  `aua proxy start` and `network.app_hosts` naming the app's own backend; the key is absent when
  nothing is in the air, when no backend is named, and when no proxy is running, so a quiet
  response pays nothing for it. Only calls started in the last few seconds count, because a chat
  app holds a streamed connection open by design and a signal that is always on is not a signal.

  Naming the backend is not optional tidying. With the proxy running against a real app, every
  call it caught belonged to a vendor SDK — push registration, a Firebase config stream that
  stayed open across three screens, RevenueCat, Facebook, an analytics beacon — and none of them
  hold a screen up. Replayed through a navigator they raised no false wait, but cost confidence
  on every screen that carried one, enough to push two ready screens back to the expensive model
  for nothing. Nothing in a URL tells a backend from a vendor, so the caller says which.

- The run judge is shown what the app asked its backend, and told what it is worth. A frame's
  `meta.network_calls` is the only evidence in a judgement that did not come off the screen, so
  it arrives with a note: host-observed at the proxy, scoped to the app's own backend, and proof
  that a request was sent and answered — never proof that anything rendered. A criterion about
  what a screen shows still needs a frame that shows it. The note is attached only when some
  frame carries the field.

- The System One navigator asks one question instead of two. A press is no longer an action plus
  a separate "which control" answer — each pressable control *is* an action, listed beside the
  actions that operate on nothing. The old shape made the model name a control even when it chose
  to wait, and gated the step on the minimum of two confidences about different things. Measured
  over 60 real screens three times, the merged form is steadier (median confidence 0.54–0.55
  against 0.47–0.48) and acts on the same taps at the same accuracy; it is one question with one
  answer and nothing discarded. Finishing still reads `outcome`, and is still held back by it.

- A System One navigator's journey now quotes each screen instead of counting its controls, and
  names each step in the words the model itself answers in. A turn used to read
  `tap_and_analyze on 'X'` / `7 controls appeared, 2 went away, out of 32`; it now reads
  `press 'X'` / `the screen did not change: Welcome back · [Sign in] · [Browse as a guest]`.
  Counts are facts about a screen the model never sees — they cannot tell a login page from a
  settings list, which is exactly how a model knows it is going in circles — and `tap_and_analyze`
  is AUA's function name for a move the model described as pressing a control. Measured over 200
  saved steps, three times: at the 0.85 gate this went from 19 steps at 74% fidelity to 26 at 83%,
  more coverage *and* more accuracy. It is the largest measured change to this navigator.

- Every judgement now records the request that produced it — the instructions, the question, the
  evidence and the schema — beside the verdict, in `judge/judgements.jsonl`. The log held only
  what came back, so a criterion marked unevidenced could not be told apart from a criterion whose
  evidence never arrived; on a real run it was the second, and the frame that proved the clause
  had been dropped before the judge ever saw it. Screenshots are counted, not stored.

- The System One navigator takes one more look before handing a near miss to the chat model: a
  move scored between 0.60 and the gate waits for the screen once and asks again. The same
  request replayed eight times scored 0.66–0.78 and never crossed 0.80; read again after a wait
  it scored 0.85 every time, because the first frame was a login page still finishing.

- A host action AUA refuses as stale (the screen moved between the read and the press, so
  nothing was sent) is marked `ignored` instead of counted as a tool error and a step: the
  navigator forgets it, the screen is read again, and the same navigator is asked about the
  screen as it is now. The run summary reports `host_ignored_actions`.

### Changed

- Every model role now defaults to GPT-6 Luna: the controller (reasoning off, so it can use an
  OpenAI key directly), the judge (reasoning on; 96% pass/not-pass agreement with DeepSeek V4.1 over
  27 saved rows, no passing row judged failed) and grounding. DeepSeek V4.1 remains the judge's
  first fallback. Icon names stay on DeepSeek V4.1 Flash: Luna named an empty radio button a
  loading indicator every time, which made a settings screen read as stuck.

- The System One navigator reads a scripted goal one step at a time. A brief like "open the
  menu and look, then close it. Send a message and wait for the reply, then open the menu
  again" is cut at its own sequence words with AUA's `goal_phases`; the model is asked about
  the current step, told what is done and what comes after, and a finish answered while steps
  remain means "this step is done" and moves the pointer (one extra question, never a device
  step). Measured on one row: the picks that had to know which phase the run was in sat at
  0.21-0.51 and were all declined; the two that did not were 0.96 and 0.85. A screen with one
  text field now offers one typing line naming that field instead of a tap line and a type
  line that split the vote 0.54/0.46 between two doors to the same room.
- The System One gate now also takes a pick whose own probability clears the gate while its
  confidence sits between 0.60 and the gate. Jev reports both numbers and the confidence runs a
  median 0.03 under the top probability (never more than 0.07 over 399 saved answers); measured
  on 112 aligned steps the picks this admits were three for three right, each of which had cost
  a wait, a re-ask and a chat-model call for the very press it named.
- Maps learned by an older AUA (map schema below 5) retire themselves on first load, on every
  install: screens, routes, contexts and research questions are archived beside the map as
  `index.v4.json` and rebuilt from scratch, while taught knowledge, deeplinks, recipes, notes,
  launch activity and vocabulary are kept. Bump `MEMORY_SCHEMA_VERSION` for any map field
  change; raise `MEMORY_LEARNING_FLOOR` only when old learned data would mislead.

- `run_realapp.py` judges with one neutral vote by default; `--judge-votes 2` still asks the
  neutral and skeptical stances to agree. Over 44 judged rows the second vote agreed 34 times,
  turned two confident fails into `unverified`, and caught one made-up proof frame, for 5–7
  seconds and double the judge cost per row.

### Fixed

- Stopping one emulator no longer shuts down the others. AUA signalled the emulator's whole
  process group, which also holds `netsimd`, the network simulator the host's first emulator
  starts and every later one shares; the other emulators then lost it and exited. Only the
  emulator process is signalled now.

- Harness run totals now include Jev navigation and judgement alongside chat-model spend,
  including known Jev usage when a later step fails or is cancelled. The cost table shows
  Jev separately and labels its input-token pricing as an estimate, not reported billing.
- The System One menu now offers a text field as "Tap the text field 'Ask me anything' so text
  can be typed into it" instead of "Press 'Ask me anything'", and the state carries its `editable` flag, so a goal that says "tap the composer and type" can tell
  the field from the button beside it. Live, every option read "Press '…'" and the only
  "composer" on screen was the attachments button's resource id, which the model pressed
  twice at 0.96 and 0.93; the chat model then paid to close the sheet each time.
- OCR readings that are not text no longer join a hierarchy observation: a single glyph read
  off an icon ("+", "2", ">"), pixels inside the status bar or the keyboard ("| g",
  "ASDFGH"), and a misread of a label the tree already has with an icon glued to its front
  or a moved space ("if. Plana trip" for "Plan a trip"). Measured on one real session,
  every landing screen carried three or four of these and a keyboard screen ten. Text the
  tree cannot see still comes through; that is what the OCR pass is for.
- MCP `session_start` now describes and schema-validates artifact prerequisites: `evidence=all`
  and `junit=true` require a non-empty `artifacts_dir`; default/`failures` and `none` do not.

- MCP action schemas distinguish returned element `id` values such as `el:...` from the app's
  resource/test `rid`, preventing agents from confusing these selector fields.

- Web actions with `until` and bounded waits now use the browser's passive read deadline
  capability, preserving one action and its arrival check instead of failing after mutation.
  Web support requires Playwright 1.63 or later to cancel and drain timed-out reads.

- Session start accepts explicit `evidence: "none"` / `--evidence none` without an artifact
  directory, matching the no-evidence intent across Engine, CLI and MCP.

- Browser CLI controls now use the same warm browser context as semantic UI actions, so
  storage, mocks, offline mode, logs, pages and traces persist across commands. Unavailable
  daemons fail explicitly instead of changing a disposable context.

- Goal-session start and finish keep the browser baseline in that same warm context so
  finishing a CLI session restores the storage and browser controls it started with.

- Long web target URLs use short deterministic daemon socket names within macOS limits,
  retaining per-target isolation and daemon discovery.

- Closing the active popup restores its live opener or another remaining page; closing the
  last page recovers a page on the next operation.

- The controller harness no longer hides an off switch from the model. Frame compaction drops
  flags whose false is merely a default, and `checked` had been swept up with them, so a
  switch that was off looked identical to an element that was no switch at all and a contract
  bullet like "the X switch is off" could never be verified. A `checked` of false is now kept
  (a `checked` of null, meaning no switch, is still dropped), matching what `aua` itself
  reports. Everything else the projection trims is unchanged.

- Frame compaction no longer reads a raw hierarchy dump's `checked: false` on every node as a
  switch, and drops Android's own status bar (`com.android.systemui`) from the model's view of a
  screen. On a real first frame 22 status-bar nodes had become thirteen junk options such as
  "Press '11:28 [switch is OFF]'" beside the one real control.

- The judge's frame sampler keeps a screen that appeared twice when seats are free, ranking
  repeats above blank frames, so a criterion proven on a repeated screen is not left unevidenced.

- A fresh emulator that session start booted itself is no longer refused because of a lease
  record left by a dead run on the same serial. Process-bound leases do not age out while their
  owner lives, and an agent process outlives every emulator it starts, so a record from a run
  whose emulator was stopped 23 hours earlier still read as live; the new boot was refused as
  "leased by <owner>", rolled back, and the caller fell back to waiting. A record acquired
  before a boot on a serial that had no live device is dropped before the claim; one acquired
  after the boot, or on a serial that was online, is still a real holder.

- A boot that session start rolls back no longer stays the session's target: the caller's
  retry used to wait its whole budget for the rolled-back, offline serial while the device it
  wanted had long been free.

## [0.30.0] - 2026-09-19

### Added

- A built-in Playwright-backed `web` platform reuses AUA's semantic `analyze`, stable selectors,
  actions, screenshots, waits, flows, maps and sessions for HTTP(S) pages. DOM test ids and HTML
  ids become `resource_id`; only viewport-intersecting nodes satisfy presence checks.
- `aua browser` and matching MCP tools add browser diagnostics, cookies/web storage and cache
  controls, offline/throttling, scoped CORS, context proxy, HAR record/replay, request mocks,
  session reset, popup/tab/frame inspection, and Playwright traces. URL path/query is now the
  shared map surface, and goal cleanup restores the browser session baseline.

## [0.29.0] - 2026-09-18

### Added

- `aua app uninstall <app-id> --yes` and MCP `app` with `action=uninstall, confirmed=true`
  remove an installed app through the selected platform adapter on Android and iOS.
- iOS simulator app data: SQLite discovery, read-only WAL-aware queries, schema inspection,
  confirmed data mutations and backup/restore; configured feature-flag deeplinks with UserDefaults
  verification; `.plist` preference setup flows with journalled restoration at session cleanup.

### Fixed

- Outcome judges now bind delayed-result criteria to the action that completes the contract,
  such as returning after background work, without accepting unrelated later actions as proof.
- iOS presence checks and `scroll-to` now exclude off-screen accessibility nodes retained by
  SwiftUI, matching the viewport used by `analyze`.
- iOS restart now launches an already-stopped app instead of aborting on simctl's multiline
  "found nothing to terminate" response.
- Automatic cleanup no longer warns about a different platform on every command or reports
  failed cleanup as a successful reset. Pending cleanup remains visible in `teardown status`.

## [0.28.0] - 2026-09-17

### Added

- The experimental real-app controller can opt into `voice-input`, exposing a
  `speak_into_microphone` action that injects synthetic speech while holding or toggling the
  app's record control. Voice scenarios no longer record emulator silence and misreport the
  app's correct "nothing heard" response as a product failure.

- Real-app controller results now warn when the current screen belongs to another package,
  helping the controller return to the app under test without blocking legitimate system flows
  such as document pickers.

### Fixed

- Outcome judges now verify action-result criteria only from the frame produced by that action.
  A later recovery action reaching the expected screen can no longer hide the original failure.

- `aua record stop` no longer fails a recording whose screen never changed at all. A wholly
  static window makes `screenrecord` emit a single frame, so its media length is exactly 0.0s,
  and the aggregate "captured nothing at all" guard fired underneath the per-segment rule that
  already excuses stillness: an 8.02s idle recording exited 3 with `recording_coverage_failed`
  while holding a valid 37 KB MP4. A partly idle window passed and a wholly idle one did not.
  Zero media now fails only where no `static_screen_no_frames` gap accounts for it. No segments,
  a missing supervisor completion, an unreadable or unfinished segment, a non-zero recorder exit,
  and stretches where the encoder was not running all still fail.

## [0.27.5] - 2026-09-16

### Fixed

- Release validation no longer assumes a second passive-read retry can be scheduled inside a
  150 ms wall deadline on a loaded macOS runner; deterministic coverage still verifies retries,
  strict deadline reuse, and that read failures never prove UI absence.

## [0.27.4] - 2026-09-16

### Fixed

- Judge packets group host-observed before/change/return row-order checkpoints with their named
  action steps, retaining recurrent states rather than deduplicating away the return. Callers may
  explicitly request five native screenshots for multi-state contracts; the default remains four.

- Judge evidence retains relative positions for labeled controls and prioritizes captured
  form-validation outcomes. Image selection spreads across the journey even without screen names,
  rather than omitting later observed states; raw handles and pixel bounds remain excluded.

- Predicate waits treat a failed bounded passive UI read as an unconfirmed poll and retry within
  the original deadline. They never reconnect the automation server or infer absence from a read
  failure; cancellation and unsupported-capability errors remain immediate.

- Controller action evidence resolves opaque UI handles against the immediately preceding fresh
  observation. Independent judges receive the observed target labels and evidence reference in
  action order, so named menu actions remain distinguishable after element handles are removed.

## [0.27.3] - 2026-09-16

### Fixed

- Android text-clear recovery no longer multiplies AdbKeyboard clear-broadcast retries through
  reconnects. It allows one semantic refocus of a verified editable field and one accessibility
  clear retry, then requires empty-field proof. Replace-input fallback never clears a second time
  through IME send; changed/unknown focus or an unverified clear fails without fallback typing.

- Public MCP long-press exposes its existing fresh semantic selector path. Real-app compact tap
  and long-press retain ID-first guidance but also allow one observed text/resource-ID/description
  selector and a bounded occurrence index for duplicate rows. Ambiguous selections remain refused;
  coordinates, first-match guessing and selector/ID combinations are not exposed by the controller.

- Finishing an unattended session now retires its registered teardown watchdog after the
  exact owned virtual-target boot stops and before lease release. An old boot's unreplayable
  undo records remain available for deliberate recovery without an orphan polling process;
  unverified watchdog termination remains a cleanup failure. Child watchdogs are also reaped.

## [0.27.2] - 2026-09-16

### Fixed

- Release qualification uses deterministic judge-budget clocks and only announces signal-test
  readiness after cleanup protection is active, removing macOS CI races without changing runtime
  deadlines, cancellation, signal forwarding or cost accounting.

## [0.27.1] - 2026-09-16

### Fixed

- Compact controllers retain target bounds to distinguish small unlabeled controls from enclosing
  headers. Real-app callers can explicitly preserve the authored end state for continuous runs;
  default return-home guidance is unchanged. Captured loading observations remain judge evidence
  without authorizing actions, settled-state claims or checkpoint completion from unsafe selectors.

## [0.27.0] - 2026-09-16

### Fixed

- Compact real-app controllers retain editable-field, resource-ID and window semantics across
  post-action observations, plus input verification/submission status. Sending guidance now
  distinguishes an IME submit action or actual app Send control from keyboard Enter and
  text-selection controls, avoiding draft/retype loops without automatically sending content.

- A confirmed pre-dispatch selector miss can recover without invalidating an evidence-valid
  scenario: it requires AUA's no-action refusal with a fresh observation, zero unknown outcomes
  and a later matching host terminal claim. Such journals never produce a replay candidate.
  Caller-owned terminal claims are not mistaken for failed cleanup; uncertain actions, missing
  claims and actual cleanup failures remain blocking.
- Judge evidence selection now preserves observed screen families, selection/checkpoint
  changes and post-restart provenance instead of silently truncating to eight text frames.
  Four bounded images prioritize a same-screen state pair and omit redundant final captures;
  host-owned image/ref/action mappings make lifecycle evidence attributable.
- Added a price-capped, data-collection-denied open GPT-5.6 Luna vision/tool fallback profile
  with reasoning disabled. A saved-evidence audit validated two consistent structured votes
  while preserving an unavailable independent-system fact as unverified; no device acceptance
  is implied by this audit and vote timeouts remain unchanged.
- Judge replies with oversized optional `satisfied`/`unsatisfied` summaries now retain only
  the schema-bounded prefix, with content-free normalization diagnostics. Every item must
  already be valid; required fields, verdicts and criterion identities/results/evidence
  remain strict and unchanged.
- Optional primary-flow preview/export failures now produce a sanitized QA warning and no
  replay candidate, without rewriting product evidence or falsely marking cleanup failed.
  Corrupt/incomplete execution journals and actual lifecycle cleanup failures remain blocking.
- An explicit forced-tool capability rejection no longer consumes the relaxed answer's time
  allowance. The relaxed request receives up to 45 seconds, still capped by the unchanged
  absolute vote deadline, with renewed-budget diagnostics and normal cancellation handling.
- After a provider forces relaxed tool choice, judges may recover a single strict JSON object
  from message content (optionally one JSON fence), under the same compact schema and evidence
  validation. Prose/ambiguous output is rejected, native calls take precedence, and sanitized
  schema-repair diagnostics explain failures without retaining private model text.
- Judge tool replies identify contract bullets by compact zero-based indexes instead of
  repeating long criterion strings. The host validates identities and restores exact authored
  labels in source order; duplicates/out-of-range values require repair, and omissions stay unverified.
- Reasoning-only judge responses advance immediately even when providers label them normal
  completions. Remaining routes share the vote deadline fairly, and judges request a separate
  2048-token reasoning budget without changing controller profiles. Reported upstream charges
  count toward spend limits and totals when the provider's top-level cost is zero.
- Judge requests now share a 45-second route deadline across transport retries/backoff and schema
  repairs, with a 90-second deadline per vote. Stalled routes advance promptly; cancellation
  preserves cleanup and result artifacts, and requests without usage mark reported cost incomplete.
- Real-app judging keeps routing retries separate from each model's schema-repair budget,
  escalates reasoning-only exhaustion, and normalizes unambiguous criterion formatting while
  leaving missing evidence unverified. Failed judgement spend remains in report totals. Safe
  primary-flow previews may accompany an unverified evidence gap without promoting the verdict.
- The real-app harness supports opt-in private database setup proof through its existing
  read-only database API. A device-clock boundary and SQL-level expected-label allowlist keep
  stale evidence and private fields out of reports; final revalidation revokes obsolete proof.
- A failed provision claim retains its exact boot identity for a same-worker retry. When the
  fallback acquires that same target, unattended session cleanup retires the original boot;
  replacement boots and different worker, owner, or cache scopes are never adopted.
- Explicit teardown discard can archive a lost target's undo records while its own process-bound
  lease is still alive, allowing cleanup and release without reconnecting to the missing target.
  Other owners and sibling worker scopes remain protected.
- The real-app harness can prove setup from a current-run log mark with whitespace-tolerant
  regular expressions and latest-value comparison, without exposing captured log fields. It
  rechecks before cleanup so later state changes cannot inherit an earlier positive proof.
- Opt-in primary-flow export previews the exact clean controller action suffix into the run's
  output directory. Incomplete or unsaveable journals cannot produce a passing export, and
  observation-only runs never borrow setup actions or write global flow memory.
- The real-app harness offers a separate forbidden-foreground-package guard: installed sibling
  apps are allowed, but setup, controller, and final evidence entering a forbidden app aborts the
  run independently of model judgment. The existing installed-package exclusion remains available.

## [0.26.1] - 2026-09-14

### Fixed

- The cross-process lock-order regression test now flushes child-process queue events before it
  signals completion, avoiding a Linux CI race that could block an otherwise valid release.

## [0.26.0] - 2026-09-14

### Added

- `session_finish` can now stop only the exact virtual-target boot created by its session via
  CLI `--stop-started-target` or MCP `retain_started_target=false`. Reused and pre-existing
  targets are never stopped, and an unconfirmed stop keeps cleanup unsuccessful.
- The experimental real-app controller exposes the same opt-in cleanup policy so unattended QA
  harnesses can guarantee that a run does not leave its AUA-started emulator open.

## [0.25.0] - 2026-09-14

### Added

- The experimental real-app controller has an opt-in `async-ui-wait` capability. Its bounded
  `wait_for_ui_condition` tool starts one durable AUA predicate job, polls it in the harness with
  no paid model turns, returns the completed observation and cancels the job if supervision ends.

### Fixed

- Ordered controller fallbacks now reject abnormal or malformed model-response envelopes before
  any device dispatch, and continue on the next model after one rung exhausts protocol/schema
  repairs. Rejected assistant/tool feedback stays in the shared conversation and uncertain device
  actions are never replayed.

## [0.24.1] - 2026-09-14

### Fixed

- `back-gesture-and-analyze` now dispatches through a warm AUA daemon instead of returning
  `unknown_command`; `v0.24.0` did not publish because release CI caught the missing branch.

## [0.24.0] - 2026-09-14

### Added

- `back-gesture-and-analyze` / `back_gesture_and_analyze` performs Android's left-edge back
  gesture through a platform-owned semantic operation; agents never supply swipe coordinates.

### Fixed

- The experimental real-app controller now exposes ID-only `long_press_and_analyze` for context
  menus and the coordinate-free back gesture, while keeping the fixed compact-v1 comparison
  profile unchanged.
- The experimental real-app controller gives only its explicit detached wall-clock wait a longer
  tool deadline, so a 620-second inactivity proof is no longer cancelled by the ordinary
  90-second model/tool request timeout.
- The experimental controller accepts an ordered model fallback ladder for inference failures;
  escalation continues from the existing live evidence and never retries a device call whose
  outcome is unknown.
- `aua config exec` forwards SIGINT, SIGTERM, SIGHUP and SIGQUIT to its configured command and
  waits for orderly cleanup before escalating, preventing an interrupted harness from being
  terminated while it is releasing its session, emulator and lease.

## [0.23.0] - 2026-09-14

### Added

- `aua helper model-run GOAL --checks FILE` runs the experimental DeepSeek V4.1 Flash control
  loop inside the on-device helper with one host handoff, deterministic checks and an ephemeral
  runtime credential.
- `aua job start idle-duration --timeout-ms N` detaches a durable no-device-touch interval. It
  keeps that leased target exclusively guarded, persists reconnectable timing evidence and leaves
  other devices free for parallel work.
- The real-app controller harness can reuse an existing AUA session across related rows, separate
  the session goal from the current row, request headed mode and narrow network, app-lifecycle or
  wall-clock capabilities, and reject forbidden packages before navigation.

### Fixed

- Controller byte and step budgets now preserve the frames already collected for independent
  judgement instead of turning an otherwise usable run into an infrastructure error.
- Visual judges retain the final frame while sampling the whole journey, can fall back when an
  OpenRouter route cannot honor forced tool choice, and receive exact authored criteria without
  treating safe route detours as product failures.
- Best-effort recording failures no longer overwrite a valid product verdict; transient recording
  stop timeouts are retried once, and bare opaque element UUIDs are repaired without rewriting
  labels or stable selectors.

## [0.22.1] - 2026-09-14

### Fixed

- The macOS release suite no longer depends on a low process id being free on the runner. The
  emulator stop tests record a pid above any host's pid_max, so the exit probe a stop performs
  answers "gone" everywhere; v0.22.0's release run failed on this alone and published nothing.

## [0.22.0] - 2026-09-14

### Fixed

- `aua emulator stop` drops the lease of every device it stops - `--mine`, `--owner`, `--avd` and
  `--all` included; only the serial-scoped rollback did before. A lease is bound to the calling
  agent's process, which is an IDE, an agent harness or a reused CI worker and outlives every
  emulator it starts, so the file a clean stop left behind never expired, and the next
  `emulator start --port` on that console port was refused as already in use (#12). A stop now also
  waits for the process to exit before reporting it `stopped`; one that survives the signal is
  listed under `still_running` and keeps its record and lease for the next stop to find.
- A lease holds its emulator's console port only while that console still answers. The
  registry is in the port allocator so a live emulator's port stays safe when adb blinks - but a
  live lease is not a live emulator: bound to a long-lived agent process, a lease for an emulator
  that had died 27 hours earlier still read as live, and `emulator start --port` was refused on a
  port nothing listened on (#12). A running emulator always answers on `127.0.0.1:<port>`; a dead
  one never does. A lease whose record cannot be read still holds its port, and the refusal now
  names the holder and says its emulator is running, so nobody has to open the registry to find
  out why.
- When PATH resolves `emulator` to the SDK's removed `tools/emulator`, the `emulator/emulator`
  beside it is used instead. The legacy launcher hard-codes an Intel QEMU path that no longer
  exists, so an Apple Silicon host whose PATH listed only `tools/` failed to boot any AVD with an
  error about `darwin-x86_64` that AUA never composed (#11). `aua doctor` shows the launcher it
  chose under `emulator bin=`.

## [0.21.1] - 2026-09-14

### Fixed

- The Linux release suite no longer assumes delivery order between messages written to one
  multiprocessing queue by different worker processes. The lock-order regression still proves
  recovery completes without deadlocking before the ordinary worker is released.

## [0.21.0] - 2026-09-14

### Added

- `aua prepare` — the conversation between AUA and the agent that wrote the feature, before any
  device is touched. `prepare start` returns only what AUA cannot work out for itself (the build,
  how to sign in, what the pre-condition is in terms the app stores, how to reach it, UI-only or
  end-to-end, and what must be on screen); questions the app map already answers are not asked.
  `prepare answer` records answers across processes, and the last one writes a proof contract,
  saves the scenario, and returns the run command. `prepare show`, `prepare list`, and
  `prepare run` complete the surface, with matching MCP tools. Every operation except the run is
  lease-free because a conversation about an app should not queue behind a device it does not use
  yet.
- Every generated checkpoint is a literal translation of exactly one answer, and the `provenance`
  beside it quotes what was said. `success`/`repeat` are AUA predicate terms (`rid:hubBadge`,
  `!text:New`), never prose: a generated oracle that looks right and asserts the wrong thing is
  worse than no contract. The emitted YAML is re-parsed by the authored contract schema before it
  is returned, so a generator mistake fails before a device is leased.
- AUA suggests how to reach a pre-condition and shows its arithmetic — datastore, database, flags,
  mock, reinstall or driving the UI, each ranked by reversibility and cost, each carrying what it
  does not prove. `scope=e2e` pushes mocking down the list. Only `seeding=reinstall` wipes an app.
- `aua prepare run <scenario>` runs a prepared scenario. With `controller.enabled` and its API key
  set, AUA drives the whole loop on its own model and the calling agent pays for one question and
  one answer instead of one round trip per tap; otherwise the same contract comes back as commands
  to run yourself. The result always says which, and why the other was unavailable. A controller
  that dies before judging returns `blocked`, never `failed`.
- Every run hands back an `evidence` block: screenshots, video, report and raw call log grouped by
  kind with paths and sizes, so a bundle can be published without re-walking the directory. Device
  and network records are flagged for review rather than dropped, and the payload says plainly
  that AUA has not read them for you.
- New `controller` configuration section (disabled by default) naming the model, judge ladder,
  budget and recording for driven runs.
- `aua prepare discard <id>` and MCP `prepare_discard` drop an interview that was never finished;
  saved scenarios are untouched.
- New-feature preparation is now visible in the generated Claude/Codex skill, brief guide, root
  CLI orientation, and MCP initialization instructions. Goal discovery recognizes natural variants
  such as "newly implemented feature", "appears once", and "stops appearing" instead of requiring
  one exact trigger phrase.

### Fixed

Everything here was found by running `aua prepare` end to end against a real app, and every one of
them cost a complete device run.

- A predicate value containing a comma is one assertion again. `desc:"Create, New"` was split on
  the comma, and the second half asserted a bare word anywhere on screen - a contract that passes
  on the wrong evidence. Quote the value; an unclosed quote is refused.
- Only an answer a previous interview saved may skip a question. Matching every question against
  the goal, at a threshold that accepted one shared word, let a note about theme stand as the
  answer to *where is the build* - which does not add a wrong answer, it removes the question, and
  the run starts with no APK. Merely related knowledge is now shown beside the question as
  `related_knowledge` and never substituted for it.
- The controller's verdict is an object, not a string. Reading it as one raised
  `unhashable type: 'dict'` after a successful install, sign-in, drive and recording had all
  completed - losing a finished result at the last step. Both shapes are read, and the reasons
  come through.
- The controller no longer calls `flags_apply`, which was renamed to `flags_apply_and_analyze`.
  Its usage error arrived as "feature flags not applied and verified", a precondition failure
  wearing a product failure's clothes. A test now checks every tool name the controller mentions
  against what AUA publishes, so the next rename fails in CI rather than on a leased device.
- Each `prepare run` gets its own cache. The cache holds the in-flight screen-recording marker,
  keyed by serial, so a run that died mid-recording blocked the next run on a recycled serial with
  "a screen recording is already in progress". Leases stay host-wide, which is what stops two
  workers driving one device.
- `seeding=reinstall` grants runtime permissions. A fresh install has none, and the system dialog
  that follows looks exactly like a product failure to anything judging the screen.
- A flow's `wait_for` failure now says whether the budget or the app ended it. Every observation
  wait is capped by `perf.max_wait_ms`, so a flow that writes `timeout_ms: 60000` gets ~5s and
  reports `wait_timeout` whether the screen was late or absent - two opposite remedies behind one
  code. The clamp was already on the wait; it now rides out to the failure, beside the
  `resume_from_step` that acts on it.
- A setup flow that ran out of wait budget is re-issued from the step it stopped on, up to four
  times, instead of ending the run. A guest-entry flow whose cold start needed seven seconds
  produced `unverified` and spent a whole device on a healthy app. Any other divergence still
  reports once: repeating a missing element only buys the same answer.
- A setup flow that diverges for any other reason now hands the controller the screen it reached
  instead of ending the run, with a note saying which part of the precondition is still owed. A
  committed flow whose arrival marker had moved returned `unverified` on an app that was plainly
  running. The adaptation comes back on the handback as `adapted` and stands in the verdict, so a
  run that worked around a stale flow is never mistaken for one that did not.
- The contract `prepare` writes is now AUA's own oracle, not just the judge's reading material. It
  was passed as acceptance criteria only, so the session fell back to one phase derived from the
  goal sentence carrying no assertions - `session_finish` could never be accepted, and a passing
  run came back `model_judgement_v1` / `verified: false` with the stronger answer sitting unused in
  the same file. `run_realapp --session-contract` hands it to `session_start`, and `prepare run`
  passes it every time.
- A contract run that proved nothing now says which checkpoint stayed open, and that an assertion
  whose selector the app never publishes can never match. An unsatisfiable contract and a broken
  feature are the same `unverified` otherwise; finding out which took two device runs.
- A setup flow the app has outgrown is now a question rather than a silent workaround. `prepare run`
  returns `flow_repair`: the step that stopped, the screen the app reached instead, the markers that
  screen does publish, and what to do under each answer - AUA asks whether the change was intended,
  because a stale flow and a broken feature look identical from here and only the caller knows.
  It writes nothing: a flow is replayed by every later run, and one rewritten on a guess is worse
  than one that diverges loudly.

## [0.20.0] - 2026-09-14

### Added

- The controller harness can route a hosted model through any provider instead of one pinned
  endpoint. Manifest entries may now set `provider.allow_fallbacks: true` with an optional
  `sort` of `throughput`, `latency` or `price`; the single-provider pin stays valid and stays
  required for benchmark runs, where a number is only attributable to the endpoint that produced
  it. `max_price` is required on both shapes. Two open QA routes ship with the manifest.
- Hosted model requests retry a provider that asks for the request later - 429, 5xx, 408/409/425
  and transport failures - with jittered backoff and the provider's own `Retry-After` when it
  sends a usable one. A 400 or 402 is a real refusal and still fails on the first try. Retries
  are printed and recorded under `provider_retries`.
- `run_realapp --judge-fallback <candidate>` builds a ladder of judge models. Each rung spends
  its full repair budget, then the next model reads the same thread and answers, so a judge that
  cannot produce its own schema no longer ends the run. Repeatable; the ladder, the escalation
  count and the model that actually answered are all recorded.

- `aua session start` returns `relevant_knowledge`: accepted knowledge items whose aliases,
  name or text match the goal, best first, with a warning line pointing at them. Knowledge
  items gain `aliases` (goal phrasings), settable with `aua knowledge add --alias` and the MCP
  `knowledge_add` `aliases` field; `aua knowledge list --query "<goal>"` and MCP
  `knowledge_list` `query` return the same ranked view mid-run. Facts recorded for an app used
  to be pull-only through `aua about`; nothing surfaced them for the goal at hand.
- MCP `session_start` now exposes the existing APK install/fresh bootstrap, optional runtime
  permission grant, and deferred app launch. The CLI exposes the same permission and launch
  controls, so harnesses can acquire or provision a leased target, install the app, and prepare
  recording before the first product launch without agent orchestration.
- Built-in `ios` platform: `aua --platform ios` (or `AUA_PLATFORM=ios`) drives iOS simulators
  through Apple's `simctl` and the AXe accessibility CLI. `analyze`, id-based actions, waits,
  flows and maps work unchanged; elements carry accessibility identifiers as `resource_id`,
  bounds are screenshot pixels, and `key-and-analyze back` performs the iOS back gesture.
  Covers the attached-target profile plus app install/launch/stop/clear/grant, links,
  clipboard and location. Logs, recording and simulator provisioning stay Android-only for
  now. See `docs/ios.md`.

### Changed

- Recording coverage no longer fails a run because the media is shorter than the wall clock.
  `screenrecord` emits a frame when the screen *changes*, so an idle stretch costs duration
  without costing footage: one run recorded 126 frames over 168s with its largest inter-frame
  gaps landing exactly where the controller was waiting on the model. `duration_check` now fails
  only on the two results stillness cannot explain - capturing nothing at all, and stretches
  where the encoder was not running. The shortfall is reported as `coverage_shortfall_s`, a
  `static_screen_no_frames` gap, and `encoder_idle_gaps`.
- Controller and judge token caps are no longer set below what the models use. The controller
  default moves from 4096 to 32768 and the judge from 1024 to 8192, after a run lost nine steps
  of device work to `model completion truncated` and the judge was measured answering at exactly
  its ceiling on nearly every request. `CostGuard` is the real money stop and takes matching
  headroom: `cost_limit_usd` 0.05 to 0.15, `judge_cost_limit_usd` 0.02 to 0.10.

- The real-app controller owns the complete deterministic lifecycle: reuse or provision a target,
  fall back to waiting when provisioning cannot start, bootstrap the app through `session_start`,
  run prelaunch environment setup and verified flags before the pinned product launch, record from
  first launch through judgement, and release the session. MCP transport timeouts now
  cover both sequential lease waits, and authored contract bullets receive independent evidence
  results from both judges with an output budget that scales to the criterion count.

### Fixed

- `.venv` is no longer tracked. It had been committed as a symlink, and `.gitignore` listed
  `.venv/`, which matches a directory - in a linked worktree `.venv` is a symlink, so the rule
  never applied. Pulling it replaced a real virtualenv with a link to its own path. If you
  pulled `main` between those commits, re-run `uv sync --all-extras`.
- An AUA run cache written to `.run/` inside the checkout is ignored. It was enumerated as
  untracked and tripped the app-specific-reference guard on its lane name.

- Recording starts recover same-session failed-start metadata after the target proves no recording
  owner is live, and quarantine legacy host-local recording paths without weakening target cleanup
  identity checks.
- Failed bootstrap on a reused device releases the lease acquired by that attempt. A controller's
  `session_finish` call is now only an untrusted completion claim, so it cannot release the target
  before fresh evidence, judgement, and recording finalization. Recording or session cleanup
  failures invalidate otherwise successful harness verdicts.
- `aua doctor` names discovered targets by their neutral `target_id` when a platform reports no
  Android serial, instead of `?`.

## [0.19.1] - 2026-09-13

### Fixed

- Emulator startup honors an explicitly configured Android SDK ahead of PATH. A launcher that
  exits before connecting now fails promptly with its executable, exit code, and only the
  current attempt's log tail, instead of waiting for the full boot timeout.
- Emulator startup reclaims reservations left by dead starters immediately, including across
  isolated lane caches. Live starters and their booting emulator children stay protected;
  port-conflict hints identify the shared reservation directory.

## [0.19.0] - 2026-09-13

### Breaking

- Default TSV observations append `id_reusable` and `selector` recovery columns. Scripts that
  require the original three columns should request `--fields id,text,clickable` explicitly.

### Fixed

- Explicit running timer/progress labels and native clock widgets no longer invalidate an
  otherwise unchanged sibling control handle. Dialog subjects and item identities stay strict.
- Indistinguishable controls expose `id_reusable: false` and an explicit current-screen selector,
  including an index where needed. Missing-handle observations also offer selectors, so agents
  can recover without repeatedly requesting unusable IDs.
- Failed emulator startup waits for its owned process to exit and escalates when it ignores
  termination. Ownership records and the console-port reservation remain until exit is confirmed.

## [0.18.0] - 2026-09-13

- `aua config exec --env-file PATH --require NAME -- COMMAND ARGS...` resolves required
  credentials and launches the selected command once, continuing automatically after the
  private dialog's Save. Existing process values take precedence; only named dotenv values
  are passed to the child. Cancellation or missing credentials with `--no-prompt` prevents
  launch. Child output and exit status pass through with literal required values redacted.
  A descendant holding output pipes after the direct child exits returns an explicit
  `credential_output_incomplete` error instead of hanging or rerunning the command.

- Credential saves use an OS-held lock that releases on process exit or crash. The persistent
  sibling lock file is reused, including leftovers from older versions, so interrupted saves
  no longer require deleting a stale lock. Finish older-version saves before upgrading.

## [0.17.0] - 2026-09-12

- `aua config secret NAME --env-file PATH` and MCP `credential_request` open a private masked
  Save/Cancel dialog to save an API key or other environment variable. Values stay out of command
  arguments and tool results; existing nonempty values are preserved unless replacement is
  requested. The host-only operation needs no device or session. The generated skills and new
  `docs/credentials.md` tutorial cover setup and loading the resulting dotenv file as data.

- Opt-in shared agent responses normalize CLI and MCP results into one error, observation and
  context envelope, preserving recovery evidence and existing image references. `aua run init`
  saves host configuration for one caller and goal; `aua run exec PATH -- ...` keeps that context
  across commands. MCP enables the same response shape with `configure(agent_response=true)`.
  Legacy output remains unchanged; the wrapper adds no capture, previous-screen substitution or retry.

- Android key validation accepts the existing `paste` and `backspace` runtime aliases through both CLI and MCP; unknown key names still fail before input.

- Missing-target errors preserve explicitly requested full observations, including default state flags, element sources and parent handles, without another UI read. Compact defaults remain unchanged.

- Recording cleanup and recovery verify encoder/supervisor roles as well as owned paths, so processes reading the same footage are neither signalled nor mistaken for active recorders.
- Recording export supports output filesystems without hardlinks by copying to exclusively created destinations; existing files remain protected and failed publication retains remote evidence.

- Multi-segment MP4 export copies video and optional audio while retaining Android non-media tracks in the original segment files, avoiding unsupported metadata-stream export failures.

- Native recording protects the child from hangup before launch, detaches its session, and requires a bounded supervisor readiness acknowledgement before checking capture startup. Targets without `setsid` fail explicitly.

- Recover prior-boot recording metadata under the normal command fence, avoiding a forbidden shared-to-exclusive lock upgrade while retaining old evidence and undo identity.

### Breaking

- External adapters need `ui.read_deadline` for bounded UI waits and the attached-target
  conformance profile. Verified list movement requires real parent links between rows and their
  scroll container; flat trees remain valid for analysis but cannot prove movement or an end.
- Explicit action-and-analyze commands (including microphone commands) reject contradictory
  `--no-observe` before their action callback. Observation flags do not disable screenshot,
  rolling capture, or journal persistence.

### Fixed

- Missing selectors return the observation already read during resolution, allowing recovery
  without a separate analyze call. Text fallback uses its latest complete vision frame.
  MCP preserves existing observation metadata for other errors, including missing handles.
- Detached daemons preserve explicitly unset local policy settings, preventing discovered model
  paths from causing repeated configuration mismatches for saved agent runs or explicit configs.
- Verified scrolling can use two distinct resource-addressed leaf controls translating together
  when clipped text stays fixed. The evidence requires unchanged control sizes and proven list
  ownership; duplicate controls, resizing, and movement on the other axis do not qualify.
- Emulator startup retries a timed-out readiness read while its existing overall boot budget
  remains. It does not restart the emulator or ADB; terminal failure still cleans up only the
  newly started instance.
- Verified scrolling no longer treats a popup or replaced scroll container as successful movement
  merely because its labels changed. Movement remains unverified when the original container
  cannot be identified, including requests to reach the end; the gesture is not repeated.
- UI waits share one deadline across predicate checks and the final observation. Android reads
  stop at that deadline without reconnecting or restarting automation; an unavailable final
  observation is explicit instead of extending an expired wait with more captures. A zero timeout
  keeps its single-probe meaning, with its effective read budget reported as `wait_budget_ms`.
- Microphone injection accepts the emulator's empty successful gRPC response instead of turning
  it into a false uncertain-delivery error. Genuine uncertain delivery still never triggers a replay.
- Microphone injection waits for an active recording input before opening the emulator audio
  stream. Missing readiness returns a typed refusal with an observation and releases an acquired
  hold. Late readiness and foreground changes during the wait cannot authorize injection.
- Microphone MCP calls accept published element handles through `id` or `stable_key` and reject
  conflicting targets or orphan selector modifiers before preparing audio.
- The offline release evaluator counts failed calls and post-finish cleanup from complete session
  journals, checks saved evidence files, and keeps incomplete attempts in the denominator.
  Missing manifests, accounting, or independent verification remain explicit instead of appearing
  as a pass.

- Native recording launches remain valid when the transport appends an exit-status suffix.
  Empty process command lines are ignored only with matching zombie/dead stat evidence;
  ambiguous or live processes still block cleanup. Legacy pending recordings retain their
  actionable cleanup hint.

- Failed bootstrap releases its own lease only after exact-instance cleanup confirms the target
  stopped. Failed or skipped cleanup preserves ownership; foreign leases remain protected.
- Android recording rotates native encoders beyond the 180-second process limit, bounded to
  30 minutes by default. Original segments preserve their timestamps and pauses. Lifecycle,
  rotation gaps, duration shortfalls and unverified coverage are explicit; gapless recording is
  not guaranteed. `record stop PATH` still writes a playable MP4 at PATH and returns it in
  `detail`, using direct copy for one segment or optional host ffmpeg bitstream-copy concat
  for multiple segments. Original segments and a gap/coverage sidecar are retained; the
  export omits uncaptured gaps. Missing ffmpeg or export failure preserves remote evidence
  for retry. Incomplete coverage returns `ok=false` after collecting evidence.
- Recording cleanup verifies directory ownership; failed directory allocation cannot authorize
  deletion of an existing directory. An absent allocation can be safely undone.
- A verified new recording boot quarantines stale metadata and retains its original undo and
  footage, allowing fresh recording without deleting earlier evidence. Ambiguous identity or
  process inspection fails closed; process discovery uses wide output and batched full cmdlines.
- Encoder failure salvages finalized segments into the requested MP4 with failed coverage;
  partial originals and missing/unexported segment metadata remain available. No playable
  segments produces a structured error with retained diagnostics and originals. The bounded
  recording default can use several GB; export retains both originals and concatenated media.
- Missing microphone endpoints point callers to the existing `session start --audio` preflight
  without assuming that missing opt-in was the cause.

## [0.16.2] - 2026-09-08

### Fixed

- Includes the automatic MCP capture, capture-call accounting, and microphone error-evidence
  fixes from the v0.16.1 tag, whose release publication was blocked by a timing-sensitive test.
- Partial-render verification uses a controlled test clock so host scheduling delays cannot
  change the expected settle path. Production polling, settling, and timeout behavior are unchanged.

## [0.16.1] - 2026-09-08

### Fixed

- MCP now starts the same optional rolling capture service as the CLI daemon. Actions expose
  durable capture evidence windows without a separate daemon or screen read. Initialization
  waits for foreground target selection, honors capture configuration and adapter support,
  preserves existing buffers, and completes before the engine closes its device.
- MCP capture status and exports now correlate to the active goal session in the journal;
  their separate capture-buffer session IDs remain intact in results. Session review counts
  these calls instead of silently filtering them out of the goal's totals.

- Microphone errors that carry an observation now publish the same opaque element IDs as
  successful actions. MCP applies the requested observation projection and preserves the
  returned fingerprint and existing image while retaining the uncertain-delivery error;
  it does not repeat the action or run its success-bound wait.

## [0.16.0] - 2026-09-08

### Breaking

- Published element IDs now use persisted `el:` handles instead of frame-local integers or
  selector keys. Treat returned IDs as opaque and pass them back unchanged; do not parse them,
  infer list positions, or keep them across a target reboot. Refresh saved observations when
  upgrading, and finish active sessions before returning to an older runtime. Reusable selectors
  remain available through resource IDs, text and descriptions;
  legacy selector and numeric input compatibility does not make those values durable identities.

### Fixed

- Concurrent emulator provisioning rechecks port occupancy after claiming a reservation, so a
  boot that becomes live during allocation cannot have its port assigned again. Explicit port
  requests also refuse when another process wins the reservation.
- CLI, MCP and session review share one observation contract, separating action execution,
  fresh evidence, returned controls and destination readiness. Empty/transitional reads no
  longer claim settled arrival; local capture exports preserve the prior observation's validity.
  Existing observation screenshots are reused for artifacts and visual inspection.
- Analyzed actions and waits accept consistent `--with-image [PATH]` options and exact command
  corrections reduce CLI discovery retries. Final phase facts can be attached directly to
  `session finish`; generated agent guidance consumes returned evidence before recovery.
- App launch verifies readiness before reporting success, audio setup checks an authenticated
  microphone endpoint before app installation, and device-clock writes require matching readback.
- Invocation accounting includes CLI discovery and parser failures, folds restart sub-operations
  beneath their parent call, and separates confirmed redundant reads from possible navigation
  optimizations. Expected typed-error probes can be declared without poisoning the run verdict.
- Session bootstrap now returns persisted element handles before trimming its observation, matching
  analysis and action responses. Labelled list rows retain handles when an English relative-age
  suffix changes; same-title rows with different ages remain ambiguous and refuse actions.
- Published element IDs are now persisted `el:` handles, separate from reusable selector keys.
  Identified rows and their child controls keep their handles when reordered or scrolled;
  recycled items cannot inherit a previous item's handle. Matching uses full resource names,
  labels and semantic ancestry, excluding input values and interaction flags. Ambiguous or
  unavailable identities are refused with a fresh observation. Records are platform, target,
  boot and adapter-configuration scoped, shared across CLI/MCP/dashboard processes, and bounded.
  Without boot evidence handles remain local to the connected runtime. Legacy selectors and
  numeric scripts remain supported; selector keys with ordinal suffixes still describe positions.

### Added

- Session records and artifact manifests include the starting AUA version. Journal events record
  the producing runtime version, and session reviews identify mixed-version or unversioned events
  so usage comparisons do not attribute older processes to a newly installed release.
- Actions and background wait jobs expose durable `capture_evidence.ref` windows. All capture
  readers accept `--evidence REF` / MCP `evidence_ref`; delayed contact-sheet and GIF exports use
  exactly the same recorded frames without another device read or guessed relative time window.
  Evidence survives rolling-frame pruning and process turnover for up to one hour, within the
  existing per-target byte budget and a 128-window limit. Missing/expired references refuse
  explicitly instead of substituting another action, session or target.

## [0.15.0] - 2026-09-07

### Fixed

- `aua dashboard` no longer opens a uiautomator2 session to draw a tile. The grid used to
  connect to every device for its foreground app every 5 s and for its picture almost every poll,
  each time a fresh session inside the host-wide adb lock - a cold server launch on a freshly
  booted emulator, so the new tile appeared late and everything else waited behind it. It also
  refused to picture a device another agent held, which with no capture frame on disk was a 1x1
  black tile for the whole run - the "one agent works, the rest are black" grid. Tiles now read
  through the new read-only `ui.peek` platform capability (plain `dumpsys window` and
  `screencap` on Android), so a held device shows its real screen and the agent driving it is
  never disturbed. The page also polls one request at a time, and a held tile names its holder.
- Scroll verification now recognizes coherent movement in rows and grids whose unlabelled items all
  fall back to the same Android class name. Repeated items are paired along the requested axis, with
  jitter and cross-axis motion excluded, so a real thumbnail-strip or icon-grid swipe no longer
  reports `already-at-end` solely because every visible item has the same fallback label.
- Recovery retains undos when boot identity or backup evidence is missing, and refuses to overwrite
  unreadable ledgers. One unavailable platform plugin no longer prevents other targets' cleanup.
  MCP exit cleanup stops only its exact owned boot, preserving replacements and handoffs.
- `aua teardown discard` and matching MCP recovery operations provide explicit, audited archival
  of stale undo keys when the original target/configuration is gone. They do not restore device
  state. Corrupt ledger files are reported individually without blocking other targets' cleanup.

### Added

- Platform adapters can declare the read-only `ui.peek` capability (`peek_foreground_app`,
  `peek_screenshot`): a watcher's view of a target that connects no runtime and never disturbs
  an agent's automation session. Android implements it with `dumpsys window` and `screencap`.
- Platform adapter API v1 now exposes a stable `android_ui_analyser.platforms` facade, lazy
  `aua.platforms` entry-point discovery, neutral target/app/geometry/diagnostic contracts, typed
  capability failures, and an executable attached-target conformance profile. The repository gate
  also installs a strict fixture wheel in a fresh process with Android imports blocked.
- Platform plugins can now implement the typed `virtual_targets` service for reusable target
  definitions, provisioning, exact-instance rollback, status, stop, reclaim, create, and delete.
  The new `aua virtual-target ...` and `virtual_target_*` MCP transports share one Engine path;
  existing Android `emulator` commands and payloads remain compatibility aliases.
- Leases, sessions, memory, flows, journals, captures, daemons, dashboards, and teardown records
  are platform/target scoped. Detached workers receive selected plugin options through anonymous
  file descriptors and verify a keyed, non-secret configuration identity before reuse or recovery.
- Persistent target changes now have write-ahead neutral undo registration (or a documented reason
  no truthful undo exists), including orientation, runtime permissions, media, recordings,
  developer settings, helper state, touch capture, automatic helper installation, and temporary
  Android root escalation.

## [0.14.2] - 2026-09-04

### Fixed

- A goal phase that says something is *labelled* missing no longer asks for it to be off the screen.
  "Cook Assistant excludes ingredients the user marked as missing" parsed as
  `expected: "absent"` over its own subject terms, so no fact about the feature working could
  acknowledge the phase — `session finish` refused a run that was finished and validated, and the
  only exit recorded it as `incomplete`. `marked/flagged/labelled as missing` is a label something
  carries, not a claim that it is gone. Real absence claims ("the banner is absent", "the spinner
  is not visible", "no error banner is shown") are unchanged.
- `--by id` now accepts the element id AUA published. A Compose/Flutter input with no resource-id is
  offered as `px:EditText:0005…` or `geo:EditText:q51:…`, and pasting that back under `--by id`
  became a lookup for a resource-id of that name, which cannot exist — while the identical string as
  a bare positional worked. `--by id` still means resource-id for everything else, bare tail
  included.

- A device refused because a *sibling worker* holds it now says so. Per-worker lease scoping made
  the old message read as a contradiction — "emulator-5556 is leased by rec-probe … You are
  `rec-probe`: pass `--owner rec-probe`" — advice that cannot work, leaving `--force` as the only
  route the message offered, against a worker that may still be alive. It now names the run cache
  (`AUA_CACHE__DIR` / `AUA_WORKER_SCOPE`) as the thing that differs. A genuinely different owner
  still gets the old advice.

- `aua record stop` now returns a playable MP4 instead of failing with "the moov box is missing".
  It sent the on-device SIGINT and then immediately signalled its own `adb shell` client, which tore
  the shell session down and killed `screenrecord` while the muxer was still writing the file's index
  — so every recording froze at a 3 KB header, for a four-second clip and a two-minute one alike, and
  a whole QA sweep shipped screenshots in place of video. `record stop` now waits for the device
  process to finish before touching the local client. A second `record start` in the same session
  also works again, and each `record stop` names its own `--remote` path rather than the first one's.

- `aua drive` and `aua helper drive` now answer a goal that only asks what is on screen — "is the
  banner showing", "confirm the row is here", "can you see the badge" — by reporting it found,
  instead of pressing the thing being asked about. Acting on a screen destroys the state the
  question was about, which made the driver unusable for assertions. Measured on 5,741 held-out
  decisions, that class went from 7.6% correct to 100%, and the driver overall from 82.8% to 89.5%.
- `aua drive` and `aua helper drive` now recognise a host capability named by two ordinary words —
  "copy a photo into the gallery", "record a video of the screen", "list the connected devices",
  "check the app logs for errors" — and hand back immediately instead of hunting the screen for a
  button that cannot exist. Neither word of such a pair can be refused on its own without breaking
  navigation to a real destination that uses it. Measured on the same 5,741 decisions, that class
  went from 68.1% to 89.8% and the driver overall from 89.5% to 92.3%, with no other class affected.
  Goals naming the device clock stay unrefusable on purpose: Date & time is a real destination.
- `aua shell dumpsys audio` and `aua shell dumpsys media.audio_flinger` are now allowed. Both are
  read-only state dumps, and they are the only way to prove from the host that a device is actually
  producing sound — the check a "does Pause really stop the audio" test rests on. Refusing them
  pushed that test back to raw `adb`, outside the lease that keeps two agents off one device.
- Two agents running inside one process — parallel subagents of the same coding assistant — no
  longer collapse into a single lease holder. Ownership was derived from the process alone, so
  `aua lease list` told both siblings the same device was `mine:true` and both drove it at once,
  interleaving taps on one screen. Each worker's lease now records the cache it was given, and a
  worker only recognises a lease as its own. A warm daemon holds the lease as the caller that
  started it rather than deriving its own identity, so a caller is never refused its own device.
  A lease written before this shipped carries no such mark and is treated as foreign, so it ages
  out with its process instead of being adopted.
- Naming a device is no longer refused by a lease on a device that no longer exists. A lease
  outlives its emulator — it is bound to the owning process, which for a long-running agent stays
  alive — so once an emulator was stopped, every later `--serial` failed with
  `lease_switch_required` naming a device that was not attached, and pointed at a cleanup for
  state that had gone. Such a lease is now dropped when another device is asked for. Nothing is
  lost: pending undos live in the device ledger, and teardown declines to run them while a live
  holder still owns them, so releasing unblocks that cleanup rather than discarding it. Asking for
  a device that is *also* offline changes nothing, and an unpinned call still keeps its own
  vanished target rather than being rerouted to a stranger's screen.
- A flow whose first step is `stop_app` now runs from wherever the device is sitting, instead of
  being refused because its own app is not already in the foreground. `stop_app` kills the app and
  leaves the launcher, exactly as `clear_data` does — and `clear_data` was already allowed to
  establish a flow's own origin for that reason. Such a flow used to pass only when a previous run
  happened to leave the app in front, and fail from a cold start; one suite has 27 of 54 flows in
  that shape.
- `aua logcat` no longer loses an entire dump to a single byte an app logged that is not valid
  UTF-8. A window that was clean at 60 seconds failed at 300, 900 and 1800 with `can't decode byte
  0xc0`, and `--tag` could not narrow past it because filtering happens after the decode. Undecodable
  bytes are replaced, as `aua shell` already did, so one bad line costs one character rather than
  every other line.

## [0.14.1] - 2026-08-28

### Changed

- The README now collects prerequisites, Claude Code and Codex plugin setup, no-install `uvx`
  usage, permanent CLI setup, verification, and updates into one copy-paste installation guide.
- Version bumps now update and verify the release tags in the README's copy-paste install commands,
  preventing installation help from silently pointing users at an older release.

## [0.14.0] - 2026-08-28

### Added

- Claude Code and Codex plugins now start the AUA MCP server from the matching GitHub release
  through `uvx`. Plugin users need `uv`, `adb`, and a device/emulator, but no clone or permanent
  AUA installation; the generated skill now teaches the MCP-first operating path.
- Any released AUA CLI can be run without installation by passing its Git tag to `uvx`, and the
  repository ships a Codex plugin manifest alongside its existing Claude marketplace.

## [0.13.0] - 2026-08-28

First tagged release. AUA has been developed in the open since 2026-08-20 with no tags, no
releases and no changelog, so this entry is one honest summary of everything the tool does as
of the first tag rather than a reconstruction of the untagged versions it passed through.

### Breaking

- **`elements[].id` is now the stable element identity, and `stable_elements` is gone.** `id`
  used to be a frame-local reading-order ordinal, renumbered on every analyze, with the durable
  id sitting beside it in a second array. The stable id is now published as `id` on every
  surface — elements, `parent`, `next_actions`, `meta.element_diff`, and the `acting` element an
  action reports — so a published id pastes straight back into any command
  (`aua tap-and-analyze rid:continue_btn`); MCP accepts either kind. The shipped agent guidance
  was regenerated to match. **If you read `stable_elements`, or joined two lists to address one
  element, that code must change.**
- **`aua dashboard start` now publishes an mDNS name, binds every interface, and serves with no
  access token.** It used to bind loopback only and require a 43-character token on every network
  path, and it drives the device, streams logcat and queries app databases. The start result
  prints a warning naming exactly what is exposed. Restore the old shape with `--auth` (re-arm the
  token), `--local` (loopback only), `--name ""` (publish nothing) or `--port N`, or move the
  default itself in the new `dashboard` config block. Port 80 is a preference, not a requirement:
  where the kernel refuses it the dashboard falls back to 48765 and says so, while a `--port` you
  pinned is never moved.

### Added

- Versioned releases: every `vX.Y.Z` tag is tested and published as a GitHub Release with wheel,
  source archive, helper APK, and notes from this changelog. `aua update --check` compares the
  installed version with the latest release without pulling `main`; `--json` and exit code 10 make
  the same check usable from automation.
- `aua drive "<goal>"` — hand AUA a goal in plain words and it picks and taps each step itself,
  one host round trip per step (~1500 ms/step). It needs no root, no sideloaded service and no
  permission grant, so it works on retail phones and Play-image emulators. It also stops instead
  of looping: a control that was tapped and changed nothing is remembered per element, and the run
  ends with `no_progress` rather than pressing the same row until the budget runs out.
- `aua helper drive "<goal>"` — the same goal-driving rule running entirely on the device via the
  optional helper APK: observe, choose, act, observe, with no host round trip at all.
- `aua app exists|status|foreground|launch|launch-and-analyze|restart|restart-and-analyze|stop|kill|clear|grant <pkg>`
  for package presence, version and lifecycle, and `aua shell <argv…>` for leased, argv-quoted,
  read-only device diagnostics (each output stream capped at 256 KiB) — neither drops to raw adb,
  and `aua emulator start` plus the app lifecycle calls stream real progress instead of blocking
  silently.
- `aua dashboard start|status|open|qr|stop|run` — the browser dashboard is a persistent service you
  start once, ask the status of, reopen, hand to a phone as a QR code, and stop. With
  `--name aua` it publishes an mDNS record and answers at a typeable `http://aua.local/` from the
  host and from any device on the network, with no privilege and no `/etc/hosts` edit.
- The dashboard opens on a live grid of every device, discovers emulators that appear later, and
  shows each one's lease holder, the owner that started it, the idle-watchdog state and the
  remaining auto-stop time. A device detail view adds the agent I/O journal, logcat, the screen
  map, a database workspace, a proxy panel and local-model control — syntax-coloured and
  filterable, with a fails-only toggle on the journal — and the Lease chip gains an Unlease button
  that runs the same clean-then-release the CLI escape hatch does, so the next agent never inherits
  somebody else's proxy, clock or radio change.
- `aua mock rewrite` — patch a real response (status, headers, whole body, JSON field set/delete,
  literal substitution) instead of only stubbing it away, from the CLI, MCP or the dashboard's
  click-a-request panel, with `--host` and `--times` scoping.
- Every observed action now carries `app_logs`: what the app under test itself logged during that
  action's own window, scoped to its process, priority-filtered and line-budgeted. A crash still
  supersedes it with the fuller `crash_evidence` block.
- `aua logcat prefs show|set|reset --app <pkg>` — say once which log tags matter for an app and
  every later action in every later session honours it: `--ignore-tag`, `--keep-tag` (rescue a tag
  the built-in noise list hides), `--only-tag`, levels, line and per-tag caps. No filter can ever
  drop an `F` line.
- Locale awareness: every analyze reports `meta.device_locale`, `aua devices` lists each device's
  locale, and `aua has` / `wait-and-analyze --for` / `scroll-to-and-analyze` match a query written
  in one language against a device rendering another — reporting which string key and rendering
  matched, and naming the expected rendering on a miss.
- `input-and-analyze --send rid:<control>` types and taps the app's own semantic send control in
  one call, and every input result now reports `submitted` plus a `recommended_call`, so "the text
  was typed" is no longer mistaken for "the app accepted it".
- `aua capture sheet <path> --since last-action --max-frames N --timestamps` — a bounded,
  timestamped PNG contact sheet of the rolling frame buffer, with no ffmpeg on the host.
- `aua session start --animations` (or `--needs animations`, or animation words in the goal)
  enables the device's animation scales for motion and easing checks, and restores their exact
  prior values at `session finish`.
- `aua lease transfer <serial>` / `aua lease accept <token>` / `aua lease cancel-transfer` — hand a
  running device to a child agent without resetting it, via a one-time five-minute token.
- `aua session start` owns device selection: `--needs root,play,proxy,animations` picks a capable
  free target or boots a matching AVD, and when every compatible target is leased by a live agent
  it leaves them alone and provisions a unique read-only instance.
- AUA no longer reports a confident wrong screen. `meta.screen_moved` plus a `WARNING:` note say
  when an overlay, interstitial or dialog arrived between your last observation and this call, with
  `capture_hint` pointing at the frames that show it arriving; and when a tap lands on an activity
  that has started but not rendered, AUA waits for content and returns the real destination — or
  says `stale_risk` with an `arrival` verdict when it cannot.
- Coaching for runs that cannot succeed: a target that is not on screen hands back the screen AUA
  already looked at instead of telling you to go and analyze; a positive `rid:` no mapped screen has
  ever carried is named as impossible with the nearest real ids; three relaunches or the same call
  three times on one target earns a hint; and `session start` warns when the screen a goal is about
  was last seen empty, quoting its own words, and names device changes another run left behind.
- Every action accepts `stable_key` (`--key` on the CLI, `stable_key` on MCP, with optional bounds
  to pick between list rows sharing one key) — the safe way to act on an observation another process
  produced.

### Changed

- `aua db query` no longer stops the app: it copies the database plus WAL through `run-as` and reads
  it host-side, so the screen you were looking at is still there afterwards. **Script-breaking** —
  pass `--coherent` for the old stop-and-relaunch behaviour when you need transactional coherence.
- `aua session finish` returns a compact verdict by default (`--full`, or the returned
  `full_review_call`, for the whole timeline). **Script-breaking** — incomplete closure now exits
  nonzero, but keeps the session and the lease alive and returns the missing checkpoints plus one
  exact next call; `--allow-incomplete` now explicitly means "abandon the goal".
- One leased device is implicit: omit `--serial` from ordinary commands. Switching targets needs the
  warned `aua lease acquire <new> --replace`, which cleans and releases the old device, and a dead
  owner's lease is released immediately.
- The frame is kept by default (`with_image` is on), so `meta.raw_image` always points at a picture
  you can look at when the element tree does not explain itself. **Script-breaking** — `runs/` now
  accumulates frames, pruned to the newest N auto-named frames per device.
- An action result leads with the screen you asked for: `observation` is rendered in its declared
  position rather than appended behind a dozen diagnostics, and `note` sits above it.
- A stable key names exactly one element — colliding keys on a reusable row layout take an ordinal
  suffix, so `rid:row#2` is the second such row down the screen, while a bare key still returns the
  whole group — and selectors now accept the spelling AUA published, so `--rid rid:continue_btn` no
  longer misses, and the same holds for `text:` with `--text` and `desc:` with `--desc`.
- `goto` and arrival proof now treat context variants of one screen as the same screen family when
  the mapped `logical_name`, `state` and `surface` agree; loading shells and modals stay distinct.
- The on-device driver shows 28 actionable nodes per screen instead of 14. 14 truncated 13.2% of 638
  real harvested screens — including fixed bottom navigation bars, which are last in tree order,
  first in importance, and can never be scrolled into view. 28 truncates 2.0%.
- `aua --help` page 1 now lists every command instead of ~55 lines of global options, and an unknown
  command is answered with the real vocabulary instead of a hint naming nothing.
- Helper protocol 1 → 2 on both sides: an older helper APK left on a device is refused with a hint
  rather than answering without the stall fields, where a caller cannot tell "never stalled" from
  "does not report stalls".

### Removed

- `stable_elements` — its content is now `elements[].id`.
- `next_actions` from action responses by default (re-enable with `output.next_actions`); the learned
  per-control cost it carried moved onto the element itself as `Element.cost`.
- `meta.element_diff` from the default action observation — `--observe-meta all` or naming it brings
  it back, and `--format delta` keeps it unconditionally.
- `tier_used`, `via`, `path` and `duration_ms` from the action observation's `meta`, and `rid` from
  the default columns. None of the first four changes the next call and `analyze` still reports all
  of them; `rid` was a restatement of `id` on well over half the rows, `--fields id,rid` returns it,
  and `--where-rid` never depended on it.

### Fixed

- Two agents driving two emulators through the proxy at once no longer read each other's traffic:
  every piece of proxy state — rules, cassette record, flow log, bodies — is keyed by device serial,
  so one agent's rewrite rule cannot fire on the other's device and `mock clear` cannot wipe rules
  its owner never armed.
- The proxy panel no longer serves the app's bearer token, API key, device key or stream token into
  a page people paste into bug reports. Header and field names survive; the values do not.
- A session start can no longer kill or hijack another worker's emulator: port allocation consults
  the host-wide lease registry, a boot detects a serial collision before touching any device, every
  stop path refuses a serial whose live lease belongs to someone else and reports `skipped_leased`,
  and a device-bound warm daemon claims exactly its own device instead of acquiring a different free
  emulator and then refusing to use it — which used to strand a lease for its full TTL on a device
  the caller never touched.
- Parallel session startup, transport recovery and stale Android UiAutomation are fenced and
  recovered once rather than cascading into a failed run.
- The dashboard was broken outright and silently: a stray newline in the page script blanked the
  whole page behind a working header, and clicks coerced the element id to a number — never valid
  for a stable id — so a click on an element the page had just drawn came back as "needs a
  non-negative AUA element id". The ids the page drew are now recorded too, so clicking a box no
  longer validates against whichever screen last wrote the cache.
- Stable ids are published on all four boundaries, not one: the CLI's `--fields` / `--format tsv`
  path, MCP and the daemon used to hand back frame ordinals while the same payload carried stable ids
  elsewhere — one response, two id spaces, naming the same controls.
- `aua input-and-analyze rid:searchField "some text"` was refused with "with --rid/--desc, pass only
  the text to type" — typing by a published id, the whole point of publishing them, did not work.
- `meta.element_diff.changed` crashed the response with an unhashable-dict `TypeError` whenever
  something actually changed between two frames, which is why it surfaced as an intermittent
  `internal_error` on real taps.
- Dashboard grid tiles were pinned to the first frame they ever drew, and a dead capture served its
  last file forever; tiles now key on a frame token that moves with the bytes, fall through to a live
  screencap, and never take the UiAutomation slot from the agent that is driving.
- The dashboard's evidence panels are readable: a scrolled-away journal reader is no longer dragged
  down one row per event (a pill counts what you have not seen and clicking it returns you), the live
  frame column hugs the device instead of taking the wide column, logcat gets a full-width row and
  stops chopping identifiers mid-token, and the copy buttons work on `http://aua.local/` — not a
  browser secure context, so `navigator.clipboard` is simply absent.
- A dashboard-armed proxy rule left no undo record at all — the one mutation the device ledger exists
  to retract was the one it never saw — and `mock map` silently dropped the host and `--times` budget
  it was given, arming a stub against every host, forever.
- Clicking an older proxy exchange showed a newer one's headers and body, because rows were matched
  on a sequence number that restarts with every mitmdump process.
- A malformed `--set` / `--header` / `--replace` printed a Python traceback and exited 1 instead of
  one line of JSON and exit 2.
- Soft lints, screenshots and screen recording no longer bypass the session daemon or connect to the
  device on their own, and a headed session never lands on a physical device.

### Performance

- The screen returned by every action costs 68% less: 919 → 292 tokens on one real settings screen.
  `meta` dropped 16 empty keys and three hints no action asks for, and element rows stopped restating
  `clickable: false`, `enabled: true`, `checked: false` and `selected: false` on every row. `analyze`
  itself is untouched byte for byte, and a switch's `checked: false` — the one flag whose off state is
  the whole reading — always survives.
- A change is reported once rather than three times. On one real tap, change reporting was 473 of 921
  tokens (51%), of which `element_diff.removed` alone was 91% — ids of elements that are gone from the
  screen and cannot be tapped, read or asserted on. What remains answers the same question for a tenth
  of the price, and names the added and removed *text* rather than ids you would have to look up.
- Responses are ~350 tokens lighter with the derived `next_actions` list off by default: it re-listed
  the actionable subset of `elements` and cost more than the whole list it was filtering, while
  silently capping at 12 when 15 controls were clickable. The dashboard now trims what it serves the
  same way — one inspection 3365 B → 817 B, one tap 3498 B → 919 B, `meta` 29 keys → 14 — and what is
  stored is untouched, so every drawn box stays clickable.
- A miss is no longer the most expensive payload the tool emits: on one measured miss, the screen
  attached to "your target is not here" went from 147 rows / ~9200 tokens to 7 rows / ~277, in the row
  shape you already know. A healthy action is also 98 bytes cheaper, because `capture_hint` is attached
  only when something is actually wrong.
- The new evidence is close to free, each measured on one specific action: keeping the frame was
  247.2 ms against 254.7 ms without it (median of five on an emulator, because the screenshot was
  already captured and decoded and `false` only threw it away), `app_logs` costs +16 ms on a ~1050 ms
  action, and waiting for a real destination costs only the calls that were previously returning a
  wrong answer — a settled same-screen tap is 427 ms against 426 ms.

### Notes

- The optional on-device helper APK is **off by default** (`helper.enabled: false`). Turning that one
  switch on does everything: AUA probes rootability, installs the APK and enables the service itself.
  It needs `adb root`, so it cannot be used on retail phones or Play-image emulators — that is what
  `aua drive` is for.
- The published dashboard is unauthenticated on your network by default, and it drives the device,
  streams logcat and queries app databases. `--auth`, `--local`, or `auth: true` in config restores the
  guarded shape.
- The on-device and host driving lanes share one scoring rule (word overlap with a first-token bonus,
  not a language model), measured at 82.2% on 5,741 held-out rows and 17 of 19 reachable destinations
  on a live device. `done` and `no_progress`-style "is X on screen" goals are deliberately unsupported:
  a false "done" is a silent claim of success.
- Time travel (`clock set`) still invalidates auth; always `clock restore`. Use `network_offline`
  (never airplane mode) to prove offline behaviour, and `network_restore` or `session finish` to put it
  back.

[0.13.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.13.0

[0.14.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.14.0

[0.14.1]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.14.1

[0.14.2]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.14.2

[0.15.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.15.0

[0.16.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.16.0

[0.16.1]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.16.1

[0.16.2]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.16.2

[0.17.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.17.0

[0.18.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.18.0

[0.19.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.19.0

[0.19.1]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.19.1

[0.20.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.20.0

[0.21.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.21.0

[0.21.1]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.21.1

[0.22.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.22.0

[0.22.1]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.22.1

[0.23.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.23.0

[0.24.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.24.0

[0.24.1]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.24.1

[0.25.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.25.0

[0.26.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.26.0

[0.26.1]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.26.1

[0.27.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.27.0

[0.27.1]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.27.1

[0.27.2]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.27.2
[0.27.3]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.27.3

[0.27.4]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.27.4

[0.27.5]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.27.5

[0.28.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.28.0

[0.29.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.29.0

[Unreleased]: https://github.com/The-Wordlab/Android-UI-Analyser/compare/v0.30.0...HEAD
[0.30.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.30.0
