# AUA agent-loop evaluation

This directory contains a public, fictional Android fixture and an app-agnostic evaluator for
measuring whether an Android testing agent closes the loop. Nothing here depends on private app
copy, selectors, routes, or packages.

## Fixture app

The fixture package is `dev.aua.fixture`. It has six deterministic lanes reachable from its home
screen:

- a classic Android View product grid with name/price sorting;
- an equivalent Compose grid with resource-style test tags;
- an asynchronous loading screen that fails once and succeeds after Retry;
- a local WebView containing a canvas-only Reveal token control;
- the Android notification permission dialog; and
- an in-app reset action that restores fixture data and returns home.

Build it with JDK 17 and an Android SDK containing API 37:

```bash
cd evals/agent_loop/fixture_app
./gradlew :app:assembleDebug
aua install app/build/outputs/apk/debug/app-debug.apk --launch
```

`aua install` replaces a hand-rolled `adb install -r`: it targets the leased device, skips the
push when that version is already installed, verifies the package manager actually registered it,
and with `--launch` returns the screen the fixture opened on. On a fresh emulator the whole
bring-up is one call — `aua emulator start --apk app/build/outputs/apk/debug/app-debug.apk
--launch`.

The fixture intentionally uses visible fictional data. Agents should receive a goal and discover
selectors themselves; benchmark instructions must not reveal expected actions or resource IDs.
The harness, not the agent prompt, passes the matching file from `contracts/` to `session start`.
Those files give AUA a deterministic verifier while the agent still sees only the checkpoint
description and its own observations. `flows/reset-fixture.yaml` is the explicit reset required
for candidate replay/promotion.

## Evaluator

Create a campaign JSON matching `campaign.schema.json`, then run:

```bash
python3 evals/agent_loop/evaluate.py campaign.json --output-dir artifacts/evaluation
```

Each run points at an AUA session bundle containing `result.json`, `manifest.json`, and optionally
`calls.jsonl`. A run may also point at an independent verifier JSON. The evaluator writes
`evaluation.json` and `evaluation.md` with per-run, per-lane, and baseline-versus-candidate metrics.
It never invokes an agent, device, AUA, or app and therefore cannot leak benchmark hints into an
executor.

For current native sessions, the external runner also saves `session.json` (the actual session
state) and `journal.jsonl` (all session-correlated journal events in order). Hydrate journal results
and request arguments from their latest retained detail records before saving; retain failures,
list-returning calls, internal waits and calls after finish. Mark `_detail_hydrated: true` only
when the full exchange was retained and matched; missing proof makes redundant-read counts unknown.
Artifact `calls.jsonl` and manifest
invocation IDs are sparse, so neither is a complete caller log. Run native evaluation with the
project environment (`uv run python evals/agent_loop/evaluate.py ...`) to reuse AUA's existing
session reviewer for caller folding and confirmed redundant reads.

The evaluator checks the retained prefix against the unique terminal finish's saved accounting.
`terminated: true, finished: false` is a finished attempt with an incomplete goal; it stays in the
denominator. Failed nonterminal finish attempts remain calls. Post-finish calls remain in total
cost and are also reported separately. Sparse manifest IDs corroborate retained calls; they are
not expected to equal the complete invocation count. A verified prefix cannot prove that every
post-finish call was retained. Missing native inputs or count mismatches are explicitly unavailable
or incomplete, and suppress normalized call-cost comparisons. A missing bundle also remains an
attempted run with unknown calls. The runner must list every attempted run in the campaign.

Native baseline/candidate comparisons also require matching `execution.json` files with a
`comparison_fingerprint`: the runner's hash of the APK, scenario/profile, reset state, model and
other fixed execution inputs. Exclude the AUA version/lane/run ID being compared. Missing or
different fingerprints suppress call/duration improvement claims. Legacy campaigns without this
metadata remain explicitly `legacy_unverified`; their figures are descriptive, not release proof.

`completed` is intentionally stricter than an agent-reported pass: every scenario checkpoint must
be marked passed, cleanup must be verified when required, and the recorded duration must be inside
the scenario limit. Required evidence IDs must resolve to files inside the bundle. Native cleanup
lists must show successful cleanup operations and a released lease; their saved `result.json`
record is itself the cleanup proof. Merely naming an evidence ID does not prove that its file exists.

A configured independent verifier must explicitly pass before a run counts as completed. Until
reviewed, save `{"status":"pending_review","passed":null,"cleanup_verified":null}`; verifier and
false-pass status stay unknown and the run cannot pass. An explicit failed verifier blocks
completion and can establish a false pass. Omitting the verifier retains the older checkpoint-only
mode, with unknown independent verification. The executor's own narrative is not an independent
verifier. Capture and logical-identity scenarios should have a verifier that checks their actual
saved frames, metadata and target evidence. Keep private app scenarios, reset instructions,
selectors, model/runtime/app-build provenance and evidence outside this public repository.

The example campaign defines seven public scenarios. Each scenario may name its `contract` and
`reset_flow`; paths are resolved by the external runner relative to the campaign file. The
offline evaluator deliberately does not execute these files.

Run its tests with:

```bash
python3 -m unittest discover -s evals/agent_loop/tests -v
```
