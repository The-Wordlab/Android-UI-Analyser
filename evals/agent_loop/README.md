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

`completed` is intentionally stricter than an agent-reported pass: every scenario checkpoint must
be marked passed, cleanup must be verified when required, and the recorded duration must be inside
the scenario limit. `false_pass` is reported only when an independent verifier artifact exists;
missing verification remains `null` instead of being treated as success.

The example campaign defines seven public scenarios. Each scenario may name its `contract` and
`reset_flow`; paths are resolved by the external runner relative to the campaign file. The
offline evaluator deliberately does not execute these files.

Run its tests with:

```bash
python3 -m unittest discover -s evals/agent_loop/tests -v
```
