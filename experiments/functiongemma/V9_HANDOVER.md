# FunctionGemma V9 data-preparation handover

Status: **V9 curriculum built and first training run executed** (2026-08-18). The historical AUA
data remains curriculum *source material*, used only to choose families and proportions — never
copied into rows. What changed since this file was first written:

- `v9_learning_material.py` emits 22 fictional families across four groups; `v9_curriculum.py`
  renders them through the packaged `policy_messages`/`policy_tools` and counterbalances candidate
  IDs and list positions independently per variant.
- The corpus is 66,000 / 8,400 / 8,400 rows over 6,900 split-exclusive groups, with 14,748 handoff
  targets (17.8%, against V8's 450 / 0.7%). Manifest SHA256
  `d9dbe75c50e8125b870606e6ee4aa9795479777e8e209013c0d731a78890d638`.
- `v9_acceptance_probe.py` scores an adapter on three live-derived shapes under six order/ID
  permutations. The shipped V8 baseline is `tie` 5/6, `leaf` 1/6, `offgoal` **0/6**.
- `v9_portable_adapter.py` rebinds a Pod-trained checkpoint to a local base model and pins
  base/adapter/manifest by SHA-256, closing the V8 portability incident.

The requirements below still stand — in particular the trace work in section 1 and the live
collection matrix in section 3. Static accuracy remains insufficient for promotion.

This handover is written for a fresh Claude/Codex session. Read, in order:

1. repository-root `CLAUDE.md`
2. this file
3. `EXPERIMENT_LOG.md`
4. `history_miner.py` and `tests/test_functiongemma_history_miner.py`
5. `v8_learning_material.py`, `v8_curriculum.py`, and their tests

Do not infer model readiness from static accuracy. Do not open or copy private app maps, screenshots,
or raw UI text into source, prompts, tests, documentation, or training artifacts.

## What is complete

`history_miner.py` joins AUA's append-only command journal to final goal-session state and emits a
privacy-safe structural corpus. It deliberately strips:

- goals and phase objectives
- UI labels, descriptions, resource IDs, selector values, typed values, and URIs
- packages, activities, serials, owners, timestamps, and raw errors
- every physical-device session

It retains only controlled structural fields: command kind, success/failure, returned-observation
counts, phase/proof state, policy status, candidate counts, selected opaque ID, compiler counts,
next-action relationship, immediate progress, and training-use classification.

Manual evidence is never accepted as structured training truth. A model decision becomes a positive
seed only when the exact/same command is followed, the next event advances goal progress, and the
final phase carries non-manual verified proof. A rejected model suggestion becomes a hard-negative
seed only when the alternative action immediately advances a phase with structured proof.

Generated outputs live under ignored `runs/`; never commit them or force-add them.

## Current historical snapshot

Command:

```bash
.venv/bin/python -m experiments.functiongemma.history_miner \
  --output-dir runs/functiongemma/history-v9-prep-20260818 \
  --overwrite
```

Source snapshot:

- 8,003 journal events in 17 files; zero corrupt rows
- 250 session states; 249 emulator episodes emitted and one non-emulator session excluded
- 1,688 journal events correlated to sessions; 6,315 unscoped operational events
- 127 policy decisions, including 67 `selected` decisions
- 556 failed AUA commands in the source snapshot
- 1,118 distinct sensitive source values checked; zero emitted

Sanitized outputs:

| File | Rows | SHA256 |
|---|---:|---|
| `episodes.jsonl` | 249 | `7d36bef3ba62f6c8d3bbbb9f15d03d0f3eb766d59f3220d9c8897a1ec0038d83` |
| `policy_decisions.jsonl` | 127 | `9435761392bdf7a632deba1e1e39a4485d7ceaf6f7ece311e5398170a67f9d28` |
| `curriculum_seeds.jsonl` | 464 | `c1a5f016574a13345271921c591dd5753d3a637957afc029da0ad6149233d447` |

Episode truth:

- 53 finished with every phase marked completed
- 171 terminated incomplete
- 10 finished without typed phases
- 15 still active
- only 12 whole episodes have structured proof for all their completed phases

Seed families:

- 12 structured sequence successes
- 1 model selection followed by immediate structured progress
- 41 handoff/no-candidate cases
- 65 action-failure recovery cases
- 171 terminated-incomplete cases
- 170 finish-called-while-incomplete cases
- 1 incomplete-cleanup case
- 3 unsupported-cardinality cases

These counts are not training-row counts. Several families overlap within one episode. The 170
finish cases are evidence that finish occurred while session truth remained incomplete; they do not
prove the agent/model should have continued, because some sessions correctly terminate after an
unrecoverable failure. They require fictionalization and an independently specified oracle.

## Why the raw logs cannot be training data

AUA logs nearly every public daemon/CLI invocation, but successful observations are slimmed to an
element count and known-screen metadata. The historical journal normally does not contain the full
candidate semantics shown to the policy model. Conversely, short goals, UI text, packages, resource
IDs, and URIs are retained for debugging. Feeding raw JSONL into a model would both leak application
knowledge and create incorrectly labelled examples.

The safe corpus answers:

- which operational patterns are common
- where failures and incomplete sessions cluster
- which sessions have trustworthy structured proof
- when the model ran, what opaque ID it selected, and whether the following command/progress agreed
- which curriculum families deserve more examples

It does **not** answer, by itself, which semantic candidate was correct when the complete candidate
set is absent.

## Required work before V9 training

### 1. Strengthen future trace capture locally

Add an opt-in, ignored, local-only policy-training trace. It should record the exact privacy-screened
`PolicyContext` already sent to the model, all guarded candidates, the selected/handoff ID, the
agent's next action, fresh transition fingerprint, phase-progress delta, proof source, cleanup state,
and final verdict. It must:

- default off
- never enter the normal journal, dashboard, Git, wheel, or telemetry
- exclude typed values, secrets, screenshots, raw XML, and arbitrary app identifiers unless the
  operator explicitly keeps the trace private
- bind every decision to session, phase, frame fingerprint, package boundary, and invocation
- record `followed`, `rejected`, `stale`, `failed`, `proved`, `cleanup_complete`, and `handoff`
- fail closed if a candidate or outcome cannot be correlated

Do not treat the model's own selection or prose as an oracle.

### 2. Convert historical seeds into fictional semantic families

Use `curriculum_seeds.jsonl` only to choose families and proportions. For each accepted seed, author
fictional, app-agnostic semantics using the source-oracle layer demonstrated by
`v8_learning_material.py`. At minimum add:

1. **destination versus breadcrumb leaf** — requesting a bare destination row must beat child
   settings that merely repeat its breadcrumb; derive train/validation variants, but keep the exact
   public `Sound & vibration` live case untouched for evaluation
2. **target absent and explicit handoff** — plausible singular/plural, case, `and`/`&`, substring,
   and related-control distractors; no random action when no candidate advances the bounded goal
3. **failed-action recovery** — fresh observation/replan/handoff instead of mutation replay
4. **incomplete proof or cleanup** — finish is not offered/selected until deterministic session truth
   authorizes it
5. **candidate-recall/cardinality gaps** — 0/1 bypass, 2/3/4 learned cardinalities, and explicit
   handoff; the oracle action must be in the guarded list before selector accuracy is measured
6. **two-hop destination proof** — search-result copy is not arrival; intermediate pages must not
   complete the final phase

Every semantic group must be counterbalanced across opaque candidate IDs and positions. Split by
entire semantic scenario/app/build family before rendering; never row-split paraphrases.

### 3. Collect more trustworthy episodes

The current 12 fully structured successful episodes are far too few. Build a public, reproducible
collection matrix across Android Settings and other stock/test apps:

- at least 500 structured positive episodes total
- at least 100 destination-versus-leaf cases
- at least 100 absent-target/handoff cases
- at least 100 stale/unknown/failure-recovery cases
- at least 100 proof/cleanup/finish cases
- multiple API levels, locales, resolutions, clean states, and changed builds

Use a dedicated emulator only. Disable memory/maps for no-cheating evaluations but keep a unique
frame cache enabled so fingerprint-bound numeric IDs survive the next CLI call. Start with a typed
contract, consume returned observations, carry `phase_done`, and always call `session finish`.
Preserve policy-off and policy-shadow arms from the identical starting screen.

For every collection run require:

- exact initial screen/candidate-set equivalence between paired arms
- fresh destination proof beyond title-only evidence
- no state mutation unless the contract explicitly authorizes it
- session truth and cleanup completion, not agent self-report
- authoritative journal call accounting
- emulator lease/daemon/artifact cleanup

### 4. Build V9 only after the data gates

V9 generation should append new fictional, group-isolated rows to the frozen V8 foundations. It
must render through the exact packaged `policy_messages` and `policy_tools`; do not invent a second
training-only prompt.

Before launching a Pod, require:

- zero privacy violations and zero cross-split semantic overlap
- 100% native tokenizer/parser validation
- exact candidate-ID and position counterbalancing
- oracle-action offered rate at least 99%, reported separately from selector accuracy
- all source groups have an independently checkable oracle
- no manual-evidence row labelled as truth
- the exact V8 live failure and broader public matrix remain untouched evaluation

Then train at least three FunctionGemma seeds and one Qwen3-1.7B capacity challenger on identical
semantic groups. Select checkpoints by zero safety violations plus worst-family accuracy—not loss or
aggregate accuracy. Do not use reinforcement learning until SFT plus DAgger-style disagreement
collection has stopped improving held-out recovery families.

## Evaluation and promotion gates

Static/model gates:

- 100% strict parser and exactly one offered ID/handoff outcome
- zero unauthorized, destructive, redundant, stale, premature-finish, or absent-target actions
- permutation invariance across ID and list position
- worst critical family at least 99.5%, with confidence intervals and raw failure taxonomy

Closed-loop gates:

- thousands of held-out graph episodes with dialogs, loading, wrong routes, stale frames, and unknown
  outcomes
- zero mutation replay while outcome is unknown
- 100% proof and session-owned cleanup
- bounded non-progress detection and deterministic handoff

Live gates:

- at least 100 untouched public emulator cases across multiple scenario families
- paired fresh agents with and without the model from identical state
- model must be invoked (`model_used=true`) in the assisted arm
- no suggestion is executed until the guard revalidates the current frame/session/call
- assisted agents must measurably reduce total wall time, agent turns or AUA calls without reducing
  completion/proof/cleanup rates

Promotion remains shadow -> advisory -> bounded autonomous recovery -> safe navigation. Do not
promote from one easy live row, synthetic accuracy, or a model choosing the same action deterministic
AUA already chose.

## Exact continuation commands

Regenerate the sanitized historical snapshot:

```bash
.venv/bin/python -m experiments.functiongemma.history_miner \
  --output-dir runs/functiongemma/history-v9-prep \
  --overwrite
```

Validate the miner and existing privacy boundary:

```bash
.venv/bin/pytest -q \
  tests/test_functiongemma_history_miner.py \
  tests/test_no_app_specific_refs.py
.venv/bin/ruff check \
  experiments/functiongemma/history_miner.py \
  tests/test_functiongemma_history_miner.py
.venv/bin/ruff format --check \
  experiments/functiongemma/history_miner.py \
  tests/test_functiongemma_history_miner.py
git diff --check
```

Before training, rerun the complete FunctionGemma and public privacy suite and record exact dataset,
model, adapter, tokenizer, configuration, source, and evaluator hashes in `EXPERIMENT_LOG.md`.

## Claude start prompt

> Continue FunctionGemma V9 data preparation from `experiments/functiongemma/V9_HANDOVER.md`.
> Do not train yet. First verify the sanitized history miner and current ignored manifest, then
> implement the opt-in local policy-training trace with fictional privacy tests. Use historical
> seeds only for structural family selection; never copy private app/UI/map data into source or
> training. Build a public controlled collection matrix and prove oracle-action recall before
> rendering any V9 rows. Preserve the frozen V8 live cases as untouched evaluation.
