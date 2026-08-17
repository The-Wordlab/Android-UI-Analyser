# FunctionGemma experiment log

This is the durable handoff for AUA's small local policy experiment. It records what was built,
what each training cycle proved, what failed, and what the next session should do. Detailed commands
and the public safety boundary remain in [README.md](README.md).

Last updated: 2026-08-17 after the completed v6 RunPod and live-emulator evaluations.

## Objective and non-negotiable boundary

The long-term objective is a fast, inexpensive AUA-native controller that can take over routine
next-action choices while AUA remains the trusted driver. The model receives a short set of exact,
prevalidated calls and returns one opaque candidate ID. It cannot author arguments, authorize risk,
execute an action, declare proof, waive cleanup, or finish an incomplete session.

The public learning corpus is synthetic and app-agnostic. It must not contain private app names,
packages, maps, routes, UI copy, screenshots, journals, typed values, credentials, or user data.
Device observations are evaluation material only unless they pass a separate fail-closed scrubber and
explicit public-corpus review.

## What has been built

- A strict FunctionGemma function-call curriculum with one `select_candidate(candidate_id)` output.
- Opaque candidate IDs, complete candidate-order/ID permutation tests, split isolation, deterministic
  generation, manifest hashes, token-length validation, and public-repository privacy gates.
- LoRA training, resume, checkpoint metadata, provenance, static evaluation, checkpoint selection,
  production-serializer smoke, stateful closed-loop simulation, and live-context smoke runners.
- A packaged optional AUA policy core with deterministic guardrails, lazy MLX loading, authenticated
  local adapter provenance, shadow/advisory rollout caps, CLI/MCP/daemon parity, and no model import or
  load while policy is off.
- A bundled roughly 15 MB v3 LoRA resource; the roughly 543 MB base model remains external. Bundled v3
  is authenticated shadow-only and cannot execute or expose an advisory call.
- A cost-bounded RunPod launcher with a server-side hard deadline, no persistent volume, secret-safe
  transfer, exact-Pod deletion, network-volume audit, and temporary SSH-key inventory restoration.
- Host-only safety matrices for unknown outcomes, cleanup, stale IDs, invalid output, unauthorized or
  redundant choices, and candidate-order/ID bias.

## Iteration history

| Version | Main change | Positive result | Blocking result | Decision |
| --- | --- | --- | --- | --- |
| v1 | Strict parser, opaque IDs, grouped splits, provenance | Established a reproducible end-to-end pipeline | Static test still contained unsafe/redundant errors | Reject |
| v2 | More static task coverage | High static accuracy with zero reported unauthorized/redundant choices | Stateful closed loop failed with repeated observations and early finish | Reject |
| v3 | Runtime-shaped sequences and cleanup semantics | Closed loop 4/4; became the reproducible packaged experiment | Held-out had 1 unauthorized and 1 redundant choice; production smoke only 60/96 | Bundle shadow-only |
| v4 | Production-shaped rows and bounded continuation | Production smoke 96/96; cardinality 2/3/4 families perfect; closed loop 4/4 | Independent test made 4 unauthorized early-finish choices | Reject |
| v5 | Recovery-focused material and longer fresh-base training | Production smoke 96/96 and closed loop 4/4 | Held-out retained 1 unauthorized and 5 redundant choices | Reject |
| v6 | Exact packaged serializer, full semantic permutations, rank-32 LoRA, exhaustive checkpoint selection | One strict-safe checkpoint; untouched test 100%; both smokes 100%; closed loop 4/4 | Real AUA advisory matrix only 1/4, below deterministic AUA's 2/4 | Reject |

The repeated lesson is that aggregate static accuracy, even 100% on an untouched synthetic split, is
not enough. Each new gate has uncovered a different distribution shift: stateful recovery, actual
production serialization, candidate order/ID bias, and finally genuine compiler/UI text.

## V6 authoritative record

### Dataset and training

- Dataset format: `functiongemma-aua-candidate-policy-v6`
- Records: 52,416 train, 9,116 validation, 9,116 test; 70,648 total
- Dataset aggregate SHA256:
  `177022c25c5ec12004b05f415beaf21ac77d0fd0ff1d75a559436374ac357df7`
- Dataset manifest SHA256:
  `d3900c58a698810aa1eb378a6fb51b7b4a997f351b2173b455a046c63ad98364`
- Base snapshot revision:
  `bb327a9ad61044e1496a2bee2365a6b6a6684c72`
- LoRA: rank 32, scale 64, dropout 0.05, learning rate `2e-6`, seed 53
- Training: 8,192 microbatch iterations; checkpoints every 512
- Selected adapter SHA256:
  `5c1b426dd35b9fe3f2cc07c31316d402dce707da4b313e1deea563cc2aa57072`

### RunPod execution

- Run ID: `fg-20260817-153841-063c0a0b`
- GPU: NVIDIA L40S 48 GB at $0.99/hour
- Worker duration components:
  - dependency install: 33.810 seconds
  - model download: 5.156 seconds
  - dataset generation and validation: 398.121 seconds
  - training: 1,613.197 seconds
  - checkpoint selection and static evaluation: 2,771.628 seconds
  - production/live-context/closed-loop gates: 239.029 seconds
- Launcher wall time: 2026-08-17 15:38:43Z to 17:05:44Z, about 87 minutes
- Cleanup: exact Pod deleted in one attempt; zero active Pods; zero network volumes; temporary SSH
  key inventory restored to its exact baseline
- Local ignored artifacts:
  `runs/functiongemma/runpod/fg-20260817-153841-063c0a0b/`

The L40S was sufficient. Training used about 20 GB and took roughly 27 minutes. Evaluation took
longer than training because all 16 checkpoints each generated 9,116 decisions. A more expensive GPU
is not the first optimization; use a diagnostic subset to reject weak checkpoints, cache tokenized
prompts, increase evaluation batch size, then fully evaluate only the best few candidates.

### Selected checkpoint gates

- Validation: 9,108/9,116 = 99.9122%; critical accuracy 100%; parse 100%; unauthorized 0;
  redundant 0; complete permutation groups 8/8
- Untouched test: 9,116/9,116 = 100%; critical 100%; parse 100%; unauthorized 0; redundant 0;
  complete permutation groups 8/8
- Production serializer smoke: 96/96, zero target-ID or target-position bias
- New live-context smoke: 384/384
- Fictional stateful closed loop: 4/4 with cleanup, recovery, safety, and no repeated mutation

Only checkpoint 8,192 passed every validation gate. Earlier checkpoints occasionally approached
99.9% but retained redundant or unauthorized choices or incomplete permutation groups. Longer
training was useful in this run, but only because selection was based on the strict family/safety
metrics rather than final loss or aggregate accuracy.

### Real AUA advisory audit

The live audit used an AUA-owned headless API-36 `Medium_Phone` emulator and the public Android
Settings home. No suggestion was executed. Four rows were offered—Network & internet, Connected
devices, Apps, and Notifications—and the requested target varied across the same four rows.

| Requested | FunctionGemma suggestion | Correct | Deterministic recommendation | Correct |
| --- | --- | ---: | --- | ---: |
| Network & internet | Connected devices | No | Network & internet | Yes |
| Connected devices | Connected devices | Yes | Connected devices | Yes |
| Apps | Network & internet | No | Network & internet | No |
| Notifications | Network & internet | No | Network & internet | No |

- FunctionGemma: 1/4 = 25%
- Deterministic AUA: 2/4 = 50%
- Warm four-session policy batch: 10.7 seconds
- Policy-off four-session batch: 9.26 seconds
- Verdict: v6 showed no real AUA benefit and is not promoted or bundled

The live failure is not evidence that the 270M model cannot learn the task. V6 moved the earlier
production-shaped smoke from failure to 100%, and reached perfect untouched synthetic results. It is
evidence that the current learning material still does not reproduce the exact semantic ambiguity of
real compiler output: title-plus-summary control text, multiple overlapping destination names, and
the way a natural objective distinguishes its requested target from visible alternatives.

Operational lesson: always pin `--serial` before `session start`. `--start-emulator` only boots when
no device is attached; it does not override an existing connected device. The v6 audit detected one
misrouted read-only observation before any tap/input/model execution, then switched to a pinned
AUA-owned emulator. The exact temporary cache and per-device daemon were removed afterward.

### Post-v6 diagnosis and target-aware AUA fix

A source-level review found that both the deterministic current-frame recommendation and the
optional policy compiler compared every visible row with the entire phase objective. The live audit
objective intentionally enumerated all four destinations after naming the requested one, so every
alternative became false relevance evidence. Longer title-plus-summary rows could then accumulate
more shared words and outrank the actual target before the model made a useful decision.

The fix reuses `arrival_destination_terms()` for both lanes. Candidate filtering, deterministic
ranking, and the model-facing `PolicyContext.goal` now receive only the object of the navigation
request. A compound fictional regression proves that enumerated alternatives produce exactly one
deterministic target when unambiguous; a second proves that two rows sharing the requested target
still reach the model, without the unrelated alternative list in its prompt. The focused policy,
goal-session, arrival, and privacy gate passed 80 tests after the change.

The exact four-case public Settings audit was then rerun policy-off on an AUA-owned API-36 emulator.
With one stable Settings-home observation, deterministic AUA recommended the exact requested row in
all four cases: Network & internet, Connected devices, Apps, and Notifications. No model loaded, no
suggestion was exposed, and no row was tapped. This improves the identical deterministic audit from
2/4 to 4/4. The emulator was stopped by the owning session cleanup.

The frozen v6 advisory configuration was then rerun unchanged against the corrected compiler. All
four cases again produced the exact requested call, but now each had one eligible candidate, so AUA
correctly bypassed model construction and inference (`model_used=false`) in 4/4. This is causal
evidence that the AUA change fixed the historical task; it is not evidence that the v6 weights
reasoned better. The earlier v6 live audit remains preserved rather than overwritten.

The first app-launch bootstrap frame was still weaker than the immediately following fresh hierarchy
frame and returned `manual_observation`; this is a separate launch-readback/candidate-recall case to
keep in the independent benchmark. The learned selector must now be evaluated on genuinely ambiguous
target-sharing rows rather than artificially forcing four unrelated candidates.

That bootstrap gap is now fixed for the explicit unstable-readback contract. When `app launch`
withholds `next_actions` and says the folded screen has not stabilized, `session start` consumes one
bounded authoritative hierarchy read before planning. A stable launch still reuses its existing
observation without another capture.

The policy side channel now reports value-free compiler stage counts (`elements`, enabled clickable,
safe control, stable selector, non-destructive, target matched, and offered) plus a boolean stating
whether AUA's deterministic MCP call appeared in the guarded shortlist. It exposes no hierarchy
copy, selector value, package, session id, or typed data.

A separate curriculum-independent host benchmark was added at
`experiments/functiongemma/aua_candidate_benchmark.py`. Its v2 corpus contains 208 fictional cases:
128 policy-eligible taps across four compound-goal forms, eight unrelated destination quartets, and
resource-id/text/description selector representations; 64 deterministic stale/loading/progress/
scroll recoveries; and 16 disabled-target abstentions. Current result: 100% requested-target
extraction, 100% oracle action offered, 100% deterministic action, 100% deterministic recovery, and
100% safe abstention. This is a compiler regression gate, not a substitute for the planned broad
public-emulator benchmark.

The recovery calls deliberately remain AUA-owned rather than learned choices. An explicitly
stale-risk frame forces one uncached hierarchy read; named loading waits for the marker to disappear;
an unlabeled progress frame waits for one hierarchy change; and a target missing from a confirmed
app-owned scrollable surface scrolls exactly one page. Every wait/scroll returns the resulting
analyzed frame. MCP `analyze_screen` now exposes the same `no_cache` refresh control as CLI.

Verification after this AUA pass: the complete FunctionGemma plus public-privacy selection passed
all 122 collected tests; the complete repository suite passed all 2,376 collected tests; repository
Ruff and `git diff --check` passed. Full mypy still reports 21 errors in unrelated source lines,
but none points at the launch refresh, destination extraction, deterministic recovery, MCP refresh,
compiler diagnostics, or benchmark code.

### Small-model comparison candidate: Needle 2

Needle 2 is a separate 45M-parameter, Apache-2.0 tool model with a 14 MB CQ2-bit archive, a
grammar-constrained call format, and a 256-token sliding inference window. Its published LoRA path
uses JAX Metal and reports about 0.71 seconds per training step on an M5 Max. It is therefore a
low-cost Mac-side candidate for the same frozen AUA decision benchmark.

Its role must remain narrow: compact requested target plus 2-4 guarded candidate summaries and an
explicit abstain call. It is not a scenario compiler or whole-test controller. The tuned model also
disables Needle's calibrated confidence head, so AUA's deterministic guard and abstain/escalation
contract remain authoritative. Compare it with FunctionGemma and a 0.6-1.7B challenger on identical
semantic groups, not on independently generated model-specific test sets.

## What to do next

Do not start v7 by merely adding iterations or reinforcement learning. The next bottleneck is the
compiler/context/data boundary.

1. **Instrument candidate recall and target truth.** For each held-out step, record whether the
   oracle action was offered, whether AUA's deterministic target extraction was correct, and whether
   the selector chose the right offered call. The live deterministic 2/4 result proves these are
   separate problems.
2. **Capture the exact production policy context.** Add an explicit opt-in host/emulator recorder at
   the packaged `PolicyContext` boundary, after fail-closed privacy scrubbing. Store value-free,
   app-agnostic training rows—not raw journals, screenshots, hierarchy XML, typed text, or private
   maps. Begin with public Android/system and fictional test apps.
3. **Build a broad live benchmark before more training.** Use at least 100 independent public or
   fictional screen/scenario families, variable 2/3/4 cardinalities, title-plus-summary controls,
   icons, scroll/back/wait/recovery candidates, target paraphrases, and hard distractors. Split by
   screen/scenario/build family, never by row.
4. **Train v7 with SFT plus DAgger-style corrections.** Relabel states where the model or
   deterministic compiler disagrees with an oracle. Oversample premature finish, unknown outcome,
   repeated actions, overlapping destination names, stale selectors, and abstain/escalate decisions.
5. **Keep RL off initially.** The correct candidate is already known, so supervised learning and
   DAgger are more sample-efficient and easier to audit. Consider offline preference/RL only after
   the exact-context SFT plateaus and only with reward derived from AUA proof, cleanup, progress, and
   externally verified task completion.
6. **Optimize iteration time.** Run a small family-balanced diagnostic evaluation on every
   checkpoint, then fully evaluate only the best strict candidates. This should save more time and
   money than moving this 270M model from L40S to H100.
7. **Promote slice-by-slice.** Require 100% parsing/offered-ID enforcement, zero unauthorized,
   destructive, stale, redundant, or premature-finish choices, 100% cleanup, and a live-emulator
   improvement over deterministic AUA before advisory. Autonomous execution remains out of scope.

FunctionGemma 270M remains worth testing as a bounded, resident local selector because it is small,
fast to train, and demonstrably learns targeted serializer failures. It is not yet an autopilot. If
v7 exact-context training still cannot clear the broad live benchmark, compare the same corpus and
gates against a 1–3B controller before investing in a larger general planner.

### V7 supervised-learning contract

V7 keeps the frozen v5 recovery foundation and deliberately drops v6's 576-way expansion of only
forty semantic states. It adds 1,536 independent training states in which two, three, or four safe
controls share a requested destination and differ by candidate-backed summary evidence. Each state
uses only `N²` variants to balance target ID and list position. The resulting corpus contains 58,808
rows: 44,224 train, 7,292 validation, and 7,292 untouched test. Native FunctionGemma validation
passes, the longest row is 744 tokens, split vocabularies are disjoint, and the corpus contains no
journals, maps, screenshots, hierarchy XML, typed input, device reads, or private app material.

The provider manifest now supports authenticated `candidate_counts: [2, 3, 4]` while retaining the
legacy bundled v3 `candidate_count: 4` contract. An untouched 99-case semantic smoke covers all
three cardinalities with balanced target IDs/positions. The unchanged v6 adapter scores 94/99
(94.95%): c2 7/8, c3 26/27, c4 61/64, with 100% parsing and offered-ID enforcement. Its five misses
form the pre-v7 baseline; v7 must reach 99/99 without regressing the strict recovery/static gates.

Training is supervised from the base FunctionGemma checkpoint, not reinforcement learning or a v6
continuation. Three 6,144-iteration rank-32 seeds (61, 67, 71) share the same validated corpus and
strict checkpoint-selection policy. Aggregate loss cannot promote a checkpoint: it must pass exact
parsing, offered-ID, authorization, redundancy, worst-family, untouched test, production smoke,
semantic smoke, and closed-loop gates. RunPod launchers use no network volume, retrieve artifacts
before deletion, and require exact Pod/SSH-key/network-volume cleanup evidence.

## Resume checklist for another session

1. Read this file and [README.md](README.md); do not infer readiness from synthetic accuracy alone.
2. Inspect the ignored v6 artifact and `checkpoint-selection.json`; adapter SHA must match the value
   above.
3. Confirm RunPod has zero Pods/volumes before any new billable run.
4. Preserve the current public privacy gate: `tests/test_no_app_specific_refs.py`.
5. Run the focused gate before editing or training:

   ```bash
   .venv/bin/pytest -q tests/test_functiongemma\*.py tests/test_no_app_specific_refs.py
   .venv/bin/ruff check experiments/functiongemma tests/test_functiongemma\*.py
   .venv/bin/ruff format --check experiments/functiongemma tests/test_functiongemma\*.py
   git diff --check
   ```

6. At this handoff, that focused suite contains 122 tests and passes. The v5/v6 source and tests are
   present in the worktree but are not committed or pushed by this experiment turn.
7. Do not launch another training cycle until the exact-context recorder/benchmark and candidate
   truth metrics exist. More of the current synthetic distribution is unlikely to solve the observed
   live gap.
