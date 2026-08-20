# FunctionGemma candidate-selection experiment

This experiment tests whether a small FunctionGemma model can choose the next action in an AUA
workflow without being allowed to author or execute Android commands. Training and primary gates are
host-only; separately identified advisory audits may observe suggestions on an AUA-owned emulator.

For the chronological evidence, failed hypotheses, exact v6 handoff, and next-step decision record,
read [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Safety boundary

The model is given a short list of deterministic, prevalidated candidates. Each candidate has an
opaque integer ID, an exact AUA call, purpose, proof, cleanup, risk, authorization, and redundancy
metadata. The model may call only:

```text
select_candidate(candidate_id: integer)
```

The model does **not** write AUA arguments, execute actions, decide authorization, or waive cleanup.
Deterministic AUA code owns candidate construction, schema validation, safety policy, execution,
post-action observation, and proof. Model output must parse as exactly one selector call and name an
offered ID. The raw closed-loop evaluator deliberately records unsafe, unauthorized, or redundant
model choices instead of hiding them; an execution integration must reject such choices fail-closed.

## Data and privacy

The curriculum is generated entirely from fictional `com.example.*` applications, states, and
identifiers. Generation does not connect to Android, read AUA session memory, or ingest device
journals. Raw journals are intentionally excluded.

`validate_dataset.py` checks the FunctionGemma message/tool contract, group separation, candidate-ID
randomization, split hashes, sequence length, host/private markers, repository privacy fingerprints,
and possible high-entropy secrets. The training runner rechecks the frozen manifest, every split
SHA256, and the validation token-length contract before MLX starts.

### Historical usage preparation for V9

Do not train V9 directly from AUA journals. Historical records retain short UI copy, goals,
packages, selectors, and URIs while slimming successful observations. Use the privacy-safe structural
miner instead:

```bash
.venv/bin/python -m experiments.functiongemma.history_miner \
  --output-dir runs/functiongemma/history-v9-prep \
  --overwrite
```

It emits ignored structural episodes, policy decisions, and curriculum seeds with no source copy,
package, selector value, serial, owner, timestamp, or physical-device session. Its output is
curation material requiring fictional semantics and an independent oracle; it intentionally emits
zero native training rows. See [V9_HANDOVER.md](V9_HANDOVER.md) for the current snapshot, exact
privacy contract, data-growth plan, and promotion gates.

## Setup

Apple silicon and Python 3.11+ are required by the local MLX path used here.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r experiments/functiongemma/requirements.txt
uv pip install --python .venv/bin/python -e .

FG_MODEL_PATH=/absolute/path/to/functiongemma-270m-it-bf16
```

The recorded runs use the public MLX conversion `mlx-community/functiongemma-270m-it-bf16` resolved
to a local snapshot. Commands below require that local directory; the training runner does not fetch
a model implicitly.

## Cost-safe RunPod CUDA benchmark

`runpod_benchmark.py` measures CUDA compatibility and throughput without connecting to Android. Its
default is the frozen v3 corpus, `train-lora.yaml`, and exactly 128 MLX microbatch iterations
(32 optimizer updates with batch size 8 and gradient accumulation 4)
from the pinned base-model revision. It is a speed/compatibility benchmark, not a quality promotion.

Install the current RunPod CLI, then inspect the no-cost plan. Dry-run is the default:

```bash
brew install runpod/runpodctl/runpodctl

.venv/bin/python experiments/functiongemma/runpod_benchmark.py
```

The launcher reads `RUNPOD_API_KEY` and `HF_TOKEN` from the process environment or the ignored root
`.env`. It never prints either value. The RunPod key is passed to `runpodctl` only through its process
environment; the Hugging Face token is sent to the worker only over encrypted SSH stdin and is not
stored in Pod environment metadata, command arguments, or artifacts.

RunPod official images inject account SSH keys when the container starts. The launcher therefore
generates a unique ephemeral key, snapshots the account's fingerprint inventory, registers that
public key before Pod creation, and still supplies the same key through `SSH_PUBLIC_KEY` as a
defense-in-depth override. The private key remains only in a mode-restricted temporary directory.

Explicitly authorize one billable L40S benchmark:

```bash
.venv/bin/python experiments/functiongemma/runpod_benchmark.py --execute
```

Use `--gpu "NVIDIA GeForce RTX 5090"` for the matching 5090 measurement. Defaults cap the accepted
price at `$1.25/hour`, cap the 90-minute hard-TTL ceiling at `$2.00`, and use no persistent volume.
The actual hourly price is checked immediately after creation; an over-limit Pod is terminated before
training.

Pod cost containment does not depend on the launcher remaining alive:

- Pod creation includes RunPod's server-side absolute `--terminate-after` deadline.
- The local launcher secret-scans the downloaded tarball in memory before any artifact file or
  extraction directory is created, then writes and hashes it before cleanup.
- A `finally` path sends DELETE for the exact Pod ID on success, failure, timeout, or any available
  SIGHUP/SIGINT/SIGQUIT/SIGTERM termination signal.
- Those signals received during artifact export or cleanup are deferred until DELETE and the audits
  finish, so a repeated termination signal cannot skip the exact-ID deletion path.
- It then lists Pods and requires zero active resources matching the exact ID or unique run name.
- Only after Pod deletion is attempted, it removes the exact temporary account-key fingerprint and
  requires the final fingerprint inventory to equal the pre-run baseline. A lost add/remove response
  is recovered from the authoritative inventory instead of blindly repeating or abandoning cleanup.
- It snapshots `GET /networkvolumes` before creation and after Pod deletion, proving that no new
  persistent network volume appeared; the create request itself sets volume size zero and supplies no
  network-volume ID.
- A lost create response is recovered by that unique name. An unverified cleanup is always the
  primary reported failure—including temporary account-key cleanup—even after an earlier worker
  failure, while the server-side Pod deadline remains armed.

An uncatchable local `SIGKILL` or machine power loss cannot run account-key removal. Before adding
the key, the launcher atomically records its exact fingerprint and the baseline inventory in
`launcher-metadata.json`, so recovery can remove only that fingerprint; the server-side Pod deadline
still bounds spend independently.

Each run writes under `runs/functiongemma/runpod/<unique-run-id>/`:

- `artifacts.tar.gz` and its SHA256 in `launcher-metadata.json`
- safely extracted adapter, worker metadata, frozen manifest/validation report, and training config
- exact base revision, reviewed source overrides, GPU/package details, timing, price, and cleanup audit
- the temporary SSH-key fingerprint, baseline/after inventories, lost-response recovery, and exact
  fingerprint-removal audit (never the private key)

The uploaded source contains the pinned Git archive plus an explicit allowlist of reviewed
FunctionGemma worktree overrides recorded in launcher metadata. It cannot include ignored `.env`,
`runs/`, model caches, or device journals.

An optional v4-shaped **throughput-only** benchmark must pin its distinct manifest explicitly:

```bash
.venv/bin/python experiments/functiongemma/runpod_benchmark.py --execute \
  --curriculum-version v4 \
  --config experiments/functiongemma/train-lora-v4.yaml \
  --expected-manifest-sha256 3a271e8ff153b9179997edbb9822962b383348405bc77b15259dc3a733b6a9b7
```

That command still starts from the base model and must not be reported as the historical v4 quality
run. The historical v4 was a continuation from the exact validation-selected v3 adapter. A true cloud
continuation requires an authenticated parent-adapter upload and `resume` provenance, which this
benchmark launcher deliberately does not guess or substitute with the distribution bundle.

The completed v6 quality cycle used the following shape with a three-hour server ceiling; its actual
Pod was deleted after about 87 minutes and training itself took about 27 minutes:

```bash
.venv/bin/python experiments/functiongemma/runpod_benchmark.py --execute \
  --mode full \
  --curriculum-version v6 \
  --config experiments/functiongemma/train-lora-v6.yaml \
  --expected-manifest-sha256 d3900c58a698810aa1eb378a6fb51b7b4a997f351b2173b455a046c63ad98364 \
  --ttl-minutes 180 \
  --wait-minutes 160 \
  --max-total-usd 3.50
```

The long ceiling is a failure bound, not intended runtime. Exact-ID deletion, zero-volume audit, and
temporary-key removal completed successfully after artifact transfer.

## Reproduce the pipeline

Generate the deterministic synthetic splits:

```bash
.venv/bin/python -m experiments.functiongemma.generate_dataset \
  --output-dir runs/functiongemma/data
```

Validate structure, privacy, split isolation, and the 1,024-token contract:

```bash
.venv/bin/python experiments/functiongemma/validate_dataset.py \
  --model "$FG_MODEL_PATH" \
  --data-dir runs/functiongemma/data \
  --max-seq-length 1024
```

Run a bounded smoke train into a new adapter directory:

```bash
.venv/bin/python experiments/functiongemma/train.py smoke \
  --model "$FG_MODEL_PATH" \
  --data-dir runs/functiongemma/data \
  --adapter-path runs/functiongemma/adapters/local-smoke \
  --expected-manifest-sha256 d96d69e7f25df0b10272d6e20027eea3f609a34a741ce50d60d75b7f983df60b
```

Run the configured full train. The target must be empty unless explicit `resume` mode is used:

```bash
.venv/bin/python experiments/functiongemma/train.py full \
  --model "$FG_MODEL_PATH" \
  --data-dir runs/functiongemma/data \
  --adapter-path runs/functiongemma/adapters/local-full \
  --expected-manifest-sha256 d96d69e7f25df0b10272d6e20027eea3f609a34a741ce50d60d75b7f983df60b
```

Evaluate greedy generation on the frozen held-out split with fail-closed gates:

```bash
.venv/bin/python experiments/functiongemma/evaluate.py \
  --model "$FG_MODEL_PATH" \
  --adapter runs/functiongemma/adapters/local-full \
  --data runs/functiongemma/data/test.jsonl \
  --output runs/functiongemma/local-heldout.json \
  --min-candidate-accuracy 0.99 \
  --min-critical-accuracy 0.99 \
  --min-parse-success 1.0 \
  --max-unauthorized-selections 0 \
  --max-redundant-selections 0
```

Finally, run the fictional stateful scenario across four opaque-ID permutations:

```bash
.venv/bin/python experiments/functiongemma/run_closed_loop.py \
  --model "$FG_MODEL_PATH" \
  --adapter runs/functiongemma/adapters/local-full \
  --output runs/functiongemma/local-closed-loop.json
```

Static held-out accuracy is not a substitute for this closed-loop gate. The simulator verifies
semantic-trace invariance, unknown-outcome recovery, cleanup, terminal completion, and zero unsafe,
unauthorized, redundant, invalid, or repeated-mutation selections.

Exercise the bundled adapter through the production prompt serializer across a balanced 96-case
matrix of candidate orders and dense-ID permutations:

```bash
.venv/bin/python experiments/functiongemma/run_production_smoke.py \
  --model "$FG_MODEL_PATH" \
  --adapter bundled \
  --output runs/functiongemma/production-smoke-bundled.json
```

This command exits nonzero when any production-shaped gate fails and still writes the full report.

Before loading any model, gate AUA's trusted candidate compiler itself. This curriculum-independent,
host-only corpus contains 128 fictional actionable cases and 16 mandatory abstentions across
resource-id, text, description, title-plus-summary, overlapping-target, and compound-goal shapes:

```bash
.venv/bin/python -m experiments.functiongemma.aua_candidate_benchmark \
  --output runs/functiongemma/aua-candidate-benchmark.json
```

The report separates requested-target extraction, exact oracle-call inclusion, deterministic
recommendation accuracy, and safe abstention. A model score is not meaningful unless the oracle
action was first present in the guarded candidate set.

## Failure-driven iterations

- **v1** established strict FunctionGemma parsing, opaque-ID counterfactuals, grouped splits, and
  provenance. Its held-out result was high but still selected a redundant candidate, so it was not
  accepted as a safe policy.
- **v2** improved static held-out candidate accuracy to 99.84% with no held-out unauthorized or
  redundant selections, but failed the closed-loop test: it repeatedly chose a redundant observation
  while preparing offline and could finish early. Investigation showed the static test reused
  train-seen candidate bundles and missed runtime-shaped prompt/schema variation.
- **v3** added runtime-shaped sequential families, exact call defaults and cleanup semantics, and an
  out-of-distribution pre-training gate. Exploratory checkpoints demonstrated why both static and
  closed-loop evaluation are required. The reproducible validation-selected 1,408-iteration rerun is
  the final evaluated run.

## v3 provenance snapshot

The following identities are frozen for the validation-selected rerun:

| Artifact | SHA256 / revision |
| --- | --- |
| Required base-model runtime files | `76aabb2800b6b9e6da9160028dfb233bbfa723d8c33e21623022ca87a8fa9fd5` |
| Model snapshot revision | `bb327a9ad61044e1496a2bee2365a6b6a6684c72` |
| Dataset aggregate | `f8983d45603a308ba95071af6497d6569ff1e03f1a11e332777bd0bacaa080db` |
| Dataset manifest | `d96d69e7f25df0b10272d6e20027eea3f609a34a741ce50d60d75b7f983df60b` |
| Train split | `6b5fe6ffa43280004d72298c6ed3c340c9fdd884e180a076e7e571b89df3b609` |
| Validation split | `cd86d1019c5c062c92245b0cd6e53754ae2ebfa030075a5c656ca8cf9c767831` |
| Test split | `5653df0fc3222b55c26052e4c171673dfa4389b62738415b3f7add6e85ed5900` |
| Validation report | `5ec684fc9cef7edf8d1d89c5c8179640344b18d16f0a6ccc5b194359b89c6ec1` |
| 1,408-iteration config | `e38a94afa9ae9db1c5989c85f02d32834c50e4883279996961d8eb4bf2e171d5` |

The v3 dataset contains 12,288 train, 2,048 validation, and 2,048 test records. The final run used
seed 42, LoRA rank 16, learning rate `1e-6`, maximum sequence length 1,024, and wrote atomic metadata
to `runs/functiongemma/adapters/lora-r16-v3-early-stop/run-metadata.json`. That record, not this README,
is authoritative for timestamps, status, exact arguments, interruption truth, checkpoint path, and
final adapter hashes.

The base digest above is the canonical hash of the seven manifest-required relative-path/content
SHA-256 pairs. It deliberately ignores unrelated cache and snapshot files, so another local layout
can prove the same runtime identity. The exact file list, byte sizes, and individual hashes are
checked into the [bundled manifest](../../src/android_ui_analyser/resources/functiongemma/manifest.json).

## Final v3 result

The validation-selected run completed all 1,408 iterations. Its final adapter SHA256 is
`f4d2f2ed67ea1b50cdc8db511900df789d8767961ffc6f7271fe40478718575b`.

| Evaluation | Cases | Candidate accuracy | Critical accuracy | Parse success | Unauthorized | Redundant | Strict gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Frozen v3 held-out | 2,048 | 99.8535% | 99.8438% | 100% | 1 | 1 | **FAIL** |
| Preserved v2 held-out | 1,280 | 99.9219% | 99.8698% | 100% | 0 | 0 | **PASS** |
| Preserved v1 held-out | 1,024 | 99.6094% | 99.2188% | 100% | 1 | 1 | **FAIL** |

The unchanged stateful scenario passed all four opaque-ID permutations. All four runs completed the
goal, recovered the unknown outcome, restored the environment, and finished the session. Across the
24 decisions there were zero unsafe, unauthorized, redundant, invalid, or repeated-mutation
selections, and every semantic trace was identical.

The later host-only production-serializer smoke **failed**. The bundled adapter selected the intended
semantic call in only 60 of 96 balanced engine-shaped cases (62.5%). Protocol parsing, offered-ID
validity, and provider/parser agreement were each 100%, so valid syntax hid a large semantic error:
accuracy ranged by 37.5 percentage points across target candidate IDs and 54.17 points across target
positions. This is direct evidence of engine-shaped ID/position sensitivity that the earlier static
and single-scenario gates did not expose.

## Failure-driven v4 continuation (not promoted)

V4 added leak-audited production-shaped rows and one bounded continuation from the validation-selected
v3 adapter. Several gates improved substantially:

| V4 evaluation | Result |
| --- | ---: |
| Validation, combined corpus | 2,767/2,768 (99.9639%) |
| Production-shaped validation subset | 719/720 |
| Untouched production smoke | 96/96 (v3: 60/96) |
| Held-out production, cardinality 2 | 64/64 |
| Held-out production, cardinality 3 | 144/144 |
| Held-out production, cardinality 4 | 512/512 |
| Fictional closed loop | 4/4 clean |

The independent combined test nevertheless **failed**: 2,764/2,768 correct (99.8555%), 99.6875%
critical accuracy, and 100% parse success, with zero redundant selections but four unauthorized
choices. Every error was in `sequence_recover_unknown`: the model chose an early `session_finish`
instead of `analyze_screen` to observe the uncertain outcome. The apparently perfect production
choices therefore did not establish recovery safety.

The v4 adapter remains under the ignored run tree and is **not bundled**. Bundled v3 remains
shadow-only. The next iteration needs independent recovery-focused data and evaluation, especially
counterfactuals that keep unknown-outcome observation distinct from legitimate session completion.
The checked-in reproduction path is [production_curriculum.py](production_curriculum.py),
[train-lora-v4.yaml](train-lora-v4.yaml), [evaluate.py](evaluate.py),
[run_production_smoke.py](run_production_smoke.py), and [run_closed_loop.py](run_closed_loop.py).
No ignored adapter, dataset, or report is linked as a repository artifact.

## Failure-driven v5 and v6 runs (not promoted)

V5 added recovery-focused examples and a longer fresh-base run. Its held-out report still contained
one unauthorized and five redundant selections, so it was rejected before promotion. V6 then added
exact packaged `policy_messages()`/`policy_tools()` serialization, full 24-by-24 semantic
permutations for live-shaped groups, rank-32 LoRA, and fail-closed checkpoint selection over all 16
saved checkpoints. The CUDA run used an L40S and completed 8,192 microbatch iterations in 1,613.197
seconds. Its selected final adapter SHA256 is
`5c1b426dd35b9fe3f2cc07c31316d402dce707da4b313e1deea563cc2aa57072`.

The selected 8,192-step checkpoint was the only checkpoint to satisfy every validation safety gate:

| V6 evaluation | Result |
| --- | ---: |
| Validation | 9,108/9,116 (99.9122%); critical 100%; unauthorized 0; redundant 0; permutation groups 8/8 |
| Untouched combined test | 9,116/9,116; critical 100%; unauthorized 0; redundant 0; permutation groups 8/8 |
| Production serializer smoke | 96/96 |
| New live-context smoke | 384/384 |
| Fictional closed loop | 4/4 clean |

Those host-only gates were necessary but still not sufficient. A subsequent advisory-only AUA test
used an AUA-owned API-36 `Medium_Phone` emulator on the public Android Settings home screen. Four
real clickable rows were offered while the requested destination varied across those same four rows.
FunctionGemma selected the requested row in only 1/4 cases; AUA's deterministic recommendation was
correct in 2/4. No suggestion was executed. A warm four-session policy batch took 10.7 seconds versus
9.26 seconds with policy off, so this sample showed both lower accuracy and modest overhead.

V6 is therefore **not bundled or promoted**. The failure demonstrates remaining context shift between
the synthetic serializer corpus and actual AUA compiler output, especially real control text that
combines titles with summaries and objectives containing several overlapping destination names. The
next iteration must use privacy-scrubbed contexts captured at the exact production compiler boundary,
split by screen/scenario family, and retain an independent live-emulator gate. It must not learn
private app names, maps, routes, UI copy, screenshots, journals, or user data.

A subsequent source audit found that the entire compound phase objective—including the list of
visible alternatives—was used to rank every row. AUA now extracts only the requested navigation
object for deterministic ranking, policy filtering, and the model-facing goal. The identical
four-target Settings audit then improved from 2/4 to 4/4 with policy off. The new 208-case trusted
action-compiler benchmark covers 128 policy taps, 64 deterministic stale/loading/progress/scroll
recoveries, and 16 disabled-target abstentions. It passes target extraction, oracle-action
inclusion, deterministic action/recovery, and safe abstention at 100%. Recovery stays AUA-owned:
stale frames refresh uncached, loading waits for evidence, and one confirmed scroll advances only
one page while returning the analyzed frame. Policy audit responses also expose value-free stage counts and whether
AUA's own recommended call was actually offered, so selector accuracy can no longer hide candidate
recall. Session bootstrap additionally replaces an explicitly unstable folded launch frame with one
bounded authoritative hierarchy read before planning.

This remains an experimental bounded candidate selector, not an execution authority: the frozen
v3 and preserved v1 static gates each exposed one unauthorized and one redundant choice. AUA packages
it only for the optional deterministic-guarded **shadow** path. The advisory interface exists for
development, but bundled v3's authenticated manifest caps rollout at shadow, so advisory fails
closed as `unsupported_mode` before inference. The model is not an autonomous AUA agent, never
executes a candidate, and is withheld for two/three candidates because bundled v3 training and
evaluation froze exactly four-way sets. The failed v4 continuation and v6 live-emulator regression
show that production-smoke accuracy alone is insufficient; advisory must wait for recovery-safe and
live-context-safe evidence. No result here is a claim of production speed.

Generated datasets, run metadata, reports, and intermediate checkpoints live under the intentionally
ignored `runs/functiongemma/` tree. Checked-in evidence is the immutable
[bundle manifest](../../src/android_ui_analyser/resources/functiongemma/manifest.json), the
[model card](../../src/android_ui_analyser/resources/functiongemma/MODEL_CARD.md), and the
[policy](../../tests/test_policy_core.py), [integration](../../tests/test_functiongemma_engine_policy.py),
[closed-loop](../../tests/test_functiongemma_closed_loop.py), and
[production-smoke](../../tests/test_functiongemma_production_smoke.py) regression suites. The
[smoke runner](run_production_smoke.py) reproduces the ignored detailed report; the checked-in test
pins the matrix and gates without pretending its fixture model is the trained adapter.

All generation, training, static evaluation, closed-loop simulation, and production-serializer smoke
described here are host-only. The separately identified v6 advisory audit used only an AUA-owned
emulator and public Android Settings UI; its suggestions were observed but never executed. No private
app data was used as learning material or committed evidence.

## V8 handoff and live-failure correction corpus

V8 keeps the frozen v7 foundations and adds an authenticated, non-executing handoff outcome plus
fictional corrections for the five-agent no-map audit. The model still emits exactly one
`select_candidate(candidate_id)` call. Candidate ID `-1` means return control without acting, and is
accepted only when both the prompt context and the adapter's pinned manifest authorize that
protocol. It is not an AUA command. Zero eligible actions hand off deterministically without model
inference.

The 1,000 source-oracle examples render into 2,950 counterbalanced native rows. The combined corpus
contains 61,758 rows and passes the real FunctionGemma tokenizer at a maximum of 772/1,024 tokens.
Generate and validate it with:

```bash
.venv/bin/python -m experiments.functiongemma.generate_dataset \
  --curriculum-version v8 \
  --output-dir runs/functiongemma/data-v8
.venv/bin/python -m experiments.functiongemma.validate_dataset \
  --model "$FG_MODEL_PATH" \
  --data-dir runs/functiongemma/data-v8 \
  --max-seq-length 1024
```

The pinned manifest-file SHA256 is
`6acd0f21797c3ba62b7b979b5db569af9fe75cc325a3cb311d8bc2f99a809ccc`. The RunPod full-cycle config
is [train-lora-v8-seed83.yaml](train-lora-v8-seed83.yaml): fresh base, rank 32, 8,192 iterations,
checkpoint/evaluation every 512.

The L40S training run completed all 8,192 iterations in 1,838.897 seconds. Evaluation selected the
earliest fully converged checkpoint, step 4,608, SHA256
`f1d108aa178f9e9171855d7ee6a505bf1ace17ce6191289c009f281c634100e2`. It scored 7,587/7,587 on
validation with perfect critical accuracy, zero unauthorized/redundant selections, and complete
candidate-order/ID invariance. Seven other checkpoints also passed the strict validation safety
contract.

The untouched test prevented promotion: 7,581/7,587 = 99.9209%, critical 99.9638%, parse and
offered-ID checks 100%, zero unauthorized selections, but one critical redundant cleanup choice.
The other five misses were safe four-way semantic destination choices. All 45 held-out
`target_absent_handoff` cases and all 45 `two_hop_navigation` cases were correct. The portable
production smoke passed 96/96, variable-cardinality semantic smoke passed 99/99, and the existing
four-permutation closed loop passed every completion, cleanup, recovery, and safety gate.

V8 is therefore a strong positive experiment but remains unbundled and unpromoted under the
zero-safety-error rule. Evaluation also exposed a deployment bug: authenticated explicit adapters
retained a deleted training-Pod model path. Runtime validation now permits that stale provenance
path only when the manifest, adapter config, weights, and current base-model digest are all pinned;
unverified adapters remain fail-closed.
