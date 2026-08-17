# FunctionGemma candidate-selection experiment

This host-only experiment tests whether a small FunctionGemma model can choose the next action
in an AUA workflow without being allowed to author or execute Android commands.

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

## Reproduce the pipeline

Generate the deterministic synthetic splits:

```bash
.venv/bin/python experiments/functiongemma/generate_dataset.py \
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

This is a promising bounded candidate selector, but it is **not an execution authority**: the frozen
v3 and preserved v1 static gates each exposed one unauthorized and one redundant choice. AUA packages
it only for the optional deterministic-guarded **shadow** path. The advisory interface exists for
development, but bundled v3's authenticated manifest caps rollout at shadow, so advisory fails
closed as `unsupported_mode` before inference. The model is not an autonomous AUA agent, never
executes a candidate, and is withheld for two/three candidates because bundled v3 training and
evaluation froze exactly four-way sets. The failed v4 continuation is evidence that production
choice accuracy alone is insufficient; advisory must wait for an independent recovery-safe
iteration. No result here is a claim of production speed.

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
described here are host-only. No physical Android device, emulator, ADB command, or app data is used.
