# MODIFIED FUNCTIONGEMMA MODEL DERIVATIVE

Modified by The Wordlab on 2026-08-19 using MLX LoRA fine-tuning for guarded
AUA candidate selection. Google did not create or endorse these modifications.

## AUA FunctionGemma candidate policy v10

This directory contains a **modified FunctionGemma Model Derivative**: a LoRA
adapter trained by the Android UI Analyser project. It selects one opaque integer
ID from two, three, or four complete candidate calls authored and guarded by AUA,
or returns the reserved ID `-1` to decline. It does not perceive Android UI, author
arguments, grant authorization, execute calls, or own cleanup. AUA bypasses the
model entirely for zero or one candidate, deciding those deterministically.

## What is distributed

- `adapters.safetensors`: 30,403,414-byte rank-32 LoRA adapter, SHA-256
  `349413cfe24e664e614f15f3c12a3e53621bb858cd55c5909db23464672ec2a5`.
  Its tensors are the evaluated training output (SHA-256 `69b1a360af4aa75b0…`);
  the distributed file adds only prominent Safetensors modification metadata.
- `adapter_config.json`: portable MLX LoRA configuration. It references the base by
  pinned repository and revision rather than by any absolute path, so the artifact
  carries neither the packager's machine nor the training host's filesystem.
- `manifest.json`: immutable base, prompt-protocol, training, evaluation, and license
  provenance, including the candidate cardinalities and handoff protocol this adapter
  was trained for. The provider trusts that declaration, so it is pinned by test.

The approximately 543 MiB base model is deliberately **not** distributed with
AUA. Users must separately obtain the compatible local MLX conversion identified
in `manifest.json`, after reviewing and accepting Google's Gemma terms.

## Training

Trained on the V10 command-surface curriculum: 87 fictional families spanning 67 AUA
commands, with several confounds removed by construction. Candidate count is
decorrelated from the label, action directions are mirrored so that tapping can be the
correct answer rather than a tell, and 168 goal templates prevent any single phrasing
from being memorised. Vocabulary is split-isolated. The frozen V8 splits are merged
verbatim so the earlier foundation is retained rather than re-sampled.

The reproducible generator, validator, trainer, static evaluator, deterministic
closed-loop simulator, and production-serializer smoke runner live in
`experiments/functiongemma/`. Generated datasets, base weights, intermediate
checkpoints, and detailed reports stay ignored; they can be rebuilt from the
checked-in source and recorded seeds.

## Evaluation

Scored by an **independently authored** probe whose 150 scenarios derive from the CLI
surface rather than from the training generators. This separation is load-bearing: an
in-house probe that shared its generator's phrasing reported 6/6 on a refusal
capability that independent measurement put at 0/144, because it was measuring the
phrasing rather than the model.

| | accuracy | refusal | invalid outputs |
|---|---|---|---|
| v10, best checkpoint | 0.600 | 18/38 | 0 |
| v10, mean over 16 checkpoints | 0.471 | — | — |
| v9 (same 150 jobs) | 0.620 | 6/38 | — |
| v8 (same 150 jobs) | 0.387 | — | — |

On a physical device this adapter completed 5/5 navigations and 2/2 refusals with zero
wrong taps, one refusal under a deliberately rephrased goal. The packaged copy
reproduces the external checkpoint exactly on the same 150 jobs.

### This artifact is not promoted

The manifest authenticates advisory rollout, so the guarded execution lane is reachable
without an externally supplied adapter. That is a transport permission, not a quality
claim, and it is not what keeps the model inert: AUA ships with `policy.enabled: false`
and `policy.mode: off`, so nothing is resolved, loaded, or given memory until an
operator explicitly turns the policy on.

The evidence does not support more than that. There is one seed and no live gate, and
refusal is unstable across checkpoints (0, 2, 0, 5, 4, 18, 0, 3, 8, …), so 18/38 is the
high-water mark of a noisy process rather than a converged property. **Refusal is
therefore not load-bearing anywhere in AUA.** The deterministic guard removes unsafe,
unauthorized, destructive, stale, ambiguous, and redundant candidates before inference
and revalidates before execution; the model only ever chooses among options AUA already
considers safe, and cannot widen that set.

Earlier generations are retained in the project record for the same reason. Frozen v3
scored 99.8535% on synthetic held-out data and then 62.5% on a production-serializer
matrix, with accuracy swinging 37.5 points across target IDs — the distance between a
synthetic score and a real one. A failure-driven v4 continuation also failed its
independent gate and was never bundled.

This adapter is not approved for user-facing recommendations, unguarded use, or
autonomous execution.

## Terms and modifications

This adapter is a modified Model Derivative and is subject to `LICENSE`,
including the incorporated prohibited-use restrictions reproduced in
`GEMMA_PROHIBITED_USE_POLICY.md`. The required distribution notice and a precise
modification statement are in `NOTICE` and `adapters.safetensors.NOTICE`.
