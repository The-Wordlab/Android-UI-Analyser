# MODIFIED FUNCTIONGEMMA MODEL DERIVATIVE

Modified by The Wordlab on 2026-08-14 using MLX LoRA fine-tuning for guarded
AUA candidate selection. Google did not create or endorse these modifications.

## AUA FunctionGemma candidate policy v3

This directory contains a **modified FunctionGemma Model Derivative**: a LoRA
adapter trained by the Android UI Analyser project. It selects one opaque integer
ID from four complete candidate calls authored and guarded by AUA. It does not
perceive Android UI, author arguments, grant authorization, execute calls, or own
cleanup. AUA's production policy boundary supports one to four candidates, but
bypasses the model for zero/one and currently withholds model advice for two/three:
the frozen adapter was trained and evaluated only on four-way sets.

## What is distributed

- `adapters.safetensors`: 15,215,272-byte LoRA adapter, SHA-256
  `21875181afa500ac0d11af944ae3bc92c71bf59b4bec202bc6c6b6221bf78743`.
  Its tensors are the evaluated training output (SHA-256 `f4d2f2ed67ea1b50c…`);
  the distributed file adds only prominent Safetensors modification metadata.
- `adapter_config.json`: portable MLX LoRA configuration.
- `manifest.json`: immutable base, training, evaluation, and license provenance.

The approximately 543 MiB base model is deliberately **not** distributed with
AUA. Users must separately obtain the compatible local MLX conversion identified
in `manifest.json`, after reviewing and accepting Google's Gemma terms.

## Training and evaluation

The reproducible generator, validator, trainer, static evaluator, deterministic
closed-loop simulator, and production-serializer smoke runner live in
`experiments/functiongemma/`. Generated datasets, base weights, intermediate
checkpoints, and detailed reports stay ignored; they can be rebuilt from the
checked-in source and recorded seeds.

The frozen v3 held-out synthetic evaluation selected 2,045 of 2,048 candidates
correctly (99.8535%) with 100% protocol parsing, but made one unauthorized and one
redundant raw selection. Those evaluation rows always had four candidates and did
not include an all-tap candidate set, so the result is not direct evidence for the
initial production tap-only surface. A single fictional six-step closed-loop
scenario completed cleanly under four opaque-ID permutations.

The subsequent host-only, engine-shaped production-serializer smoke **failed**.
Across all 24 candidate orders and four dense-ID permutations, the bundled adapter
made 60 of 96 semantically correct selections (62.5%). Protocol parsing, offered-ID
validity, and provider/parser agreement were 100%, but accuracy differed by 37.5
percentage points across target IDs and 54.17 points across target positions. This
exposed ID/position sensitivity that the earlier synthetic gates did not catch.

### V4 continuation was not promoted

A failure-driven v4 continuation reached 2,767/2,768 validation (99.9639%),
including 719/720 production-shaped validation cases. It scored 96/96 on the
untouched production smoke versus v3's 60/96, passed held-out production choices
at cardinalities two (64/64), three (144/144), and four (512/512), and completed
the fictional closed loop 4/4 cleanly.

The independent combined test still failed: 2,764/2,768 correct (99.8555%),
99.6875% critical accuracy, 100% parsing, zero redundant selections, and four
unauthorized selections. All four were `sequence_recover_unknown` cases where the
model chose an early `session_finish` instead of `analyze_screen`. V4 therefore
remains ignored and is not included in this directory. The next iteration needs
independent recovery-focused data and evaluation.

Consequently bundled v3 is approved only for AUA's optional guarded **shadow** path.
Its authenticated manifest caps rollout at shadow, so the provider rejects advisory
as `unsupported_mode` before inference. The failed v4 cycle does not change this
artifact or its rollout. This adapter is not approved for user-facing recommendations,
unguarded use, or autonomous execution.

## Terms and modifications

This adapter is a modified Model Derivative and is subject to `LICENSE`,
including the incorporated prohibited-use restrictions reproduced in
`GEMMA_PROHIBITED_USE_POLICY.md`. The required distribution notice and a precise
modification statement are in `NOTICE` and `adapters.safetensors.NOTICE`.
