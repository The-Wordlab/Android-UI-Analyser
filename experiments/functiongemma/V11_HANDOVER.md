# V11 handover

Written 2026-08-22, after re-reading [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) and measuring the
runtime against the corpus. Nothing here is a plan for more iterations of V10. Two of the log's
recorded conclusions are wrong, and one previously unmeasured defect is large enough to explain the
single pattern that has repeated since v6: **synthetic accuracy that does not survive contact with a
device**.

Read this file before `V9_HANDOVER.md`. Read both before touching a generator.

## 1. Every pass count in the log is 4x too high

`experiments/functiongemma/requirements.txt` pins `mlx-lm[train]==0.31.3`, and `runpod_worker.py`
installs from that file alongside `mlx[cuda12]` — MLX has a CUDA backend, so the RunPod runs used
the *same* trainer as a local Apple-silicon run, not a HuggingFace one. In
`mlx_lm/tuner/trainer.py` the loop is:

```python
for it, batch in zip(range(1, args.iters + 1),
                     iterate_batches(batch_size=args.batch_size, ...))
```

One iteration consumes **one batch of `batch_size`**. `grad_accumulation_steps` only gates
`do_update` / `optimizer.update`; it does not add sequences. Trained-token counts confirm it:
`batch_size: 8` + `grad_accumulation_steps: 4` logs 136 tokens/iter, `batch_size: 32` +
`grad_accumulation_steps: 1` logs 544.

Every config in this directory uses `batch_size: 8`, so sequences seen = `iters x 8`:

| run | sequences | real passes | log claimed |
|---|---:|---:|---|
| V9 seed101, 8,192 it, 66,000 rows | 65,536 | 0.99 | ~4 |
| V10 "one pass", 8,192 it, 262,584 rows | 65,536 | 0.25 | ~1 |
| V10 "four passes", 32,768 it | 262,144 | 1.00 | ~4 |
| V10 selected checkpoint, step 18,432 | 147,456 | **0.56** | (inside the "4-pass" run) |
| Qwen3-1.7B, 8,192 it | 65,536 | 0.25 | 1 |

The log's *relative* reasoning survives intact. The 4:1 exposure gap it identified between V9 and
V10 at 8,192 iterations is real, and the "epoch-matched" 32,768-iteration run genuinely does match
V9's exposure (1.00 against 0.99 passes). Only the absolute labels are wrong. Two conclusions do
not survive:

- **"Saturation is exposure, not only corpus difficulty" is withdrawn.** V9 reached 0.000
  validation loss at iteration 1,536 = **0.19 passes**, before any example had repeated. A flat zero
  without repetition cannot be a memorisation signal. The earlier reading — that the V9 corpus was
  simply too easy to be informative — was correct, and the V10 "control" did not overturn it.
- **"Refusal does not emerge within one pass" is false.** The 18/38 checkpoint sits at **0.56
  passes**. Refusal emerged in a little over half a pass and then stayed noisy
  (0, 2, 0, 5, 4, 18, 0, 3, 8, 11, 1, 1, 5). Instability across checkpoints is the finding; exposure
  is not the explanation.

Practical consequence: the run that produced the best refusal number needs **~2.7 h on an M5 Max**,
not the 19-20 h that four true passes would cost. Nothing in the evidence says four true passes is
the useful target.

## 2. Train/serve skew: every live context differs from every training row, in four places at once

This is the defect worth the next cycle. `experiments/functiongemma/v9_curriculum.py` imports
`policy_messages` and `compile_policy_context` from the live `android_ui_analyser.policy`, so the
corpus and the runtime can never disagree about *structure* — and they do not. Top-level keys and
candidate keys match exactly. The **content** of four fields does not.

Measured over 2,400 rendered rows across all three splits, against the context the live autopilot
actually builds at `engine.py:8529`:

| field | corpus | live runtime | overlap |
|---|---|---|---|
| `observation` | `{element_count, fresh, source}` (97% / 100% / 100%) | `{fresh}`, plus `known_screen` when known | `fresh` only |
| `constraints[:2]` | `read_only`, `fresh_observation_required`, `authorization_required`, `no_mutation`, ... | `"Select only a supplied guard-approved candidate."`, `"Do not invent or execute a call."` | **none** |
| `recent_outcomes` | 77 distinct strings, e.g. `mutation_authorized=false`, `helper_bound=true` | always exactly `session_active=true`, `outcome=known`, `goal_checkpoint_reached=false` | **none** |
| `candidates[].call.tool` | **67 distinct tools**; `tap_and_analyze` is only **16.7%** of candidates | `tap_and_analyze`, always | **16.7%** |

Reproduce with `python -c` over `v10_learning_material.generate` + `v9_curriculum.render_case`,
compiling a `PolicyContext` shaped like `engine.py:8529` through `compile_policy_context`.

The tool row is the severe one. There are exactly two `PolicyCandidate(` construction sites in
`engine.py` — the live autopilot at 8176 and the dashboard agent-model test at 7891 — and **both
hardcode `tap_and_analyze`**. Every other `"tool":` in `engine.py` is a `recommended_call` or MCP
suggestion handed to the parent agent, never a candidate the local model chooses among. So:

> The runtime asks this model exactly one question — *which of these 2-4 controls do I tap?* —
> and **83.3% of V10's training candidates are for commands it will never be offered.**

In a 270M model with roughly 100M non-embedding parameters, five of every six training candidates
teach a decision that is never requested. That is not a bug in the generator; the V10 corpus was
deliberately built as a "command-surface curriculum" over 67 commands. It is a misallocation
between ambition and deployment, and it is the most plausible remaining explanation for the
log's oldest unexplained pattern:

| cycle | synthetic result | live result |
|---|---|---|
| v6 | untouched test 100%, both smokes 100% | real AUA advisory 1/4, below deterministic 2/4 |
| v7 | untouched test 7,290/7,292, semantic smoke 99/99 | five-agent no-map matrix 1/4 |
| V9 | in-repo acceptance probe 6/6 on refusing an absent target | independent audit **0/144** |

The V9 probe failure was correctly diagnosed as a probe that reused its generator's phrasing. The
v6/v7 gap was attributed to successive distribution shifts. Skew of this size across four fields
simultaneously is a single sufficient cause for all of it, and it has never been measured before.

### The fix was already prescribed, three cycles ago

The log's "What to do next" list, written in the v7 era, item 2:

> **Capture the exact production policy context.** Add an explicit opt-in host/emulator recorder at
> the packaged `PolicyContext` boundary, after fail-closed privacy scrubbing.

`policy_trace.py` exists and implements `record_decision` / `record_outcome` behind a
non-configurable directory switch. V8, V9 and V10 were all trained on invented contexts anyway.
V11 must not be.

## 3. What V11's corpus has to change

Ordered by expected effect, largest first.

1. **Decide which contract the model is being trained for, then make both sides match it.**
   Two coherent options; pick one explicitly rather than drifting between them.
   - *Narrow (recommended).* The model answers only "which control do I tap". Regenerate the corpus
     with `tap_and_analyze` at or near 100% of candidates, and spend the freed 83% on the
     capabilities the independent audit actually measures — refusal, near-miss labels, destination
     versus breadcrumb, read-only goals with a mutation offered. A 270M selector has no capacity to
     spare on 66 commands it is never shown.
   - *Broad.* Keep the 67-command surface, and change the runtime so the policy path really does
     offer non-tap candidates. This is a much larger engine change and should not be assumed.
2. **Make the corpus emit the live `observation`, `constraints`, and `recent_outcomes` verbatim** —
   or change the runtime to emit the corpus's richer vocabulary. Either direction is fine; the
   present state, where 100% of live contexts open with two constraint strings and three outcome
   strings that appear nowhere in 2,400 training rows, is not. Add a gate test that fails when a
   compiled training context and a compiled live-shaped context disagree on any key's vocabulary.
   Note that `_OBSERVATION_KEYS` already permits `element_count` and `source`; the live caller
   simply never supplies them, which makes "enrich the runtime" the cheaper of the two directions.
3. **Relevance-only refusal must dominate the handoff population.** This is V10's own recorded
   diagnosis and it still stands: 88.2% of its handoff cases contain an unauthorized or destructive
   candidate, so the corpus offered the cheaper rule — *refuse when something destructive is on the
   menu*, decidable from the `risk` and `authorized` flags without reading the goal — and the model
   learned that instead of *refuse when nothing here advances what was asked*. Invert the ratio:
   relevance-only refusal becomes the majority, authorization refusal the minority.
4. **Do not expect construction to remove position bias.** The independent audit found first
   position selected 42.6% (V9) and 45.1% (V8) against a 32.9% chance rate, while opaque id zero is
   selected at exactly chance — and V9's corpus *already* counterbalanced order during training.
   This reads as a model-level inductive bias. The missing defence is order counterbalancing **at
   inference**, which is a runtime change, not a data change.
5. **Budget for the ±25% wording slop.** Meaning-preserving rewording flips one answer in four for
   both adapters. Any single-wording score carries slop larger than most per-bucket differences in
   the audit table. Score every scenario under multiple independently authored wordings, or do not
   compare buckets at all.

## 4. What to keep unchanged

- **Group isolation across splits**, and the `check_group_isolation` gate. It has never failed and
  it is the reason the held-out numbers mean anything at all.
- **Opaque dense candidate IDs, 0..N-1, with joint permutation of order and id.** The audit
  confirms id counterbalancing works — id zero is selected at exactly chance.
- **An independently authored probe.** The rule that earned its place the hard way: a probe written
  by whatever wrote the generator measures that generator's phrasing. V9 scored 6/6 in-repo and
  0/144 independently. Derive probe scenarios from `aua --help`, `aua guide --brief`, and the
  runtime contract in `policy.py`, never from a curriculum module.
- **Selection on worst-family accuracy plus zero safety errors, never validation loss.** V10's
  first run sat at 0.001 validation loss while refusing 0 of 38.
- **The privacy gate**, `tests/test_no_app_specific_refs.py`. Non-negotiable for a public corpus.

## 5. Base model: change it

FunctionGemma-270M is a poor allocation for this task, verified locally rather than taken from
marketing:

| | FunctionGemma-270M | LFM2.5-350M |
|---|---:|---:|
| total parameters | 268M | 354M |
| vocabulary table | 168M (63%) | 67M (19%) |
| **non-embedding parameters** | **100M** | **287M** |
| train throughput (identical LoRA config) | 7.0 it/s | **13.2 it/s** |
| peak memory | 4.1 GB | **2.9 GB** |
| inference, per decision | 234 ms | **92 ms** |
| context length | 32,768 | 32,768 trained (128k in config) |

Two thirds of FunctionGemma is a multilingual vocabulary table this task never uses. LFM2.5-350M
carries 2.9x the non-embedding capacity at 1.3x the total size, and is *faster and smaller in
memory* despite being nominally larger. This is consistent with the one capacity result already in
the log: Qwen3-1.7B beat FunctionGemma's mean across 16 checkpoints by 3.1 standard errors with
roughly half the spread, which matters more than its peak.

Liquid AI's published BFCLv3 tool-calling scores, for orientation only — not measured here:
LFM2.5-350M 44.11, Qwen3.5-0.8B 35.08, Gemma 3 1B IT 16.61; LFM2.5-1.2B-Instruct 49.12 against
Qwen3-1.7B 46.30.

### Integration facts, verified

- `mlx-lm 0.31.3` already supports the `lfm2` architecture. `linear_to_lora_layers` auto-discovers
  every `nn.Linear` per block, so LoRA attaches with no per-architecture key list and no new
  machinery. A 200-iteration smoke run trained clean at rank 32.
- Liquid AI publishes **official MLX bf16** builds (`LiquidAI/LFM2.5-350M-MLX-bf16`, also 230M and
  1.2B), so the `mlx-community` conversion step disappears.
- **One transport change is required.** The LFM2.5 chat template ignores `tool_calls` entirely and
  raises `'NoneType' object is not iterable` on `content: None`. The label must be a content string:

  ```
  <|tool_call_start|>[select_candidate(candidate_id=N)]<|tool_call_end|>
  ```

  Only `messages[0]` is treated as the system turn; a `developer` role is emitted literally, so
  render the activation on `system` — the same single-field change
  `providers/policy/qwen3.py` already documents and for the same reason. After 200 throwaway
  iterations the model emitted that envelope 64/64 times, so the transport is not the risk.
- Real V10 rows tokenize to ~676 tokens under LFM2.5 against ~588 under FunctionGemma; the smaller
  65k vocabulary costs ~15% more tokens per row. Both sit well inside `max_seq_length: 1024`.
- **`batch_size: 8` is already optimal** on an M5 Max. Sequences/sec by batch size: 4 -> 14.0,
  **8 -> 15.1**, 16 -> 11.4, 32 -> 12.9. Larger batches are slower per sequence and peak memory
  never approaches the limit (15.7 GB against a 115 GB working set). Do not raise batch size hoping
  for speed.

### Licence gate before any bundling

LFM2.5 ships under `lfm1.0`: Apache-2.0 verbatim plus a Section 5 that conditions **Commercial Use**
on the user not reaching a **$10,000,000 annual revenue Threshold**, with automatic termination
under Section 11 on non-compliance. Gemma's terms carry no revenue cap.

This repository currently commits trained weights at
`src/android_ui_analyser/resources/functiongemma/adapters.safetensors` (30 MB). An LFM2.5-derived
adapter must **not** be bundled the same way, because the cap would then land on every downstream
user. Follow the existing unbundled pattern instead: `models.qwen3` and `models.gemma4` ship zero
weights and require an operator-supplied absolute local directory (`model_path: null`,
`adapter_path: null`, and the provider never resolves or downloads a repo id). An `lfm2` provider
built that way distributes no LFM2.5 bytes, and each operator accepts `lfm1.0` for themselves.

## 6. Suggested order of work

1. Write the gate test from item 3.2 first, and watch it fail against the current generator. It is
   the regression test for the whole skew problem and it should exist before anything is regenerated.
2. Decide narrow-versus-broad (item 3.1). This decision determines the corpus, so make it explicitly
   and record it here.
3. Regenerate under the chosen contract with the inverted refusal ratio (item 3.3). Re-run the
   existing composition, group-isolation, token-length and privacy gates.
4. Add the `lfm2` policy provider, unbundled, mirroring `qwen3.py`.
5. Train one seed on LFM2.5-350M at 0.56-1.0 passes (~2.7-4.8 h locally, no RunPod spend needed for
   a first read) and score on an independently authored probe with multiple wordings.
6. Only then consider a second seed, a capacity challenger, or promotion. Promotion still requires
   zero safety errors and worst-family accuracy, never aggregate accuracy or validation loss.

## 7. Open questions

- Was the skew present when V8 and V9 were trained, or introduced by the post-V10 observation
  trimming? Public git history was squashed at `9ade000` (2026-08-20) and the V10 corpus was built
  2026-08-19, so this cannot be answered from this repository. It does not change what V11 must do.
- Is `known_screen` ever worth showing the model? It appears in live contexts and in **zero**
  sampled training rows. It is the one field where the runtime is richer than the corpus.
- `another_candidate_has_stronger_goal_overlap` is still deliberately excluded from the terminal
  refusal set and still has no live case, per the V9 incident record.
