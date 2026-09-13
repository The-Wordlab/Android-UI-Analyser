# Why the earlier model did not become an AUA controller

Investigation recorded 2026-09-12; candidate research checked 2026-09-11. This diagnosis covers
the earlier FunctionGemma experiments and the first serving failure. Current pilot results and
resource status belong in [STATUS.md](STATUS.md); this document does not select a winner.

The evidence points to narrow training targets, synthetic-to-live generalization failures,
separate AUA compiler defects, a later decision collapse even on sampled training rows, and
insufficient verified trajectory data. It does **not** establish that a small, inexpensive AUA
controller is impossible or that increasing parameters alone fixes it.

## What we actually trained

The earlier roughly 270M-parameter FunctionGemma runs trained a **selector**. It received short
lists of prevalidated calls and emitted `select_candidate(candidate_id)`. AUA constructed arguments,
authorized actions, executed them, tracked proof and enforced cleanup. The model could not select
a missing candidate or independently perform the complete observation/action/result/verification
loop. That is useful as a guarded component, but it is a different learning target from the
requested CLI-style agent.
See the [original boundary, lines 11–15](../functiongemma/EXPERIMENT_LOG.md#L11-L15).

Later V11/V12 work in the same experiment directory changed both the model and contract. V12 used
Liquid's LFM2.5-350M and a projected screen/node state, emitting one of `tap(n)`, `scroll(dir)`,
`done()` or `handoff(reason)`. It was no longer a candidate selector, but remained a single-decision
policy: its contract deliberately omitted text input and model-issued assertions. It therefore
also differed from the complete native AUA controller now being measured.
See the [V12 contract](../functiongemma/v12_contract.py) and
[V12b training configuration](../functiongemma/train-lora-lfm2.5-350m-v12.yaml).

A CLI product combines a model with tool schemas, state management, retries, execution and evidence.
Its billing or use of an external coding agent does not by itself establish that its vendor trained
a proprietary small model. Our comparison should measure that complete AUA interaction loop.

## What failed, and what improved

**Synthetic accuracy overestimated live reliability.** V6 achieved 100% on an untouched synthetic
test and passed four simulated flows, yet its initial live advisory audit scored 1/4, compared with
2/4 for deterministic AUA. This exposed a mismatch between generated examples and actual ambiguous
UI text; it was not proof that the model could never learn the task.
[Results and interpretation, lines 99–133](../functiongemma/EXPERIMENT_LOG.md#L99-L133).

**Some failures belonged to AUA before inference.** The compiler compared every row with the whole
objective, including names of unwanted alternatives. Longer title/summary text could accumulate
false relevance. Extracting the requested destination corrected the four-case audit to 4/4;
the corrected compiler then offered one eligible action and bypassed the model in all four cases.
This is evidence for the compiler fix, not improved reasoning by unchanged weights. Other recorded
failures involved candidate recall and launch readback: a selector cannot recover an action that
never reaches its input. [Causal rerun, lines 141–176](../functiongemma/EXPERIMENT_LOG.md#L141-L176).

**Model errors remained after compiler fixes.** Independent V8/V9 probes found approximately 25%
answer changes under meaning-preserving rewording. First-position selection exceeded chance, while
opaque ID zero did not. A later V8 tournament selected the first position in all 144 pairwise
decisions; voting produced handoffs rather than useful semantic agreement. More inference calls
therefore did not repair that weakness.
[Wording/order audit, lines 705–751](../functiongemma/EXPERIMENT_LOG.md#L705-L751);
[consensus experiment, lines 949–967](../functiongemma/EXPERIMENT_LOG.md#L949-L967).

**The curriculum allowed shortcuts.** V9 confounded candidate count and action direction with
labels. In V10, 88.2% of handoff examples contained an unauthorized/destructive candidate; only
11.8% contained safe but irrelevant actions. Risk flags could predict refusal without understanding
whether an offered action advanced the goal. This distribution mismatch is a concrete defect;
the model's internal mechanism remains an interpretation, not a direct measurement.
[Generator audit, lines 757–790](../functiongemma/EXPERIMENT_LOG.md#L757-L790).

**The historical exposure counts were four times too high.** MLX iterations consumed one batch of
eight sequences; gradient accumulation controlled optimizer updates, not extra examples. Thus
8,192 iterations gave V9 about 0.99 corpus passes and V10 about 0.25. The longer 32,768-iteration
V10 run reached about 1.00 pass; its selected step-18,432 checkpoint was at **0.56 passes**, where
refusal reached 18/38 rather than the shorter run's best 0/38. The relative exposure comparison
survives, but the old labels of “one pass” and “four passes” do not. Refusal did emerge before one
real pass and remained noisy across checkpoints. Near-zero validation loss did not establish
generalization or memorization: V9 reached it at about 0.19 passes, before examples had repeated.
Scores from repeated checkpoints on the same small probe are correlated; their mean is descriptive,
not independent replication. The [V11 correction](../functiongemma/V11_HANDOVER.md#L11-L53)
supersedes the absolute pass counts in the
[older exposure results](../functiongemma/EXPERIMENT_LOG.md#L793-L839).

**V10 made real, bounded progress.** Its selected 0.56-pass checkpoint subsequently produced nine
live decisions, five executed outcomes and four refusal decisions, with zero wrong taps. It refused
an absent destination under two wordings. This supports a useful guarded selector on those cases;
it does not establish general autonomous testing. A larger Qwen3-1.7B challenger improved descriptive
accuracy/stability on the same probe but still had weak refusal after about **0.25 passes**, so
capacity alone was not the demonstrated solution.
[Live V10 and challenger results](../functiongemma/EXPERIMENT_LOG.md#L841-L900), with exposure
corrected by the [V11 handover](../functiongemma/V11_HANDOVER.md#L30-L48).

**V12b also failed on examples from its training distribution.** A fresh offline replay of the saved
LFM2.5-350M adapter scored **57/200 (28.5%)** on sampled held-out tap rows, versus the deterministic
rule's **175/200 (87.5%)**. These are tap-target scores, not overall task-completion rates. In a
separate 25-row sample for each non-tap class, `done`, `scroll`, `needs_host`, `no_progress` and
`target_absent` each scored 0/25 and always produced `tap`; `needs_auth` scored 7/25. A further
diagnostic sampled ten training rows per class: tap scored 5/10, `needs_auth` 1/10, and every other
class 0/10. Those training-row failures rule out an explanation based only on unfamiliar live
screens. They demonstrate poor learned decisions in this checkpoint; they do not isolate model
capacity, supervision, optimization or checkpoint selection as the cause.

The ignored audit artifacts are `runs/functiongemma/audit-20260911/v12-replay.json` and
`v12-train-diagnostic.json`; the replay records adapter SHA256
`a2d68c6d1cbc857c2aa7e03a881b6f64e6df6dc49ba84d641544bf28f78cde8f`. No device was involved in these
replays. The [V12b configuration](../functiongemma/train-lora-lfm2.5-350m-v12.yaml) also records a
separate correction to older learning-rate schedules: decay steps must count optimizer updates,
not microbatch iterations. That correction is documented; it does not establish why the resulting
V12b checkpoint collapsed toward taps. Check training behavior and saved-checkpoint reproduction
before treating more epochs or a larger model as the remedy.

**Raw log volume greatly exceeded trustworthy training labels.** The historical audit joined 8,003
events to 250 sessions and recovered 249 emulator episodes. Only 12 whole episodes had structured
proof for every completed phase, and only one historical model selection linked to immediate phase
progress. Failure and incomplete-session traces can define useful scenarios; they are not automatic
correct-action labels. [Data audit, lines 930–943](../functiongemma/EXPERIMENT_LOG.md#L930-L943).

## The HTTP 429 was a separate infrastructure failure

The first A100 serving attempt received Hugging Face HTTP 429 while fetching configuration, before
loading model weights. The same pinned file and valid token returned HTTP 200 locally. Host egress
or CDN throttling was a leading explanation, but missing quota headers prevent identifying the
exact limiting policy. This was a download/setup failure, not a model evaluation or training failure.
See the dated cloud evidence in [STATUS.md](STATUS.md).

The replacement-host identity records subsequently verified all four pinned downloads, including
Hub file hashes and resolved commits. That clears the download gate; it does not prove native tool
protocol compatibility, task completion or training readiness. Keep those gates separate.

## Why the current pretrained candidates are worth measuring

The [manifest](comparison.json) pins exact model/tokenizer revisions. These are candidate rationales,
not AUA scores. Official benchmark results and post-training descriptions are vendor evidence.

| Candidate | Question it tests | Constraint that affects interpretation |
| --- | --- | --- |
| [Gemma 4 E4B-IT](https://huggingface.co/google/gemma-4-E4B-it) | Can the smaller Google tool-using checkpoint run the complete loop? | E4B is an effective-size label; the full checkpoint has about 8B parameters. |
| [Gemma 4 12B-IT](https://huggingface.co/google/gemma-4-12B-it) | Does more capacity materially improve verified completion? | Uses `gemma4_unified`, a different architecture from E4B. |
| [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) | Does a smaller agent-oriented checkpoint provide adequate reliability at lower cost? | Its pinned native template prefills thinking; account for those tokens and latency. |
| [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) | Does sparse active computation improve the capacity/cost tradeoff? | Vendor describes 8.3B total / 1.5B active; reasoning-only. Full weights still occupy memory. |

Gemma 4 supports tool use and is Apache 2.0. Liquid describes complete agent-harness interactions
in its recent post-training; that has not established AUA performance. Liquid's Open License 1.0
includes a $10 million commercial-revenue condition, so its deployment terms differ from Gemma's.
[Google model card](https://ai.google.dev/gemma/docs/core/model_card_4),
[Liquid training description](https://www.liquid.ai/blog/lfm2-5-2-6b),
[Liquid license](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/LICENSE).

Use each native template and parser: Gemma's `gemma4`, Liquid's `lfm2` tool parser with `qwen3`
reasoning parser. Liquid's `preserve_thinking` retains reasoning history; it is not a reasoning-off
switch. Keep Gemma thinking-disabled and any thinking-enabled runs as named configurations. The
first comparison uses the same semantic text state for every model; native visual input belongs
in a separate experiment. [Google serving recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html),
[Liquid serving guide](https://docs.liquid.ai/deployment/gpu-inference/vllm).

## Improvement tied to the observed failures

1. **Establish the complete pretrained baseline before training.** Run the same public AUA schemas,
   goals, verified starting states, budgets and independent completion checks. A two-turn native
   tool probe proves protocol only. Three repeated public tasks remain a development pilot, not
   evidence of unseen-app generalization. Optimize cost per verified completion, including failed
   calls, generated reasoning and idle GPU time; tokens per second alone misses useful progress.
2. **Train complete native AUA trajectories.** Record the goal, compact structured state, exact
   tool call, returned observation, recovery, proof checkpoint and cleanup. Preserve action-returned
   observations so the model learns to use them instead of repeatedly requesting the same screen.
   Retain freshness, errors and unknown outcomes; compacting state must not erase relevant evidence.
   Preserve failed actions and their results when they provide context for a reviewed recovery;
   a successful episode does not make every action in it a correct training target. Label only
   verified successful actions or explicitly reviewed recovery/handoff choices.
3. **Remove the measured shortcuts.** Counterbalance wording, presentation order, opaque IDs,
   candidate counts and action direction within semantic families. Include many safe-but-none-good
   states, alongside closely matched states with one useful safe action. Teach a structured handoff
   for no useful action, uncertainty, stale state or exhausted budget. Test rewordings and order
   permutations jointly; disagreement is not evidence that more votes will resolve the task.
4. **Freeze independent app/task/build splits.** Keep related trajectories, paraphrases and derived
   examples in one split. Author unseen task families before checkpoint selection. Use validation
   for selection and a sealed test for the final decision; public pilot repetition or mining its
   failures into training cannot then support a generalization claim. Repeat promising training
   with another seed and account for optimizer updates and exposure per example.
5. **Let evidence decide completion.** Keep the [finish verifier](verify_run.py) outside the model.
   It must check authored assertions, ordered observations, actual artifact files and restoration
   before accepting finish. `ok:true` with `finished:false` is incomplete. Missing evidence means
   unknown, not zero false passes. This observation-contract oracle is independent of the model's
   claimed success, but is not an independent perception model judging screenshot pixels.
   The current controller also receives AUA goal-progress feedback. Its score therefore measures
   AUA-assisted execution, not independent model judgment of every assertion. Keep hidden scoring
   assertions out of model input and distinguish this assistance from any future lane that tests
   the model's own verification ability.
6. **Train only on Runpod, with separate training and serving environments.** Before a paid full
   run, verify forward/backward, supervision masks, real trainable tensors, optimizer updates and
   save/reload, then test the saved adapter through the native serving protocol. Gemma 12B needs
   explicit inspected LoRA targets because PEFT 0.20 lacks `gemma4_unified` defaults. Liquid MoE
   expert projections are 3D parameters: `all-linear` alone does not adapt them. Pin dependencies,
   model/tokenizer/data revisions and templates; retrieve hashed artifacts and verify teardown.
   [Google training guide](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora),
   [PEFT mappings](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/constants.py),
   [Liquid training project](https://github.com/Liquid4All/liquid-finetune/tree/5950740f864f9580dd8b3b8218054b3a46ca9446),
   [MoE implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/lfm2_moe/modeling_lfm2_moe.py).
