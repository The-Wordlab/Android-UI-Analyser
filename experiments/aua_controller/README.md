# Gemma and Liquid AUA controller comparison

Research checked 2026-09-11. This experiment evaluates existing instruction-tuned weights before
choosing a training base. **Self-hosted inference and AUA execution run on the local Mac;
only training uses Runpod.** A separately requested OpenRouter comparison evaluates hosted APIs
through the same controller. The local launcher uses MLX on Apple Silicon. These scripts provision
no infrastructure and do not change the production AUA policy. The earlier Runpod inference run
is retained as historical evidence.

The [OpenRouter investigation](OPENROUTER_RESEARCH.md) records current candidates, prices and
measurement requirements. Its [separate manifest](openrouter-comparison.json) pins provider routes
and candidate settings; these are distinct from the original local/cloud comparison configurations.
The [completed hosted pilot](OPENROUTER_STATUS.md) favors **V4 Flash 0731 / DeepInfra** as the next
low-cost candidate: 3/3 prepared tasks, 11.20 seconds mean model time and $0.001375 per verified
completion. All 16 trial outcomes, provider failures and the separate retry are disclosed there.

**Real applications (no authored contract):** the fixture runner cannot accept a finish on a
real app, so [REALAPP.md](REALAPP.md) adds a claim-then-judge mode: `run_realapp.py` stops on
the model's `session_finish` claim, compacts every model-facing frame (`compaction.py`, ~6× smaller
on real screens), and asks a bounded second-window judgement (`judgement.py`) for a labelled,
unverified pass/fail plus screen names for the map. Same model, separate token window and budget.

**Local result: Gemma E4B with reasoning enabled and `compact-v1` passed 3/3 prepared AUA tasks
on the M5 Max in BF16.** Model HTTP time was 27–56 seconds per task, with full runner times of
68–115 seconds including setup. Compose recovered after one refused early finish. See
[local results, runtime fixes and limitations](LOCAL_STATUS.md).

The four models and their verified revisions are in [comparison.json](comparison.json).
The historical Runpod retry loaded and served all four, and **30 actual model runs** completed across four
separate configurations. Its promising result was **Gemma E4B with reasoning enabled and the
restricted `compact-v1` interface: 3/3 verified completions**, one each for classic sorting,
Compose sorting and error recovery. The same interface with reasoning disabled scored 0/3;
Liquid 2.6B scored 1/3. These are single trials on public fixture tasks, not generalization proof.

No training has run. The original 36-run comparison remains incomplete: 12 common-configuration
runs completed, while the remaining GPU budget went to separate diagnostic configurations.
The GPU was deleted, artifacts retrieved and zero ongoing spend verified. See the complete
[results and limitations](STATUS.md) and [earlier model diagnosis](DIAGNOSIS.md).

## Why these four

| Candidate | Purpose | Important distinction |
| --- | --- | --- |
| Gemma 4 E4B-IT | Smaller Google candidate | About 8B full checkpoint parameters; E4B is an effective-size label |
| Gemma 4 12B-IT | Higher-capacity reference | New `gemma4_unified` architecture, not the E4B architecture |
| LFM2.5-2.6B | First inexpensive Liquid candidate | Recent post-training includes complete agent harness interactions |
| LFM2.5-8B-A1B | Efficient MoE comparison | 8.3B total / 1.5B active; reasoning-only, so count generated reasoning |

Compare AUA semantic observations first. AUA can provide OCR/grounding results on opaque screens;
native image input is a separate follow-up experiment. Do not give only one candidate screenshots,
hidden assertion selectors, a successful prior trace, or deterministic action hints.

Gemma 4 is Apache 2.0. Liquid's LFM Open License v1.0 has a commercial-use revenue threshold of
$10 million; applicable commercial use needs separate terms above that threshold. Account for
the intended deployment before selecting a permanent training base.

## Gate 1: local serving protocol

Use the installed MLX-LM 0.31.3 / MLX 0.32.0 environment on Apple Silicon. The first local candidate
is the exact official Gemma E4B BF16 snapshot from the cloud pilot; this avoids changing weights or
quantization while measuring the Mac. MLX loads its language tensors directly and omits unused
vision/audio tensors for text-only inference. The launcher also applies the narrow
[upstream shared-KV loading fix](https://github.com/ml-explore/mlx-lm/commit/df1d3f3c9a7aae402dcbb8f41d4c36bcc13a50ae):
the exact pinned checkpoint includes 54 redundant K/V tensors in shared-attention layers that the
runtime does not use. Active weights still load strictly; on-disk weights stay unchanged. No
conversion or fine-tuning is required.

Download the pinned checkpoint once with the Hugging Face CLI:

```bash
hf download google/gemma-4-E4B-it \
  --revision ee0ef6023621cff504d758262d4e04895a5af4a2 --max-workers 2
```

Retain a matching Hub `blobs=true` identity record, or reuse the saved complete snapshot identity
from the earlier pilot. The [local launcher](serve_local.py) verifies every declared model,
tokenizer, configuration and template hash before loading:

```bash
python -m experiments.aua_controller.serve_local \
  --snapshot-dir /absolute/path/to/pinned/snapshot \
  --identity-file /absolute/path/to/saved-hub-identity.json \
  --receipt runs/aua-controller/local-runtime.json \
  --port 18000 --execute
```

Without `--execute`, this verifies files and records a dry run without loading MLX or the model.
The model loads inside MLX's generation worker, following its native thread/stream lifecycle.
The local endpoint binds to `127.0.0.1`, serves only the pinned `default_model` alias, enables
reasoning and permits one request at a time. It uses the native Gemma template/parser, with strict
rejection of malformed or multiple native calls. It never treats a parseable prefix of a broken
call sequence as an executable action. No remote-code trust, cloud fallback or training is enabled.
Keep this development server on loopback and stop its process when finished.

Run the existing two-turn protocol probe first:

```bash
python -m experiments.aua_controller.protocol_smoke \
  --base-url http://127.0.0.1:18000/v1 --model default_model \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --max-tokens 4096 --timeout 120 \
  --output runs/aua-controller/local-protocol.json
```

The alias is significant: the stock MLX server can resolve request-supplied model names, so the
local launcher rejects names other than `default_model`. Weight identity lives in the separate
verified runtime receipt. Preserve raw usage: MLX reports prompt and completion totals but does
not report a separate reasoning-token count. Missing reasoning usage must remain unknown.

The earlier Runpod experiment used vLLM 0.29.0 with native Gemma/Liquid parsers, an H100, BF16 and
32K context. Its artifacts and HTTP 429/PATH corrections remain documented in [STATUS.md](STATUS.md).
[serve_model.py](serve_model.py) retains historical cloud launch plans. Its execution path is
disabled by the current manifest's local-only inference policy. The user's clarified policy reserves Runpod for training.

The v2 probe asks for a native tool call, returns a freshly generated value through a real tool-role
JSON message, and explicitly requires extraction of its inner `value` field for the next native
call. This resolves the v1 instruction ambiguity with Gemma's native string-valued response wrapper;
it still rejects copying the outer JSON string. It preserves reasoning fields
within the tool exchange. Text that merely looks like a tool call does not pass. There are only two
requests and no retries. The default 1,024-token ceiling and temperature zero are protocol diagnostics;
record truncation and rerun with a disclosed larger ceiling if the native reasoning mode needs it.

**Passing proves only two-turn tool-protocol compatibility.** It neither verifies checkpoint identity
nor measures an Android task. Retain the serving process's pinned revision/image metadata separately.
The report leaves `aua_task_success` unset and `checkpoint_verified` false deliberately.
All four candidates passed the corrected protocol during the retry. Original failed probe artifacts
remain separate; correcting a diagnostic instruction is not a model-weight improvement.

Both families' direct Hugging Face templates expect tool arguments as mappings. Chat-completions
responses normally encode arguments as JSON strings: the serving adapter must normalize those before
applying the native template. Never reuse FunctionGemma's output parser for Gemma 4 or Liquid.

## Gate 2: shared live AUA pilot

Reuse [the public fixture, contracts and evaluator](../../evals/agent_loop/README.md). Start with
classic sorting, Compose sorting and retry recovery. Three
repetitions across four models give **36 planned runs**, not 36 independent scenarios. Add canvas
reveal after the perception path is verified, and notification denial after equivalent platform
permission reset is verified. Lease
contention and cached candidate reuse are later system tests; they would confound the first model
comparison. This small public suite is a development pilot, not an unseen promotion benchmark.
Only the first common 32K repetition completed for each candidate (12 runs). Do not substitute the
9 earlier 16K runs, 6 restricted-interface runs or 3 thinking-enabled runs for the 24 unrun repeats.

The runner must use the public AUA MCP surface with a normally acquired sticky session lease.
Do not call a private dispatcher or native Android tools: the MCP boundary publishes element IDs,
handles caller-turn freshness, trims observations, journals calls and decorates tool results.
Models receive the original goal, available tools, observations, prior actions/results and progress.
The harness owns the contracts, independent checks, artifacts, limits and cleanup.
This is oracle-assisted control: host-authored progress and assertion counts remain visible, while
hidden assertion selectors and deterministic action hints are stripped. Image bytes are withheld
from every model; some nested `image_path` metadata survives the current filter. Do not describe
this as independent model-authored verification or claim every artifact path is removed.

Reset before each run and verify the starting state. Notification denial also changes platform
permission state; an in-app reset alone does not prove equivalent permission state between runs.
Use AUA-owned reset/restore handling for that state, and mark the scenario blocked if reset cannot
be verified. Alternate model order across repetitions to reduce warm-cache/order confounding.

Use only the fixed relevant subset of AUA tools, with actual current schemas. Include observe,
tap/input/scroll/wait/back, checkpoint verification and finishing; preserve action-returned
observations rather than unconditionally spending another analyze call. Do not preselect the
correct element or route for the model. Enforce step, output-token and elapsed-time limits outside
the model. A model can request completion; it cannot set the benchmark's passed state.

Require independent evidence before crediting completion. `session_finish` returning `ok:true`
with `finished:false` is not completion. Missing verifier results are unknown, not zero false passes.
Keep cleanup checks and resolved evidence separate from the model's textual verdict.

The implemented runner resets the fixture under an AUA lease, verifies the start and device
identity, obtains current public MCP schemas, preserves action-returned observations and full
history, and enforces request, step and elapsed-time limits. It records model turns, usage,
request timing, MCP calls and the final result in an ignored output directory. An oversized
conversation fails explicitly instead of silently clipping history. Completion requires
`finished:true`, `terminated:true` and a passing independent verifier, including cleanup.

The verifier replays the authored [pilot contracts](contracts/) against recorded observations,
checks ordered automatic checkpoint proofs and session identity, and resolves observation and PNG
artifacts. Sorting contracts include restored name order as well as returning home. Later evidence
that undoes cleanup fails verification. Missing evidence remains unknown. This is an observation
contract oracle with artifact checks, not a separate vision model judging screenshot pixels.

With the public fixture installed and an AUA-capable device available, run one model/scenario
from the checkout with the AUA/MCP Python environment active:

```bash
python -m experiments.aua_controller.run_live \
  --base-url http://127.0.0.1:18000/v1 \
  --model gemma4-e4b --served-model default_model \
  --scenario classic-sort \
  --output runs/aua-controller/gemma4-e4b-classic-sort-r1
```

The runner acquires and cleans up its own headed AUA sessions. Choose a new empty output directory
for every run. Candidate IDs and their chat-template settings come from `comparison.json`;
`--served-model` supports an endpoint alias but does not prove weight identity. Repeat for the
three pilot scenarios and three repetitions, rotating candidate order. The current runner defaults
to 32 steps, 180 seconds of model interaction, a 60-second request timeout and 1,536 output tokens.

The opt-in `--profile compact-v1` exposes eight restricted public schemas: observe, ID-only tap,
input, swipe, wait, key, progress and finish. It removes `has`/`expect` and filtered queries,
clarifies JSON results and home-before-finish, provides up to three bounded schema repairs and
compacts finish refusals. Invalid calls never execute. Before finish, the host captures a fresh
complete hierarchy and PNG under the remaining interaction deadline; the independent verifier is
unchanged. This is a different controller configuration, not a transparent baseline optimization.

The local reproduction of the promising Gemma configuration uses fixed budgets across tasks:

```bash
python -m experiments.aua_controller.run_live \
  --base-url http://127.0.0.1:18000/v1 \
  --model gemma4-e4b --served-model default_model --profile compact-v1 \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --max-tokens 4096 --time-limit 180 --request-timeout 60 \
  --scenario classic-sort \
  --output runs/aua-controller/gemma4-e4b-thinking-classic-sort-r1
```

The historical cloud thinking lane used 90/30s, 60/20s and 30/15s interaction/request budgets.
Those results remain separate from local runs, which use 180/60s for every task. Full `analyze_screen` responses still produce oversized state;
uniform semantic state/history compaction remains unresolved. Three successful short flows do not
prove that long conversations fit the context window.

Replay the independent oracle without a device or model:

```bash
python -m experiments.aua_controller.verify_run \
  runs/aua-controller/gemma4-e4b-classic-sort-r1/aua classic-sort
```

Do not use the older agent-loop evaluator's `completed` field alone as acceptance. Normalize MCP
call names when aggregating its metrics: `analyze_screen` is not counted by its exact `analyze`
matcher. Aggregate model usage/timing and actual billed cost from the run and deployment records;
the existing bundle scorer does not calculate those totals.

Report:

- verified task completions / attempted and valid-start runs;
- false success claims, unknown outcomes, handoffs and cleanup failures;
- invalid native calls, unknown/stale targets, repetitions and successful recovery;
- model requests, input/output/reasoning tokens, AUA calls and unnecessary observations;
- cold start, model-request and end-to-end p50/p95 time, with small-sample limits stated;
- total billed GPU/storage time and cost per verified completion, including failures and idle time.

Record precision, sampling, context and output budgets per lane. Gemma's thinking-disabled lane
tests the fast configuration; if weak, evaluate a separately named thinking-enabled lane before
rejecting its capacity. Liquid's native reasoning cost must remain visible. Do not compare only
tokens per second or mix vendor benchmark scores collected under different harnesses.

## Paid resource lifecycle for training

`runpodctl` 2.14 does not provide a reliable pod TTL. The old `--stop-after` and
`--terminate-after` flags were removed because the backend accepted their values but never
enforced them; see [Runpod's removal and production evidence](https://github.com/runpod/runpodctl/pull/330).
Use a bounded spending plan, explicit pod teardown after artifact retrieval, and an independently
running watchdog that attempts teardown at the chosen cutoff. A watchdog depends on its host and
API access; it is not a server-enforced hard deadline. Verify the pod is stopped or deleted through
the control plane and account for retained storage charges. The runner's time limit and stopping
the vLLM process do not stop GPU billing. No watchdog or teardown automation is implemented in
these experiment scripts.

## Integrating an existing QA suite

`agent_loop.run_agent` is the reusable native-tool loop for application-owned harnesses.
The caller supplies the transport, tool schemas and execution callback, prompts, initial
observation, output directory and budgets. The caller also owns device acquisition,
persona and feature setup, evidence capture, independent review and teardown.
Application-specific scenario adapters belong with that application's QA suite.

The loop executes one validated tool call at a time. A well-formed response containing
multiple calls executes none, returns an error for every native call ID and asks for a
single call within a shared three-repair budget. Malformed envelopes, truncated output,
missing cost receipts and request timeouts stop execution. There is no automatic HTTP
retry. Model response cost is checked before executing its proposed action, and the
reported-cost limit is checked before another model request. Host-selected work still
obeys the shared action, time and step limits; it does not consume a model-request
budget. An unanswered request may incur an unknown upstream charge.

Each actual tool result receives a local evidence reference and hash. These references
prove which observation was recorded, not that the model's claim is correct. A successful
terminal submission ends the controller but remains an untrusted draft. Independently
check the complete authored contract, including visual claims, before assigning a verdict.
Keep every attempt and its implementation snapshot separate when fixing integration bugs.

## Training after the baseline

The completed pilot nominates Gemma E4B thinking-on + compact-v1 for further evaluation, not a final
training base. Repeat the promising configuration with fixed budgets on unseen app/task/build cases
and address oversized observations before collecting a training corpus. No training has started.

The initial data must contain complete, independently verified action/result trajectories, including
recovery, refusals/handoffs and stopping. Keep validation/test task families separate; do not train
on the fixed public pilot and then report it as generalization. Freeze an independently authored
unseen app/task/build suite before selecting a final checkpoint.

Use a Runpod training environment separate from the local Mac inference environment. First verify a forward/backward/save/reload
smoke, supervision masks, actual trainable modules and optimizer-update accounting. Falling token
loss is not an acceptance criterion. Pin model/tokenizer/data revisions and count microbatches and
optimizer updates separately. Retrieve and hash artifacts before explicit teardown, and apply the
budget and watchdog procedure above to every paid job.

Gemma's current Transformers/PEFT/TRL path supports fine-tuning, but the generic tutorial does not
establish every 12B default: PEFT 0.20.0 has `gemma4` LoRA defaults and lacks `gemma4_unified` defaults.
Inspect actual module matches and explicitly target language projections for the 12B smoke.
Do not unnecessarily retrain embeddings or the output head merely because a tutorial does so.
Liquid has a separate official fine-tuning project; its training dependency pins differ from the
current serving recipe. Do not silently combine those environments or assume dense LoRA settings
cover MoE expert tensors. In its current Transformers implementation the MoE expert projections
are 3D parameters, so `target_modules="all-linear"` alone does not adapt them. Inspect and disclose
which tensors receive adapters; use explicitly supported parameter targeting only after a smoke
test. Load the actual pinned `chat_template.jinja`: hosted metadata can differ, and Liquid's 2.6B
template prefills a thinking block. No training configuration is marked GPU-verified here.

## Primary sources

- [MLX-LM 0.31.3 release](https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.3)
- [MLX native Gemma loader](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/gemma4.py)
- [MLX native server](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/server.py)
- [Google Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Google QLoRA guide](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)
- [vLLM Gemma recipe](https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html)
- [vLLM 0.29.0 model registry](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/model_executor/models/registry.py)
- [PEFT 0.20.0 model mappings](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/constants.py)
- [Liquid 2.6B release and training approach](https://www.liquid.ai/blog/lfm2-5-2-6b)
- [Liquid 8B-A1B release](https://www.liquid.ai/blog/lfm2-5-8b-a1b)
- [Liquid 2.6B model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B)
- [Liquid 8B-A1B model card](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B)
- [Liquid GPU inference guide](https://docs.liquid.ai/deployment/gpu-inference/vllm)
- [Liquid fine-tuning project](https://github.com/Liquid4All/liquid-finetune/tree/5950740f864f9580dd8b3b8218054b3a46ca9446)
- [Liquid MoE expert implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/lfm2_moe/modeling_lfm2_moe.py)
- [Liquid license](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/LICENSE)
- [Runpod removal of nonfunctional pod deadline flags](https://github.com/runpod/runpodctl/pull/330)
## Host execution and session context

`agent_loop.run_agent(..., session_state=...)` can retain a host-owned `SessionState`
across calls. The ledger keeps authored check IDs, explicitly untrusted claims,
compact observation history and attempts without screen progress. Initial host
context is supplied once per invocation. Subsequent results append only changed claim fields
and host facts; earlier messages, observations and native reasoning remain intact.
The request history is append-only, preserving its reusable prefix. The request
byte limit refuses an oversized model handoff rather than silently truncating history.
The append-only correction has regression coverage; the earlier pilot measurements
do not measure this revision's cache or cost impact.
Callers may expose `evidence_recall.read_recorded_evidence` with an explicit
namespace-to-controller-root mapping. Recalled records are historical and never
become current action targets.

The optional asynchronous `host_next(latest_result)` runs before each model
opportunity. It receives a detached copy of the latest raw result and returns either
`None` to ask the model, or one `HostAction` with `tool`, `arguments` and `reason`.
Host choices use the same offered schemas, execution callback, no-progress guard,
evidence records and time/step limits as model choices. They confer no additional
tool authority. Known work can proceed without constructing or sending a model
request. Host actions enter the conversation as explicitly labelled host events;
the loop never invents an assistant tool call for them. Reports separate host
decisions, host/model callback executions and model requests. A flow callback can
contain several native steps, so callback counts are not device-action counts.

`model_observation_filter` optionally narrows each new model-facing result after
privacy filtering. It does not change raw evidence, host `SessionState`, native
execution results, evidence eligibility or earlier conversation messages. The
opt-in `flow_projection.compact_flow_observations` retains one recognized native
flow observation with the ordinary action fields. It removes legacy elements only
when they are an exact ordered subset of that frame. Conflicting or distinct
diagnostic summaries remain, as do failure details, progress and freshness flags.
This reduces payload width; it does not summarize away the session history or
establish that a flow succeeded.

`AgentConversation` optionally retains the exact transcript across explicit
invocations. A caller creates one holder for one AUA session, supplies each new
objective and current observation, and gives evidence records distinct namespaces.
The holder checks system/model/backend/request-setting consistency, preserves native
reasoning and complete assistant/tool pairs, rejects concurrent reuse, and refuses
reuse after a failed or interrupted invocation. Device/session identity and lifetime
remain caller responsibilities. The continuous-group integration is wired and unit
tested; no completed live group validates it yet. This is separate from the ledger.

`route_coordinator.discover` reads AUA's `orient`/`flow_list` knowledge and pins
only caller-approved flow files. Optional `flow_parameters` are copied into immutable
host bindings and are not exposed in the model catalogue. Candidate selection uses opaque IDs; native
`flow_run` retains preflight/divergence ownership, with destructive actions and
hidden planner assistance disabled. Candidate preview uses native
`session_candidate_flow` without replay or promotion.

The companion experimental AUA runtime supports an authored checkpoint obligation:

```yaml
required_input:
  after_checkpoint: response_ready
  target: {rid: composer}
  text: "Unsent input check"
```

This supplements the checkpoint's ordinary assertions. Only an actual unsent native
input after the named completed prerequisite, followed by that input's own fresh
matching field/text readback, can supply its causal receipt. Seeing the text later,
manual completion, or a model claim cannot substitute. A bound input during flow
replay obtains its own observation; ordinary inputs retain their existing behavior.
Use the matching native runtime revision: the generic loop does not emulate this
contract field, and older parsers reject it. Synthetic values are explicitly authored;
session contracts do not perform flow-style parameter interpolation.

Native candidate previews now expose exact checkpoint step boundaries. Required
captured input parameters stay empty until the caller supplies reviewed values.
A caller may bind a reviewed candidate to one current native checkpoint, attempt it
once, and hand a divergence back to the adviser. The convenience candidate replay/save
API rejects missing parameters before executing its reset. Phase candidates remain
unverified until their separate replay and independent review; there is no automatic
promotion or universal natural-language planner. Full multi-phase replay does not
yet update native progress at every internal step. The current experiment establishes
the active-checkpoint input path, not general replay correctness.
The revised native flow path discards a supplied pre-action frame after initial
device offload. Rich assertions return the exact hierarchy frame checked by their
final poll; legacy predicates mark their carried frame stale, and the next semantic
action must refresh it. Authored contracts keep dispatch on the host so required
input receipts cannot be bypassed by offload. These paths have focused regression
coverage; live phase replay remains unverified.

Native phase recommendations also support published opaque element IDs. An optional
recommendation failure preserves completed checkpoint proof and returns a
nonexecuting fallback, instead of dropping progress after a successful action.

These are new execution configurations, distinct from the historical model-led
fixture pilots above. The caller owns authored mappings, current phase scope,
verification, session lifecycle and cleanup. Native state/input proofs support
independent review; neither a completed checkpoint, checklist claim nor route is
an automatic semantic task verdict.
