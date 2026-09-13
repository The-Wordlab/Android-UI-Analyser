# Validation record: 2026-09-12

The retry succeeded: all four pinned candidates downloaded, loaded and passed native tool-protocol
checks on Runpod. **30 actual model runs completed. Gemma E4B with reasoning enabled and the
restricted `compact-v1` interface passed all three public fixture tasks.** No weights were changed
and no training ran. This is a promising controller configuration, not a validated general tester.

Current execution policy, clarified after this pilot: **inference and testing run on the local Mac;
only training uses Runpod**. The cloud results below are historical. The subsequent
[local Mac pilot passed 3/3 tasks](LOCAL_STATUS.md). See [README.md](README.md) for the local
inference workflow; the cloud launcher is disabled by the current manifest policy.
The user subsequently requested a separate [OpenRouter API investigation](OPENROUTER_RESEARCH.md);
this does not re-enable Runpod inference or change the historical results below.

## Results by configuration

Every lane used BF16, concurrency one, native model templates/parsers and the same three authored
public fixture tasks. Results require independent observation-contract verification and cleanup.
Do not pool these configurations into a single model ranking.

| Configuration | Gemma E4B | Gemma 12B | Liquid 2.6B | Liquid 8B-A1B |
| --- | ---: | ---: | ---: | ---: |
| Diagnostic: baseline, 16K context, 1,536 output, three trials per task | Not run | Not run | 1/9 | Not run |
| Common: baseline, 32K context, 4,096 output, one trial per task | 0/3 | 1/3 | 1/3 | 1/3 |
| Restricted: compact-v1, 32K/4,096, one trial per task | 0/3 | Not run | 1/3 | Not run |
| Reasoning enabled: compact-v1, 32K/4,096, one trial per task | **3/3** | Not run | Not run | Not run |

Gemma reasoning was disabled in the common/restricted rows and explicitly enabled in the last row.
Liquid retained its native reasoning behavior. All lanes allowed 32 model steps and 100,000 request
bytes. The first three allowed 180 seconds of interaction and 60 seconds per request. To remain
inside the cloud cutoff, the final lane used different interaction/request budgets: classic 90/30s,
Compose 60/20s, recovery 30/15s. Actual requests confirm the effective thinking setting; the candidate
registry's default remains false. These differences are recorded in [comparison.json](comparison.json).

The original plan was 36 common-configuration runs. Only its first 12 completed; **24 repeated
common runs remain unexecuted**. The remaining time funded 18 separately identified diagnostic runs.
Protocol probes and scripted harness checks are excluded from all model-run counts.

## Successful Gemma configuration

| Task | Native model calls | Model HTTP time | Overall time | Output tokens, including reasoning |
| --- | ---: | ---: | ---: | ---: |
| Classic price sorting, name-order restoration and home | 5 | 11.42s | 52.07s | 1,718 (1,512 reasoning) |
| Compose price sorting, name-order restoration and home | 5 | 13.34s | 52.52s | 2,058 (1,859 reasoning) |
| Error, retry, ready state and home | 4 | 5.64s | 45.85s | 754 (633 reasoning) |

All 14 native calls succeeded with zero tool errors, schema repairs, rejected finishes, truncations
or text-only completion fallbacks. Each model run requested `session_finish` after restoring the UI;
the host captured a fresh complete observation before accepting completion. Strict v2 replay passed
for all three without changing their original results. Nine representative screenshots were visually
inspected: both sorting orders preserve the exact name/price pairings, recovery shows the ready state
without Retry, and every flow returns to fixture home.

Combined model HTTP time was 30.40s, with 4,530 output tokens including 4,004 reasoning tokens (88.4%).
Across these 14 requests, median response latency was 1.72s and p95 3.59s; the sample is small. Setup
alone consumed 97.60s of the combined 150.45s overall time. H100 latency does not establish the same
latency on a cheaper GPU, production throughput, or a sustainable per-test price.

`compact-v1` offers eight restricted schemas derived from actual public AUA tools. It removes
`has`/`expect`, allows ID-only taps, clarifies JSON results and home-before-finish, provides at most
three invalid-schema repair opportunities, compacts finish refusals and obtains a fresh full finish
observation. The baseline remains available and unchanged. Schema JSON shrank from 15,561 to 3,470
bytes; the initial E4B prompt shrank from 5,123 to 2,645 tokens. This bundles several interface
changes, so the experiment does not isolate each one's contribution. No schema repairs were used
in any of the nine restricted-interface runs.

## Why other attempts failed

- **Context and output limits:** the first Liquid lane had four context overflows and two reasoning
  truncations. Raising context to 32K did not prevent all overflows. Full `analyze_screen` results
  still contain about 25 KB / 10.6K tokens, versus about 5–6 KB for action-returned state; repeated
  full observations exhaust context even on these short tasks. Request-byte and token caps are
  distinct constraints.
- **Tool and planning errors:** models attempted text input on buttons, combined mutually exclusive
  selectors, reused stale/invented IDs and repeated invalid assertion calls. Gemma 12B made 31 tool
  errors in its 32-call Compose run. Larger capacity alone did not solve this interface.
- **Completion errors:** some models claimed success or emitted textual pseudo-tool calls before
  cleanup. These were rejected. Gemma E4B without reasoning also treated a finish refusal as success
  in one restricted run. There was no accepted, proven false pass; that does not mean no incorrect
  success claims occurred.
- **An unresolved evidence gap:** one diagnostic Liquid recovery run passed all checkpoint assertions
  and AUA reported completion, but a subsequent filtered query had no elements or fresh fingerprint.
  It remains unverified and is not counted as a pass or a proven false pass.

The final reasoning-enabled result shows that pretrained weights can already operate AUA on these
flows. It does not prove reasoning is the only missing ingredient, that compacting alone improved
success, or that all earlier model failures have been resolved. See [DIAGNOSIS.md](DIAGNOSIS.md) for
the earlier local models' training-target, curriculum, exposure and generalization failures.

## Serving and harness corrections

The first A100 attempt received Hugging Face HTTP 429 before weights loaded. The same pinned config
and token worked locally. Host egress/CDN throttling was a leading explanation; the exact quota
policy was not established. This was a setup failure, not a model capability or training failure.
That pod was deleted and teardown verified; estimated GPU time cost was about $0.21, plus storage.

The H100 retry prefetched all four complete snapshots with authentication and bounded retries,
verified their saved Hub hashes and revisions, and served offline. A separate initial server failure
was a missing `ninja` executable on the process PATH; adding the inference virtual environment's
`bin` directory fixed it. GPU execution and all four model servers then worked.

The verified environment used vLLM 0.29.0, Torch 2.13.0, Transformers 5.17.0 and Hugging Face Hub
1.31.0. Base image: `runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
Full package, launch, model/tokenizer/template and file-hash identities are retained privately.
The endpoint bound to loopback through SSH; unauthenticated requests returned 401 and authenticated
requests succeeded. During this historical cloud pilot, no candidate weights, inference or
training ran on this Mac; the later local inference run is recorded separately above.

Protocol v1 ambiguously asked Gemma to copy a value returned as JSON text. Its native template wraps
that text inside a string-valued tool response, and it copied the outer JSON string. Protocol v2
explicitly asks for the inner JSON field; the same model server passed, with no weaker value check.
All candidates passed the corrected protocol. This tests two-turn tool compatibility, not AUA ability.

Verifier v1 rejected AUA's documented top-level `goal_progress` decoration. Version
`aua_contract_replay_v2` normalizes only that decoration for schema parsing and retains strict
identity, ordering, freshness, UI-state and artifact checks. Separate replay files identify immutable
original results by SHA256. The filtered-query evidence gap remains unverified under v2.

Before model runs, three scripted headed fixture flows passed and a premature finish was rejected.
Those checks exposed and corrected short resource-ID ambiguity, missing image capture and Compose's
separate text-leaf representation. They are recorded as `model_measured:false`, not model successes.
All sessions used normal AUA leases; capture configuration was restored and sessions released.

Final focused verification: **165 tests passed, 0 failed**, including controller, protocol, verifier,
Compose contract, offline serving and worktree checks. Ruff and whitespace checks passed.

## Limits and next improvement

This is **oracle-assisted control**: the host supplies authored goals/contracts, progress and
assertion counts, while withholding hidden assertion selectors and action hints. Candidates saw
semantic text state, not image bytes. Some nested `image_path` metadata remains in model-facing
results; the filter does not remove every artifact path. The verifier checks recorded observations
and real artifact files, not screenshot semantics with an independent vision model.

There are only three public tasks and a single thinking-enabled trial per task. No unseen app,
new build, permission dialog, opaque canvas or production workflow has been validated. Selection
followed inspection of earlier failures, so the final scores are development evidence.

Next, retain Gemma E4B thinking-on + compact-v1 as the promising baseline. Make full and action-returned
state consistently compact while retaining IDs, editable flags, text/price relationships, errors
and freshness; preserve full raw evidence outside the prompt. Then repeat with fixed budgets and
freeze unseen app/task/build tests, including no-good-action and recovery cases. Measure local quantization and
Mac throughput only after a reliable baseline. Train on Runpod only if remaining failures
justify it, using complete verified native AUA trajectories and separate validation/test families.
Do not train on these pilot flows and then call their score generalization.

## Resource closure and evidence

The H100 was created at 2026-09-11 22:54:41 UTC and deletion was verified at 23:54:27 UTC, before the
60-minute cutoff. Elapsed allocation-to-verification time was about 59m47s. At the quoted $3.49/hour,
GPU time is approximately **$3.48**, plus any storage charges; finalized billing was not available.
Model-run latency alone excludes downloads, startup, debugging and idle GPU time and is not total cost.

After retrieval, the provider returned pod-not-found, listed zero pods and zero network volumes, and
reported current spend of $0/hour. Temporary SSH and endpoint keys were removed; tunnel and watchdog
were no longer running. No paid resources from this retry remain.

Raw model turns, AUA traces, screenshots, immutable replay records, snapshot identities, serving
logs and aggregate metrics remain ignored/private under
`runs/aua-controller/20260912-retry/` in the main checkout. The final remote evidence archive contains
12 files, 52,290 bytes, SHA256 `a057d5b7fcbc6b6777bb72925ee40fe9613415ec3294714e7b22c0a0dbd6929e`.
No raw evidence, credentials or device/session identifiers are included in this report or committed.
