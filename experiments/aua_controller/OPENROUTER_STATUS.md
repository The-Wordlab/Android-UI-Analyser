# OpenRouter AUA pilot: 2026-09-12

**Start with DeepSeek V4 Flash 0731 through DeepInfra for the next evaluation.** It completed
all three prepared AUA tasks, averaging 11.20 seconds of model HTTP time and $0.001375 per
verified completion. Training was unnecessary for these tasks. Unseen-app reliability remains
unmeasured; the controller still uses authored goals, progress and independent verification.

The investigation and model/provider price snapshots are in [OPENROUTER_RESEARCH.md](OPENROUTER_RESEARCH.md).
The exact July model is `deepseek/deepseek-v4-flash-0731`, provider `deepinfra/fp8`, low reasoning.
It is distinct from the April V4 preview and September V4.1 Flash.

## Measured comparison

Every configuration used the same three public fixture tasks, `compact-v1` tools and the new
`hosted-v1` observation projection. A fresh local E4B control used that same projection.
The [earlier local results](LOCAL_STATUS.md) remain separate and unchanged.

| Configuration / provider | First trials | Mean model HTTP per verified run | Mean full verified run | Reported cost per verified completion |
| --- | --- | ---: | ---: | ---: |
| V4 Flash 0731 / DeepInfra FP8 | **3/3** | **11.20s** | **50.25s** | **$0.001375** |
| Gemma 4 26B-A4B / DekaLLM BF16 | **3/3** | 11.95s | 53.44s | $0.002229 |
| Mercury 2.5 / Inception | 2/3; one unverified | 7.04s | 47.93s | $0.002179 |
| V4.1 Flash / Novita FP8 | 2/3; one HTTP 429; separate retry passed | 8.30s, including retry | 47.26s, including retry | $0.003293, including failed attempt |
| Gemma 4 E4B / local MLX BF16 | **3/3** | 38.38s | 79.32s | N/A: local inference |

Timing averages include only verified complete runs; Mercury therefore averages two tasks.
Novita's timing averages use Classic retry r2 and Compose/async r1. Its interrupted r1 is not
treated as a fast completed test. Cost per verified completion divides **all task-attempt
charges**, including unsuccessful attempts, by verified completions; protocol charges are
separate. These small, differently sampled averages are descriptive, not a statistical ranking.

Model HTTP includes network, queueing, prefill and generation. Full runner time additionally
includes AUA setup/actions, evidence and cleanup, excluding download/model loading and outer
process startup. Setup averaged roughly 32–35 seconds per configuration and remains a large
part of total time. Local hardware/electricity cost was not measured.

There were **15 initial trials: 13 verified passes, one unverified and one provider-interrupted**.
One unchanged-settings Novita Classic retry then passed: **16 total trials, 14 verified passes**.
No model or harness code changed during the live comparison; saved source hashes were checked.

## Provider choice for V4.1

If V4.1 is required, **Novita is the working route verified here**, with an observed rate-limit
failure to account for. Its two-turn native protocol passed and all three task types eventually
completed. This does not establish dependable capacity or justify presenting first trials as 3/3.

| V4.1 route tested | Observed result |
| --- | --- |
| DeepSeek | HTTP 404: excluded by existing account data policy. Account settings were not changed. |
| Fireworks | Protocol attempts failed; diagnostic observed HTTP 429 from the shared upstream pool. One separate first-turn diagnostic succeeded. |
| DeepInfra | Protocol attempt failed; diagnostic observed shared-pool HTTP 429. |
| SiliconFlow | HTTP 404: OpenRouter could not route the requested native tool use. |
| Novita | Protocol passed; first AUA trials 2/3, then one separately recorded retry passed. |

SiliconFlow's advertised decoding speed does not establish usable tool support. The refreshed
endpoint catalogue omitted `tools` and `tool_choice`; the actual request was rejected at the
tool-compatibility filter. Novita advertised both and returned native calls. All routes were
explicitly pinned with fallbacks disabled; no tools or account restrictions were removed to
obtain a response. [OpenRouter endpoint metadata](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4.1-flash/endpoints),
[routing semantics](https://openrouter.ai/docs/guides/routing/provider-selection).

The July V4 model costs less on its selected route and already satisfied this small suite.
V4.1 was faster on its successful runs, but this pilot provides no necessary capability gain
that would require adopting it for these tasks.

## Errors and evidence

Mercury Classic completed the sorting/restoration checkpoints, then tapped a non-clickable page
title while trying to leave. AUA marked that unchanged frame as stale. Mercury recovered with
Back and fresh home evidence followed, but the strict verifier rejects that intermediate
uncertain frame after restoration. Its shared error text says the fingerprint is missing;
inspection confirmed a fingerprint exists and `stale_risk` caused the rejection. The result
remains **unverified**, with `false_pass:null`; later screenshots do not override the rule.

Hosted Gemma Classic required one schema repair before execution. July DeepSeek Classic received
one `element_not_found` error and recovered. Neither was silently treated as an error-free trace.
Novita Classic r1 encountered HTTP 429 on its second model request after one action. Its reset
and session termination commands succeeded, but there is no post-failure-home PNG, so that
failed run has no visual cleanup proof. Its successful retry remains a separate artifact.

Independent replay reproduced all original outcomes. Reviewers checked **480 original files**
before/after without changes and inspected **49 representative screenshots**, covering price/name
ordering and pairings, async error/recovery and home restoration where evidence existed.
Missing proofs in the provider-interrupted run were retained as missing.

## Cost, settings and reproducibility

Reported task-attempt charges: **$0.025051274**. Protocol/diagnostic charges: **$0.000810164**.
Total unique recorded API charges: **$0.025861438**, approximately **2.6 US cents**. Duplicate
copies of the successful Fireworks diagnostic were counted once by generation ID. These are
API-reported charges, not an independently reconciled invoice, and exclude local computation
and any separate credit-purchase fees/taxes. HTTP failures lacking usage add no reported amount;
that alone is not proof of zero billing.

The native template and reasoning mode stayed provider-specific: DeepSeek low reasoning,
Mercury low reasoning at temperature 0.5, hosted/local Gemma thinking enabled at temperature 0.
DeepSeek thinking ignores temperature. This is a practical controller comparison with disclosed
settings, not an identical-sampling or identical-weight hosting benchmark.

All runs allowed 4,096 output tokens, 32 steps, 100,000 request bytes, 180 seconds of interaction
and 60 seconds per model request. One native call was allowed per response. Hosted context
compression was disabled. Provider context windows were larger than the local 32K cap;
recorded prompt-plus-requested-output sizes are retained to assess that difference.
The largest hosted value was 16,391 tokens, so no recorded request exceeded 32,768 by that measure.

The adapter preserves returned reasoning blocks, records actual requests/provider labels/usage,
checks credentials before AUA starts, and enforces reported-spend stops plus provider price caps.
Missing/invalid cost stops further calls; an in-flight request can exceed a reported-spend stop.
No individual run reached its $0.10 stop, and the campaign stayed below $1.00.

`hosted-v1` removes unnecessary host/device/session identifiers, artifact paths and logs from
model-facing observations while preserving fresh element IDs and task semantics. It is a
fictional-fixture projection, not a general personal-data anonymizer. Original evidence remains
complete and private. Native model reasoning is not rewritten by this projection.

Load `OPEN_ROUTER_API_KEY` into the process environment without logging it, then use the
[candidate manifest](openrouter-comparison.json), for example:

```bash
python -m experiments.aua_controller.run_live \
  --base-url https://openrouter.ai/api/v1 --backend openrouter \
  --api-key-env OPEN_ROUTER_API_KEY \
  --manifest experiments/aua_controller/openrouter-comparison.json \
  --model or-deepseek-v4-flash-0731-low-deepinfra \
  --profile compact-v1 --observation-profile hosted-v1 \
  --max-tokens 4096 --time-limit 180 --request-timeout 60 --cost-limit-usd 0.10 \
  --scenario classic-sort --output runs/aua-controller/new-deepseek-classic-run
```

Use a new empty output directory and the public fixture. The CLI does not load `.env` itself;
the private pilot launcher loaded only the required credential into the child environment.
Protocol checks must pass with the same candidate settings before device testing.

Focused verification: **258 tests passed; Ruff passed**. The local MLX control server was stopped
and endpoint closure verified at **2026-09-12 12:36:15 UTC**. No training or Runpod infrastructure
was used. Private requests, costs, protocol failures, replay sidecars and screenshots remain in
the main checkout's ignored `runs/aua-controller/20260912-openrouter-research/` directory.

Next: evaluate repeated, unfamiliar flows with July V4 Flash and retain the same independent
completion boundary. Choose training only if measured capability gaps justify it; training
remains Runpod-only.
