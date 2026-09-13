# Hosted AUA controller candidates: 2026-09-12

This is a dated research snapshot and proposed pilot configuration, **not hosted model results**.
Prices and capabilities were checked against official vendor documentation and OpenRouter's public
catalogue/endpoint metadata. Actual protocol, AUA outcomes, latency and cost require separate runs.
The existing [local Gemma E4B measurements](LOCAL_STATUS.md) provide the comparison baseline.
The subsequent [measured pilot](OPENROUTER_STATUS.md) records completed trials and provider
availability; the original planned routes below are preserved as research history.

The aim is a low-cost, fast **verified test completion**. Model price, decode tokens per second,
first-token latency and complete tool-decision latency are different measurements. Inception's
advertised 1,107 tokens/second is a vendor throughput claim, not an AUA decision-time measurement;
OpenRouter's latency/throughput aggregates also mix prompts, output lengths, loads and reasoning
settings. A completed UI test additionally includes transport, prefill, AUA actions, recovery,
proof and cleanup. [Mercury launch evidence](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5).

## Planned first pilot

All prices are USD per **1 million tokens**, input / output, for the named provider route.
These configurations compare practical serving choices; their reasoning/sampling settings differ.

| OpenRouter model | Provider route | Proposed configuration | Input / output |
| --- | --- | --- | ---: |
| `deepseek/deepseek-v4.1-flash` | `deepseek` | Low reasoning; do not claim deterministic temperature control | $0.15 / $0.60 off-peak |
| `inception/mercury-2.5` | `inception` | Low reasoning, temperature 0.5 | $0.04 / $0.15 launch promotion |
| `google/gemma-4-26b-a4b-it` | `dekallm/bf16` | Thinking enabled, temperature 0.0 | $0.06 / $0.33 |

Provider-specific sources: [DeepSeek endpoint](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4.1-flash/endpoints),
[Inception endpoint](https://openrouter.ai/api/v1/models/inception/mercury-2.5/endpoints),
[DekaLLM endpoint](https://openrouter.ai/api/v1/models/google/gemma-4-26b-a4b-it/endpoints).
The advertised context limits for these routes are 1,048,576, 260,000 and 262,144 tokens respectively;
the AUA pilot's smaller fixed budget remains a separate constraint.

DeepSeek V4.1 Flash was released September 10. Direct API names `deepseek-v4-flash` and
`deepseek-v4-flash-vision-exp` now alias V4.1; OpenRouter still has separately named older V4
checkpoints. Use the exact hosted model and provider, not an ambiguous "Flash" label.
[DeepSeek release history](https://api-docs.deepseek.com/updates/).

On the official DeepSeek route, peak input/output rates are $0.30/$1.20; cache-read rates are
$0.003 off-peak and $0.006 peak. Peak periods are weekdays 01:00–04:00 and 06:00–10:00 UTC.
Other hosts set their own prices. Thinking mode ignores temperature; low/high/max are supported.
[Direct pricing](https://api-docs.deepseek.com/quick_start/pricing/),
[thinking controls](https://api-docs.deepseek.com/guides/thinking_mode/).

Mercury 2.5 was released September 8. Its regular input/output rates are $0.20/$0.75; the quoted
$0.04/$0.15 is an 80% launch promotion, not a permanent assumption. Its API accepts temperature
0.5–1.0 and resets out-of-range values, including zero, to 1.0 with a warning. Therefore this lane
explicitly uses 0.5 and must retain any returned warnings.
[Vendor pricing](https://www.inceptionlabs.ai/models),
[sampling contract](https://docs.inceptionlabs.ai/api-reference/chat/create-a-chat-completion).

The hosted Gemma is **26B-A4B, not the E4B checkpoint tested locally**. It is a larger MoE model;
BF16 identifies the selected provider's advertised precision. This can compare practical local and
hosted choices, but cannot isolate hosting speed for identical weights. Hosted identity is provider
metadata, not the independent local checkpoint-hash verification described in [LOCAL_STATUS.md](LOCAL_STATUS.md).
[Google model card](https://ai.google.dev/gemma/docs/core/model_card_4).

## Additional candidates, not first-pilot results

| Model | Provider | Input / output | Caveat |
| --- | --- | ---: | --- |
| `qwen/qwen3.8-flash` | `alibaba` | $0.15 / $0.47 | Multimodal candidate; test semantic text first. |
| `qwen/qwen3.7-flash` | `alibaba` | $0.03 / $0.13 | Applies below 32,000 prompt tokens **per request**; higher tiers cost more. |
| `deepseek/deepseek-v4-flash-0731` | `deepinfra/fp8` | $0.06 / $0.18 | July 31 V4 revision, distinct from V4.1 and the April preview. |

Sources: [Qwen 3.8 endpoints](https://openrouter.ai/api/v1/models/qwen/qwen3.8-flash/endpoints),
[Qwen 3.7 endpoints and tiers](https://openrouter.ai/api/v1/models/qwen/qwen3.7-flash/endpoints),
[July Flash endpoints](https://openrouter.ai/api/v1/models/deepseek/deepseek-v4-flash-0731/endpoints).
An endpoint being listed with tool support does not establish successful AUA use or current capacity.

## Hypothetical token-cost illustration

The three local tasks used 123,101 prompt and 4,078 output tokens in total: averages of
**41,033.667 input and 1,359.333 output tokens per test**, summed across all requests and repeated
history. Output includes reasoning; its separate token count was unavailable. [Measured inputs](LOCAL_STATUS.md).

Holding those totals artificially constant and assuming **no cache hits**, the arithmetic is
`(input_tokens * input_rate + output_tokens * output_rate) / 1,000,000`:

| Planned route | Hypothetical tokens-only cost per test |
| --- | ---: |
| V4.1 Flash / DeepSeek off-peak | $0.006971 |
| Mercury 2.5 / Inception launch promotion | $0.001845 |
| Gemma 26B-A4B / DekaLLM | $0.002911 |

These are **not measured hosted costs or predictions of equal task success**. Tokenization, tool
choices, reasoning length, retries, cache behavior and failed runs will differ. Fees and taxes are
excluded; OpenRouter lists a 5.5% pay-as-you-go platform fee. Expired promotions and peak pricing
change the calculation. [OpenRouter pricing](https://openrouter.ai/pricing).

## Evidence needed before selecting a controller

Pin the provider and settings, disable silent fallback for comparison runs, and check support for
every requested parameter. Preserve native tool-call history and returned reasoning fields; retain
structured `reasoning_details` unchanged when supplied. Validate arguments before execution and
reject malformed, multiple or truncated calls. Pass the two-turn protocol probe before device work.
[Provider routing](https://openrouter.ai/docs/guides/routing/provider-selection),
[reasoning history](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

Keep goals, `compact-v1` tools, starting states, observation content, limits and the independent
finish/cleanup oracle comparable. Report full model-request time and end-to-end time, actual usage
and billed cost, failed attempts, recovery and **cost per independently verified completion**.
Missing evidence remains unknown. Three prepared local tasks establish a narrow working example;
they neither prove unseen-app reliability nor establish that future training is unnecessary.
Broader failures should determine whether to train, using the trajectory and split requirements in
[DIAGNOSIS.md](DIAGNOSIS.md). All training remains Runpod-only.

The dated catalogue and endpoint snapshots are retained privately under the main checkout's ignored
`runs/aua-controller/20260912-openrouter-research/`. They contain research metadata, not pilot scores.
