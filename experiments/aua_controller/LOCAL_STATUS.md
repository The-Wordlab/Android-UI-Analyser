# Local Mac validation: 2026-09-12

**Gemma E4B completed all three prepared AUA tasks using inference on the local Mac.**
No weights changed, no training ran and no cloud resources were created for these runs.
Inference and testing stay local; only training uses Runpod. The historical cloud results
remain separate in [STATUS.md](STATUS.md).

## Configuration

- Apple M5 Max, 128 GB unified memory, macOS 26.6.2, arm64.
- MLX 0.32.0 and MLX-LM 0.31.3, concurrency one, original BF16 weights.
- `google/gemma-4-E4B-it`, revision `ee0ef6023621cff504d758262d4e04895a5af4a2`.
- Nine declared snapshot artifacts verified against the saved Hub identity:
  16,024,823,729 bytes. No quantization or checkpoint conversion.
- Native Gemma tool protocol, reasoning enabled on every actual request,
  `compact-v1` controller profile, temperature zero.
- Fixed limits for all tasks: 32,768 total context tokens, 4,096 output tokens,
  32 steps, 100,000 request bytes, 180 seconds of interaction and 60 seconds per request.
- Loopback endpoint `127.0.0.1:18000`, fixed `default_model` request alias,
  pinned snapshot verified offline, no model switching or cloud fallback.

The native two-turn protocol probe passed before Android testing. An actual request with an
unknown model alias returned HTTP 422 before inference. Protocol probes, setup checks and
failed server startups are excluded from the three model-run results.

## Results

| Task | Verified result | Native calls | Model HTTP time | AUA setup | Runner total |
| --- | --- | ---: | ---: | ---: | ---: |
| Classic price sorting, name-order restoration and home | PASS | 5 | 49.72s | 32.59s | 89.16s |
| Compose price sorting, name-order restoration and home | PASS | 6 | 55.90s | 47.07s | 115.15s |
| Error, retry, ready state and home | PASS | 4 | 27.09s | 32.26s | 67.75s |

Model HTTP time sums request wall time, including prefill and generation. Runner total also
includes setup, AUA actions, evidence handling and cleanup; it excludes model download/loading
and the small outer-process startup overhead. These are elapsed measurements, not GPU compute
time or an estimate of electricity cost.

Compose requested completion once before returning home. AUA refused; the model then pressed
Back and completed correctly. There were no schema repairs or output truncations. All 15
responses contained a native tool call and reasoning text. Across the three runs:

- Model HTTP time: 132.72 seconds; median request 8.22 seconds, p95 12.90 seconds.
- Prompt tokens: 123,101, including repeatedly submitted history.
- Output tokens: 4,078, including reasoning. Separate reasoning-token count is **unknown**:
  this MLX server does not report it, and no estimate was substituted.
- Largest individual prompt: 19,560 tokens, within the enforced context budget.

An independent replay of every run passed the strict `aua_contract_replay_v2` verifier,
including cleanup, without changing original result hashes. Nine checkpoint screenshots
were visually inspected: both sorting orders and their product/price pairings, temporary
error and successful recovery, and restored home screens. Capture configuration was restored.

## Local runtime corrections

The first local startup rejected 54 redundant K/V tensors in shared-attention layers 24–41.
The installed MLX version already implements shared-KV execution but lacks the corresponding
[upstream loading fix](https://github.com/ml-explore/mlx-lm/commit/df1d3f3c9a7aae402dcbb8f41d4c36bcc13a50ae).
The local launcher applies that narrow fix only to the pinned model and exact unused tensor
names, checking the expected architecture. Active tensors still load strictly; disk weights
remain unchanged. This was runtime compatibility, not evidence of failed training.

The next attempt exposed a launcher bug: loading the model on the main thread bypassed MLX's
generation-worker lifecycle and the first request failed with a thread-local GPU stream error.
Loading now happens inside the generation worker, following the
[upstream lifecycle](https://github.com/ml-explore/mlx-lm/pull/1090). HTTP starts only after
loading and tokenizer validation succeed; startup errors propagate to the runtime receipt.
Both failed attempts remain archived separately from the successful run.

Focused verification after the fixes: **216 tests passed, 0 failed; Ruff passed**.
The [local launcher](serve_local.py) also rejects malformed/multiple native calls and prevents
truncated output from becoming an action. [README.md](README.md) contains reproduction commands.

## Scope and shutdown

This is one repetition of each of three public fixture tasks. The model chose the actions;
the harness supplied authored goals, progress, assertions, limits and independent completion
checks. It is oracle-assisted control, not evidence that the model can independently design
tests or reliably operate unseen applications. No image bytes were sent to the model.

The measured BF16 Mac run establishes local feasibility. Quantized performance, broader
reliability and other candidates' local behavior remain unmeasured. The next useful evaluation
is repeated local testing on new flows, then local latency/quantization comparisons; training
should follow identified capability gaps and use independently verified AUA trajectories.

The local server was stopped and port closure verified at **2026-09-12 12:03:57 UTC**
(14:03:57 Europe/Madrid). The downloaded model cache remains available for future local runs.
No additional Runpod spend was incurred for this local pilot.

Private runtime receipts, immutable results, replay checks, timings and screenshots are in
`runs/aua-controller/20260912-local/` in the main checkout, outside version control.
The manifest records this as a separate local lane: 30 historical cloud model runs plus
3 local model runs; those heterogeneous results must not be pooled into a success rate.
