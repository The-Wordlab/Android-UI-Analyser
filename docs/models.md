# Models and keys

Every hosted model call AUA makes goes through one module, `src/android_ui_analyser/llm_route.py`.
A call names its model the way OpenRouter does (`openai/gpt-6-luna`), and the route is chosen from
the keys that exist:

| Keys set | An `openai/*` model | Any other model |
|---|---|---|
| `OPENAI_API_KEY` | straight to api.openai.com | through OpenRouter (needs its key) |
| `OPEN_ROUTER_API_KEY` only | through OpenRouter | through OpenRouter |

Going direct is the cheaper route when an organisation has its own OpenAI pricing. OpenAI does not
report what a request cost, so `llm_route` prices the answer itself (`OPENAI_PRICES`) and writes it
into `usage.cost`, exactly where OpenRouter puts it; spend stops and run totals read one field. An
OpenAI model without a known price stays on OpenRouter rather than report $0.

One kind of request stays on OpenRouter even with an OpenAI key, because OpenAI's chat completions
refuse it: function tools combined with reasoning, which is what the judge sends. A tool schema with
`oneOf`/`anyOf` at its top level is flattened for OpenAI automatically.

A bare model id (`gpt-5`) means OpenAI itself and needs `OPENAI_API_KEY`.

## Switching a model

Each role is one setting. Change it in the project's `.android-ui-analyser.yaml`, an `AUA_*`
environment variable, or on the command line; no code changes.

| Role | Setting | Default | Notes |
|---|---|---|---|
| Controller (drives the app) | `controller.model` | `or-gpt6-luna-open` | a profile id from `experiments/aua_controller/openrouter-comparison.json`; reasoning off, so it can go direct |
| Judge (decides the verdict) | `controller.judge_model` | `or-gpt6-luna-low-open` | keeps reasoning, which it needs; stays on OpenRouter |
| Judge fallbacks | `controller.judge_fallbacks` | DeepSeek V4.1, then Gemma | other vendors, so an OpenAI outage does not stop judging |
| Icon names | `models.hosted_vision.model` | `openai/gpt-6-luna` | reasoning off |
| Grounding / screen questions | `models.openai.model` | `openai/gpt-6-luna` | |
| Jev navigator and judge | `TYPESAFE_API_KEY`, else the OpenRouter key | `typesafe/jev-*` | Jev's own client; the same "own key first, OpenRouter second" rule |

A controller profile carries its request settings (reasoning, OpenRouter routing and price cap). Add
a new one to the manifest to try a model; the harness takes it with `--model <id>`.

## Keys

Keys are read from the environment; harnesses load them from an ignored `.env` with
`aua config exec --env-file .env --require OPEN_ROUTER_API_KEY --optional OPENAI_API_KEY -- …`.
`--optional` passes a key when the file has it and never asks for it.

To save a key without it ever appearing in a terminal or chat, open the masked dialog:

```bash
aua config secret OPENAI_API_KEY --env-file /absolute/path/.env
```

Nothing else changes: the next run finds the key and takes the direct route.
