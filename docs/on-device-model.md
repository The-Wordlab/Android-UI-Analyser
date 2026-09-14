# On-device hosted-model runner

`aua helper model-run` is an experimental comparison lane for a rootable Android target. After the
normal AUA session selects, leases and prepares the target, the host performs one helper handoff. The
helper APK then owns every hierarchy observation, DeepSeek V4.1 Flash request, UI action, settle and
contract check until the run stops. There is no host or adb round trip between model steps.

This is not an embedded-key design. An Android APK cannot keep a compiled bearer token secret. The
command reads `OPEN_ROUTER_API_KEY` from its process environment, sends it through the existing
adb-forwarded loopback channel for this run only, and the helper removes it from the request object
immediately. The key is never written to Android storage, logs, the result, the journal or the APK.
Use a scoped, revocable test key; a production deployment should exchange a device identity for a
short-lived token at a broker instead.

The completion boundary is deterministic. A model calling `finish` does not pass the run unless every
check supplied by the caller passes on the helper's own observations. Supported checks are:

| Kind | Meaning |
| --- | --- |
| `first_visible` | The selector matched the first helper observation. |
| `ever_visible` | The selector matched at least one observation. |
| `final_visible` | The selector matches when the run ends. |
| `final_absent` | The selector does not match when the run ends. |
| `never_visible` | The selector matched no observation. |

Selectors are `rid`, `text`, `desc`, or `package`. Resource ids accept either the full Android id or
its tail. Text and descriptions use case-insensitive contains matching.

Example `checks.json`:

```json
[
  {"id":"auth-offered","kind":"first_visible","selector":"rid","value":"containerLanding"},
  {"id":"home-reached","kind":"ever_visible","selector":"rid","value":"homeTabBROWSE"},
  {"id":"home-final","kind":"final_visible","selector":"rid","value":"homeTabBROWSE"},
  {"id":"auth-gone","kind":"final_absent","selector":"rid","value":"containerLanding"},
  {"id":"no-anr","kind":"never_visible","selector":"text","value":"isn't responding"}
]
```

Run it without shell-sourcing a secrets file:

```bash
aua config exec --env-file /absolute/project/.env.local --require OPEN_ROUTER_API_KEY -- \
  aua --format json helper model-run \
  "Continue with limited access, reach guest Home, open one primary destination, and return Home" \
  --checks /absolute/path/checks.json --max-steps 16 --time-limit 180 --cost-limit-usd 0.05
```

The result reports wall time, provider-reported spend, prompt/completion/reasoning tokens, per-request
model latency, per-action latency, actions, checks and stop reason. Spend is a post-response boundary:
one in-flight request may exceed the configured limit. A missing or invalid `usage.cost` stops the run
instead of presenting an incomplete cost as zero.

The runner is deliberately narrower than the full host harness. It can act on accessibility-backed UI
and verify hierarchy predicates. App installation, target selection, app-private data, network shaping,
proxying, logcat, screen recording, and screenshot/vision judgement remain host capabilities. This
makes it suitable for a pre-provisioned UI scenario, not a replacement for the whole AUA lifecycle.
