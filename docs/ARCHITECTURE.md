# Architecture & Design Decisions

> The reasoning behind `android-ui-analyser`. This captures the design discussion
> that produced the PRD. Read `RESEARCH.md` first for the landscape facts.

## 1. Thesis: read the tree, not the pixels

For Android UI testing, the device already hands you every element and its exact
bounding box through the accessibility / view hierarchy. That is faster **and** more
accurate than any vision model for "what's on screen and where." So:

> **Hierarchy-first, vision-second.** Use the UI tree as the primary perception
> source. Use vision only for the screens where the tree is empty or useless.

### Why not just use a big multimodal model?
The slow, expensive part of "screenshot → VLM → coordinates" is **not** the
screenshot — it is the model round-trip (2–6 s) and the per-image token cost on
every step. A persistent tree reader answers the same question in tens of
milliseconds with pixel-exact boxes and no hallucinated coordinates.

---

## 2. Perception: fast path vs. fallback

### Fast path — the UI tree (~30–150 ms)
A persistent reader returns, per element: `bounds`, `resource-id`, `text`,
`content-desc`, `class`, and `clickable`/`enabled`/`focused`.
- **openatx `uiautomator2`** (Python lib): manages a JSON-RPC agent on the device;
  `d.dump_hierarchy()` is tens of ms. No app code to write. **Default choice.**
- **AccessibilityService** (droidrun Portal-style): lowest latency, best Compose
  coverage, but you ship + enable an APK. Optional advanced backend.

### Where the tree goes blind → vision is mandatory
- Jetpack Compose without `testTag`/semantics, Flutter (single view), partial
  WebViews, games/canvas (nothing), and `FLAG_SECURE` (may block screenshots).

### Vision fallback (~150–600 ms local)
- **Detection (no LLM):** OmniParser v2 detection-only, or a custom YOLO UI detector.
- **OCR (no LLM):** Apple Vision (macOS, fastest), RapidOCR/PaddleOCR/Tesseract/EasyOCR.
- **Optional grounding VLM:** Holo1.5-7B (Apache-2.0) etc. for "where is the element
  that does X" on hard screens.
All vision results are merged back into the **same Set-of-Marks JSON** with synthetic
IDs, so downstream code does not care where an element came from.

---

## 3. Decision: do NOT fork an existing MCP server

The core of these tools is not complicated. Stripped down, an MCP/CLI screen tool is:

1. **The interface layer** (MCP tool defs or CLI subcommands) — boilerplate, ~20 lines.
2. **Get the UI tree** — the only genuinely fiddly part, and it is already a library.
3. **Parse XML → JSON with boxes** — trivial; `bounds="[x1,y1][x2,y2]"` is right there.
4. **Quality gate / Set-of-Marks IDs / vision fallback** — *our* logic, not in the
   forks anyway.

The one part worth not hand-rolling is the **device plumbing**: installing and keeping
a fast agent alive on the device, port forwarding, screenshot capture, input injection
that behaves across Android versions and OEMs. **That is solved by `uiautomator2` — a
pip-installable library, not a fork.** Two of the candidate MCP servers literally wrap it.

> **The real choice is not "fork a whole server" vs "build from scratch." It is:
> depend on `uiautomator2` for the hard plumbing, and write a thin engine + interface
> (~150 lines) that holds our special sauce.**

### When forking *would* be worth it
- You want **iOS** too (mobile-mcp's real value is cross-platform XCUITest + UiAutomator).
- You want their accumulated **OEM edge-case** fixes.

For an Android-only, speed-first tool with a custom output shape, a fork mostly hands
you 70+ tools and formatting opinions you would fight against — so we build instead.

---

## 4. Decision: CLI-first, engine as a library, MCP optional

**MCP vs CLI is just the interface; the engine is identical.** So the engine is a
plain, interface-agnostic library, and we expose it as a CLI (primary) and an MCP
server (optional thin wrapper) — ~10 lines each. No lock-in.

### Why a CLI works great for Claude Code
Claude Code has a Bash tool. It runs `aua analyze`, reads JSON off stdout, and acts.
No protocol, no server registration. Bonus: you can run it yourself, pipe it into
`jq`, use it in CI, and debug by hand — none of which MCP gives for free.

### The one tradeoff: warm state
A CLI that starts fresh each call pays Python startup + the uiautomator2 reconnect
handshake (~300–500 ms overhead/call). Two mitigations:
- **Eat it.** We are fixing Maestro's *multi-second* loop and the VLM round-trip, not
  shaving 300 ms. For UI testing (not 60 fps), a plain CLI is usually fast enough.
- **Thin client + daemon.** The CLI's first call starts a tiny background process that
  holds the warm connection; later calls are a localhost roundtrip. Best of both,
  slightly more plumbing. (The on-device agent stays warm regardless — that lives on
  the phone, not the CLI.)

### Where MCP genuinely wins (build it as a wrapper, enable later)
- **Typed tool discovery:** the agent sees `analyze_screen`, `tap(id)` as first-class
  tools and calls them on its own. With a CLI you teach it once in `CLAUDE.md`.
- **Portability:** for hosts without a shell (Claude desktop app, Cursor, etc.), MCP
  is the contract.

---

## 5. Set-of-Marks: the agent acts on IDs, not pixels

`analyze` assigns every element a stable integer **ID** and returns compact JSON.
Action commands take an ID (`tap 4`, `input 2 "text"`, `swipe up`). Benefits:
- No coordinate hallucination — the ID maps to a known box; the tool computes the
  center.
- Much smaller token footprint than passing images every step.
- Works identically whether the element came from the hierarchy, detection, or OCR.

Optionally emit an **annotated screenshot** (numbered boxes overlaid) for debugging or
for a human in the loop — but it is not required for the agent to operate.

---

## 6. Latency budget (targets the implementation must respect)

| Path | Target |
|---|---|
| Hierarchy `analyze` | < 150 ms/call (warm) |
| Local vision fallback (detect + OCR) | < 600 ms/call |
| Hosted 7B grounding VLM | 0.5–2 s (opt-in only) |
| Commercial multimodal API | 2–6 s (opt-in only, never default) |

---

## 7. Element JSON shape (canonical)

```json
{
  "screen": { "width": 1080, "height": 2400, "package": "com.example.app",
              "activity": ".MainActivity", "source": "hierarchy" },
  "elements": [
    {
      "id": 0,
      "type": "Button",
      "text": "Sign in",
      "resource_id": "com.example:id/sign_in",
      "content_desc": null,
      "bounds": [120, 1500, 960, 1610],
      "center": [540, 1555],
      "clickable": true,
      "enabled": true,
      "focused": false,
      "source": "hierarchy"
    }
  ]
}
```

- `source` ∈ `hierarchy` | `detection` | `ocr` | `grounding` so callers can reason
  about confidence.
- A `--compact` mode drops nulls and verbose fields to save tokens.

---

## 8. Provider + fallback model (the configurable brain)

Five pluggable provider kinds live behind interfaces. Perception and planning use ordered
**fallback chains** (try provider A; on failure/timeout, try B, then C):
- **OCR providers:** Apple Vision, RapidOCR, PaddleOCR, Tesseract, EasyOCR.
- **Detection providers:** OmniParser-v2 (local), YOLO (local, user weights).
- **Grounding/analysis providers:** local VLM (vLLM/Ollama/HF) and commercial
  (OpenAI / Anthropic / Gemini) selected by config, with API keys from env vars.
- **Planner providers:** an opt-in goal navigator used only by explicit assist/navigate paths.
- **Policy providers:** an optional opaque-ID selector behind a deterministic candidate guard.

Selection precedence: **CLI flags > env vars > project config > user config >
defaults.** Secrets are referenced by env-var name, never stored in plaintext config.

## 9. Optional policy has one explicit bounded execution lane

FunctionGemma is not part of the perception ladder and does not receive the Android hierarchy.
During an active verification phase, deterministic AUA code may compile up to four complete
current-frame tap calls with stable selectors. It removes unsafe, destructive, unauthorized,
redundant, ambiguous, or stale candidates, keeps exact calls in a trusted map, and exposes only
privacy-screened metadata plus dense opaque IDs to the selector. Session, phase, package, and
observation provenance are revalidated after inference.

The policy is off by default. Shadow emits audit metadata without the chosen call. The advisory
interface can emit a separate `policy_suggestion`, but the deterministic `recommended_call` remains
unchanged and that suggestion is never executed manually. Explicit `session_autopilot` is the only
execution lane: for an authenticated advisory-capable adapter, the warm daemon takes the selected
opaque ID, revalidates its trusted call against the current session/phase/frame, executes it through
the ordinary Engine action, and consumes the folded observation before another decision. The loop
is tap-only and bounded by steps/time; it stops on stale/unknown outcomes, no progress, repetition,
unsupported work, or handoff. Bundled v3's authenticated manifest caps rollout at shadow, so it
cannot use this lane and advisory returns `unsupported_mode` before inference. Zero candidates return a
non-executing structured handoff in advisory mode. A model may select reserved handoff ID `-1` only
when an authenticated adapter manifest binds that protocol; it never becomes an AUA call. One action
bypasses MLX; the frozen v3 adapter accepts exactly four and withholds advice for two/three.

AUA packages only the modified ~29 MB v10 LoRA adapter. The pinned ~543 MiB MLX base is external,
must be supplied as an absolute local path, and is never downloaded automatically. The adapter's
manifest pins each required base file, the candidate cardinalities and handoff protocol it was
trained for, training/evaluation provenance, and its separate Gemma license materials.

The bundled manifest authenticates advisory rollout, so the execution lane is reachable without an
externally supplied adapter. What keeps it inert is configuration, not a ceiling inside the artifact:
`policy.enabled` is `false` in shipped defaults, so no model is resolved, loaded, or given memory
until an operator turns the policy on. Shadow mode exists for developing and debugging the policy
locally and has no end-user purpose.

An independently authored probe — its scenarios derived from the CLI surface rather than from the
training generators — scores the bundled adapter at 0.600 on 150 jobs at its best checkpoint (0.471
mean over 16), with refusal at 18/38 and no invalid outputs. On a device it completed 5/5 navigations
and 2/2 refusals with zero wrong taps. It is nonetheless **not promoted**: one seed, no live gate,
and refusal that swings between 0 and 18 across checkpoints. Safety continues to rest on the
deterministic guard, which removes unsafe, unauthorized, destructive, stale, ambiguous, and redundant
candidates before inference and revalidates before execution. Earlier v3 illustrates why the
independent probe matters: 99.8535% on synthetic held-out data, then 60/96 (62.5%) on a
production-serializer matrix, with accuracy swinging 37.5 points across target IDs.

A failure-driven v4 continuation then reached 2,767/2,768 validation (99.9639%, including 719/720
production-shaped cases), 96/96 on the untouched production smoke, perfect held-out production
choices for cardinalities 2/3/4 (64/64, 144/144, 512/512), and 4/4 clean closed loops. It still
failed its independent combined test at 2,764/2,768 (99.8555%): critical accuracy was 99.6875% with
four unauthorized `sequence_recover_unknown` decisions, all
ending early with `session_finish` instead of using `analyze_screen`; parsing was 100% and redundant
selections were zero. V4 remains ignored and unbundled. The next iteration needs recovery-focused
data and evaluation that stay independent from training. This is why the guard, non-executing
modes, and exact-cardinality boundary are architectural constraints rather than documentation
caveats.

---

## 10. Engine layout: one class, fourteen files

`Engine` is still a single class — the CLI, MCP server, daemon and dashboard all call
`engine.<method>(...)` and every test that patches `Engine.<method>` or `engine.<method>` keeps
working — but its methods no longer live in one 21,000-line file. `engine.py` holds what is
genuinely core; every other method is a module-level function in the domain module its name
suggests, with the `Engine` instance as its first parameter (still called `self`), and the class
body binds it back:

```python
# engine_flows.py
def flow_run(self: Engine, name: str, *, params: dict[str, str] | None = None) -> FlowRunResult:
    ...

# engine.py
class Engine:
    ...
    flow_run = engine_flows.flow_run
    _flow_ref_key = staticmethod(engine_flows._flow_ref_key)
```

Binding a plain function in the class body makes it an ordinary method: `self` binds, `mypy`
type-checks the body against `Engine`, `inspect.getsource`, `__doc__` and `__module__` work, and a
`monkeypatch.setattr(Engine, "flow_run", ...)` replaces it like any other attribute. The split was
mechanical and verbatim — every function's AST and comments are identical to the original — so the
history of a method continues in its new file.

| Module | Owns |
|---|---|
| `engine.py` | construction, properties and context managers, device connect and lease lifecycle, the read-only device shell, the device-change ledger and teardown, on-device helper handover, the last-analyze id cache, foreground package/activity reads, caller-turn telemetry |
| `engine_analyze.py` | perceiving the screen: hierarchy, OCR and vision capture, the analyze pipeline and semantic query, screenshot/inspect/annotate, provider status |
| `engine_observation.py` | the post-action `observation`: the shared `_observe` pipeline, loading/readiness and settle waits, arrival and stale-risk verdicts, before/after change summary, crash and app-log evidence |
| `engine_actions.py` | acting by id: target and selector resolution, tap/long-press/double-tap, text input, clear/erase, mic injection, swipe/scroll/key, keyboard, clipboard paste/copy, a11y actions |
| `engine_waits.py` | has/expect, wait/wait_stable/wait_changed/await_predicate, the locale bridge, the caller-sized wait budget, hierarchy change detection, background jobs |
| `engine_navigation.py` | goto over the learned map, navigate and reach, the drive lanes, back_until, open_link, map_find |
| `engine_flows.py` | saved flows: flow_run/save/list/delete, the step executor and its on-device offload, nested-flow preflight, demo recording, suite_run |
| `engine_sessions.py` | session_start → session_finish: goal planning, phase progress, recommended-call ranking, candidate flows, session review |
| `engine_virtual_targets.py` | platform-neutral virtual-target list/create/delete/start/provision/status/stop/reclaim orchestration and Android emulator compatibility aliases |
| `engine_policy.py` | the optional local policy: model_control, policy tap-candidate and selection helpers, the session policy side channel, session_autopilot |
| `engine_memory.py` | learning into per-app memory and reading it back: recorded screens, actions and timings, the runtime flag context, learned control costs, `memory update`, orient, explore mine/plan |
| `engine_apps.py` | the app under test: launch/stop/clear/install and the launch observation, feature flags, private databases, logcat and per-app log preferences, process bookkeeping |
| `engine_environment.py` | the conditions the app runs under: network, airplane and profiles, the mock proxy, clock, location, orientation, clipboard, media, developer options |
| `engine_capture.py` | pixel evidence over time: the rolling capture buffer and its views, the capture sidecar, the capture hint, screen recording |
| `engine_support.py` | constants, record types and pure helpers that more than one of the above reads |

Rules that keep this shape honest:

- A domain module imports `engine_support` and the rest of the package, never `engine` at
  runtime (only under `TYPE_CHECKING`, for the `self: Engine` annotation) and never a sibling
  `engine_*` module — both are import cycles. A helper two modules need moves to `engine_support`.
- New method bodies go in the domain module; `engine.py` grows only by the one-line binding.
  `tests/test_no_shadowed_methods.py` fails if the same name is bound twice.
- The architecture guards treat `engine.py` and `engine_*.py` as one unit: the platform-boundary
  test scans all of them as generic layers, and the source-scanning guards (one wait clamp, every
  process-replacing call tells the adapter, the `record_ids` exemption is documented) read them
  together.
- A method reads module constants from *its own* module's globals. A test that wants to shorten a
  timeout patches the module that holds the method — `sys.modules[Engine.flags_set.__module__]` —
  not `android_ui_analyser.engine`.
- Target discovery and connection always go through the selected `PlatformAdapter`; the former
  Android-only `connect`/`list_devices` monkeypatch seam has been removed.
