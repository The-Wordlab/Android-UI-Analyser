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

Three pluggable provider kinds, each behind an interface with an ordered **fallback
chain** (try provider A; on failure/timeout, try B, then C):
- **OCR providers:** Apple Vision, RapidOCR, PaddleOCR, Tesseract, EasyOCR.
- **Detection providers:** OmniParser-v2 (local), YOLO (local, user weights).
- **Grounding/analysis providers:** local VLM (vLLM/Ollama/HF) and commercial
  (OpenAI / Anthropic / Gemini) selected by config, with API keys from env vars.

Selection precedence: **CLI flags > env vars > project config > user config >
defaults.** Secrets are referenced by env-var name, never stored in plaintext config.
