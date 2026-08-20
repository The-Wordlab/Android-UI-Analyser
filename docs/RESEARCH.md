# Discovery: Fast AI Screen Understanding for Android UI Testing

> Landscape research underpinning the `android-ui-analyser` project.
> Compiled June 2026. Figures are representative and drawn from vendor docs,
> model cards, and 2025–2026 sources. Re-verify before committing to anything.

## 0. Goal

Build a **fast** "screen understanding" layer for automated Android UI testing,
driven by an AI agent (Claude Code). The agent needs to know, quickly:
**what UI elements are on screen, and where each one is** (bounding box + text +
type), so it can drive tests by acting on those elements.

The slow, expensive baseline is "take a screenshot → send it to a big multimodal
model → ask where things are." The whole thesis of this project is that for
Android we can mostly avoid vision entirely, because the OS already exposes the
screen as structured data with exact coordinates — and only fall back to vision
on the screens where that data is missing.

---

## 1. Existing solutions for AI-driven Android UI testing

### Maestro / Maestro MCP (mobile-dev-inc, CLI is Apache-2.0)

- **What it is:** A mobile UI testing framework. Maestro MCP wraps the Maestro CLI
  and exposes MCP tools (list/start devices, inspect the live hierarchy, take
  screenshots, tap/input/scroll/back, run flows, cloud runs, "Maestro Viewer").
- **How it perceives the screen:** It reads the **UiAutomator view hierarchy** (XML)
  and takes screenshots. Device I/O goes through its own pure-Kotlin ADB layer
  (`dadb`) — no dependency on the adb server. It is **accessibility/hierarchy-based,
  not vision-based** for element identification. Screenshots are used for assertions
  / image matching.
- **Philosophy:** AI assists authoring, but output is **deterministic YAML flows**.
  The team explicitly stepped back from full natural-language test execution.
- **Why it feels slow (structural, not model-bound):** a re-dump of the hierarchy
  plus a stabilization/retry loop runs on **every** command:
  - settle polls: 200 ms × up to 10
  - animation wait: up to 2 s
  - `waitUntilVisible`: up to 10 s
  - element lookup timeout: 17 s default (7 s for optional elements)
  - net effect: ~1–2 s added latency per action; a happy-path login flow ~12–18 s.
- **Takeaway:** The fix is not a faster model — it is **not re-dumping and not
  blocking on idle every step**. There is also a third-party `slapglif/maestro-mcp`
  (47 tools) wrapping the same CLI.

### mobile-mcp (`@mobilenext/mobile-mcp`)

- Platform-agnostic iOS + Android MCP server. **Accessibility-first with a
  screenshot fallback.** `mobile_list_elements_on_screen` returns elements with
  coordinates + accessibility labels; also tap-at-coords, swipe, type, buttons.
- Under the hood: XCUITest (iOS), UiAutomator/accessibility (Android). Runs over
  stdio or SSE, supports bearer auth.
- **This is the closest existing thing to what we want, and the best reference for
  an accessibility-first server.** Its real value is the cross-platform handling.

### droidrun / mobilerun (MIT, ~7.8k stars)

- LLM-agnostic Android agent. Installs a **Portal APK** running an
  **AccessibilityService** that exposes the live a11y tree as JSON via a
  ContentProvider (`content://com.droidrun.portal/...`) and an optional
  HTTP/WebSocket socket server. Actions run via ADB (no root). Falls back to vision
  when the tree is empty (games / WebViews / custom views).
- Benchmarks: **91.4% on AndroidWorld using GPT-5** (Oct 2025); earlier Gemini 2.5
  Pro result 63.0%. ~$0.075/task in one comparison.
- **The reference design for the AccessibilityService perception route.**

### Mobile-Agent family / AppAgent / AutoDroid

- AppAgent (Tencent) is **vision-based**: screenshot → compute element boxes → draw
  numbered boxes (Set-of-Marks) → multimodal LLM. Mobile-Agent-v2/E add planning.
  AutoDroid combines screenshot + XML dump. These are research agents, not
  speed-optimized servers, but they validate the **Set-of-Marks** pattern.

### Appium (+ vision)

- Appium 2 supports an image-locator strategy via OpenCV (`AppiumBy.image`,
  template/feature matching) and AI element-finder plugins. Native Android screen
  capture can be slow (seconds/screen) unless an MJPEG server is used. Deterministic
  but heavyweight (WebDriver translation layer); slower than Maestro.

### Cloud farms & benchmarks

- Sauce Labs, BrowserStack, AWS Device Farm, Kobiton, LambdaTest layer AI/visual
  features (Applitools, Percy visual diffing). testRigor offers NL test authoring.
- **AndroidWorld** (Google): 116 parameterized tasks across 20 apps on a Pixel 6 /
  API 33 emulator; observation = full-res screenshot + a11y tree; the standard
  online benchmark. **AndroidControl**, **AitW**, **GUI-Odyssey** are offline.
  Current SOTA hovers ~76% (K²-Agent 76.1%, UI-TARS-v2 / Mobile-Agent-v3 73.3%).
  This confirms **a11y-tree + screenshot** is the dominant observation format.

### Existing Android MCP servers (fork candidates)

- `mobile-next/mobile-mcp` — accessibility-first, returns coords + labels.
- `tanbro/uiautomator2-mcp-server` — uiautomator2-based, 70+ tools, XPath filtering.
- `nim444/mcp-android-server-python` — uiautomator2.
- `erichung9060/Android-Mobile-MCP` — `mobile_dump_ui` → hierarchical JSON + centers.
- `ulcica/android-mcp` — Kotlin, enhanced view attributes.
- **None optimizes for raw speed as the #1 goal.** That is our differentiation.

---

## 2. Perception approaches and their speed

| Method | Latency | Returns | Limitations |
|---|---|---|---|
| `adb shell uiautomator dump` | **~3 s**, blocks on `waitForIdle()` | XML: bounds, resource-id, text, content-desc, class, clickable/enabled/focused | "could not get idle state" on animated UIs; per-call process spawn |
| openatx **uiautomator2** `dump_hierarchy()` | server-side **tens of ms** (~31 ms in logs) | same fields | persistent JSON-RPC agent on device; still idle-sensitive on video/animation |
| **AccessibilityService** (droidrun Portal) | event-driven, synchronous (no dump process) | live a11y tree JSON, center coords | requires installing + enabling a service; same canvas/game blind spots |
| Espresso / instrumented UIAutomator | in-process, fast | rich, but in-app | needs test APK; not black-box |

**Shared blind spots (the trigger for vision):**
- **Jetpack Compose** without `Modifier.semantics` / `testTag` → merged or empty nodes.
- **Flutter** → a single `FlutterView` unless semantics are enabled.
- **React Native** → usually OK (RN sets accessibility props).
- **WebViews** → partial DOM exposure.
- **Games / custom Canvas** → nothing.
- **`FLAG_SECURE`** screens → may block screenshots entirely.

Pure vision is needed **only** when the tree is empty/uninformative. Its tradeoff is
latency (model inference) and coordinate-precision risk vs. the exact bounds the
hierarchy gives for free.

---

## 3. Vision / UI-parsing models (Hugging Face focus)

### Detection + caption/OCR (no instruction needed)

- **Microsoft OmniParser v2** (`microsoft/OmniParser-v2.0`): YOLOv8 `icon_detect`
  (boxes + interactable flag) + Florence-2 `icon_caption`. Output is Set-of-Marks
  (numbered boxes + descriptions). **~0.6 s/frame on A100, ~0.8 s on a 4090**; 39.6%
  ScreenSpot-Pro paired with GPT-4o. The **caption stage dominates** — run
  detection + OCR only for speed.
  - **License split: `icon_detect` is AGPL-3.0, `icon_caption` is MIT.** AGPL is a
    real product consideration.
  - **Security: CVE-2025-55322** (unauth RCE in the OmniTool controller, bound to
    0.0.0.0 with no auth), patched in **v2.0.1 (12 Sep 2025)**. Pin ≥ 2.0.1 and never
    expose the parser server unauthenticated.
- **Custom YOLOv8 / YOLO26** fine-tuned on **RICO** (72k Android screens) / **VINS**
  (21 UI classes): fastest, fully controllable, license-clean with a permissive YOLO.
  Real-time on Apple Silicon. Boxes + class only → pair with OCR for text.

### Instruction → coordinate grounding VLMs (ScreenSpot-Pro, higher is better)

| Model | Size | License | ScreenSpot-Pro | Notes |
|---|---|---|---|---|
| Holo2-30B-A3B | 30B MoE | research-only | 66.1% | Qwen3-VL-based, adds Android |
| UI-TARS-1.5-7B | 7B | Apache-2.0 | 61.6% | ScreenSpot-V2 94.2%, AndroidWorld 64.2% |
| **Holo1.5-7B** | 7B | **Apache-2.0** | **57.9%** | **Best commercial pick**; high-res to 4K |
| Ferret-UI Lite (Apple) | 3B | research / unreleased | 53.3% | on-device target; weights not openly published |
| Qwen3-VL-8B | 8B | Apache-2.0 | ~49.9% | native grounding |
| GUI-Actor-7B (Microsoft) | 7B | research | 44.6% | coordinate-free attention head |
| OmniParser v2 (+GPT-4o) | YOLO+Florence | AGPL/MIT split | 39.6% | detection pipeline |
| Qwen2.5-VL-7B | 7B | Apache-2.0 | 29.0% | `bbox_2d` / 0–1000 points; widely hosted |
| OS-Atlas-7B | 7B | Apache-2.0 | 18.9% | good on standard mobile |
| UGround-V1-7B | 7B | Qwen2-VL terms | 16.5% | web-grounding lineage |
| Aria-UI | 3.9B act. | conditional | 11.3% | older generation |
| ShowUI-2B | 2B | Apache-2.0 | 7.7% | tiny & fast, lower accuracy |

> ScreenSpot-Pro is deliberately hard and desktop-skewed (tiny icons, 4K pro apps).
> On ordinary phone screens these models score much higher. Because the hierarchy
> already covers the easy cases, the vision fallback faces the **hard** cases —
> which favors Holo1.5-7B / UI-TARS-1.5-7B / GUI-Actor-7B.

Other open pointing/grounding options: **Molmo / MolmoPoint-GUI** (Apache-2.0,
pointing tokens), **GLM-4.5V**, **Kimi-VL**, InternVL-based GUI models.

### Fast OCR with bounding boxes

| Engine | Latency (single image) | Notes |
|---|---|---|
| **Apple Vision** (`VNRecognizeTextRequest`) | ~100–500 ms | Neural-Engine, zero Python deps, macOS only, 95–99% on clean text |
| Tesseract | ~450–770 ms | fastest OSS, tiny footprint |
| RapidOCR | lightweight | ONNX PaddleOCR, cross-platform |
| EasyOCR | ~715 ms | |
| PaddleOCR PP-OCRv5 | ~2 s | most accurate, Apache-2.0, ONNX/OpenVINO/TensorRT acceleration |
| Surya | — | best for dense layout |

For UI screenshots you mostly need fast line/word boxes → **Apple Vision** on Mac,
**RapidOCR** for portability.

---

## 4. Hosting & access options

### Self-host on Apple Silicon (preferred where reasonable)
- YOLOv8 ~**17–21 ms/frame with MPS** (~47–60 FPS) vs ~68–71 ms CPU; YOLO26 has a
  native MLX path (Mar 2026).
- OmniParser detection runs on Mac but CPU is multi-second → use MPS.
- Apple Vision OCR is native and fast.
- A 7B grounding VLM is the hard part locally: feasible on a 32–64 GB Apple Silicon
  Mac via MLX / llama.cpp Q4 (rule of thumb: 32 GB ⇒ ~30B Q4, 64 GB ⇒ ~70B Q4), but
  slower than a GPU. Better on a small CUDA box (≥16–24 GB VRAM for a 7B VLM;
  RTX 3060/8 GB minimum for OmniParser).
- Quantization: GGUF (llama.cpp/Ollama), ONNX (OCR/YOLO), CoreML (Apple Neural
  Engine), TensorRT (NVIDIA). **UI-TARS GGUF is discouraged** (quality regressions) —
  prefer vLLM.
- A Synology NAS is fine for storing weights and a headless Linux container, but it
  has **no GPU/MPS** — keep heavy inference on the Mac or a GPU host.

### Hosted inference
- **Hugging Face Inference Endpoints** (dedicated, vLLM/TGI, GPU autoscaling) can
  serve Qwen2.5-VL, UI-TARS, Holo1.5 — any vLLM/transformers-compatible model.
  HF serverless **Inference Providers** route to partners. OmniParser is **not**
  hosted by any HF provider (run it yourself or via Replicate).
- **Replicate** (OmniParser v2 ready-made), **Fireworks** (Qwen2.5-VL serverless +
  LoRA), **Baseten** (Truss, custom models), **Together**, **Fal**, **Modal / RunPod**
  (rent GPU, run your own container), **Groq / Cerebras / SambaNova** (fast, limited
  VLM support), **Novita / SiliconFlow / DeepInfra** (cheap hosted open models).
- Commercial multimodal APIs for grounding/analysis: **OpenAI** (GPT-5 vision),
  **Anthropic** (Claude vision), **Google** (Gemini vision). Best as an optional,
  configurable provider — **not** in the per-step hot loop.

### Rough latency budget per `analyze_screen`
- Hierarchy path (uiautomator2 / AccessibilityService): **~30–150 ms**
- Local YOLO + Apple Vision OCR fallback: **~150–600 ms**
- Hosted 7B grounding VLM call: **~0.5–2 s** (network + inference)
- Big multimodal API in loop: **~2–6 s** + per-image token cost → avoid in hot loop

---

## 5. Key takeaways

1. **Hierarchy-first, vision-second.** The fastest `analyze_screen` is a persistent
   on-device tree reader; vision is a fallback for ~10–20% of screens.
2. **The expensive part is the model round-trip, not the screenshot.** Returning
   structured JSON (element IDs + boxes) means the agent never needs the raw image
   in the normal case — the single biggest speed + cost win.
3. **Set-of-Marks:** assign integer IDs to elements; the agent acts on IDs, not
   pixels. This removes coordinate hallucination and shrinks token use.
4. **For the vision fallback, two tiers:** (a) detection + OCR with no LLM
   (OmniParser-detect or custom YOLO + Apple Vision/RapidOCR); (b) an optional
   grounding VLM (Holo1.5-7B, Apache-2.0) for hard screens.
5. **Licensing gates shippability:** avoid OmniParser `icon_detect` (AGPL) for a
   product — train your own YOLO; avoid research-only weights (Holo 3B/72B,
   Holo2-30B, UI-TARS-72B, Ferret-UI Lite). Holo1.5-7B, Qwen2.5/3-VL, and
   UI-TARS-1.5-7B are safe.

---

## 6. Caveats

- Some latency figures are inferential. uiautomator2 `dump_hierarchy()` is documented
  as "very fast" with tens-of-ms server logs but lacks a clean published benchmark vs.
  classic dump on complex screens; Apple Vision's exact ms on a phone screenshot is
  extrapolated; the cross-engine OCR comparison stitches separate benchmarks.
- ScreenSpot-Pro is hard and desktop-skewed; mobile grounding accuracy is generally
  higher, so the small models look worse than they will perform on typical Android UIs.
- Model licenses change — verify the exact HF model card before commercial use
  (especially Holo2-8B and Qwen-derived checkpoints).
- AccessibilityService / uiautomator approaches all fail on canvas / games / secure
  surfaces — vision is mandatory there.
- Agentic-framework AndroidWorld scores measure full LLM agents, not the
  `analyze_screen` perception layer in isolation.
