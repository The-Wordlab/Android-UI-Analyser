# Web browsers

`aua` can launch an isolated Playwright browser page or attach to one user-approved tab in an
existing Chrome profile, then drive it through the same semantic surface as Android and iOS:
`analyze`, `has`, waits, stable-id actions, screenshots, flows, maps, and goal sessions. The
browser DOM is normalized to AUA `Element` rows; callers do not need a second selector or
response format.

## Install and start

Install the optional Python transport and one Playwright browser:

```bash
uv pip install -e '.[web]'
playwright install chromium
```

If Google Chrome is already installed, `channel: chrome` uses it and the separate browser
download is unnecessary. Select a URL as the target:

```bash
aua --platform web --serial https://example.test/app doctor
aua --platform web --serial https://example.test/app --format compact analyze
aua --platform web --serial https://example.test/app tap-and-analyze --rid continue
```

As with every AUA global option, `--platform` and `--serial` go before the subcommand. For a
project, configuration is less repetitive:

```yaml
device:
  platform: web
  serial: https://example.test/app

platforms:
  web:
    browser: chromium          # chromium | firefox | webkit
    headless: true             # false opens a visible browser window
    viewport_width: 1280
    viewport_height: 800
    # channel: chrome          # use an installed Chrome build
    # storage_state: .aua/auth.json  # seed cookies/local storage; keep this file private
    # ignore_https_errors: false
    # bypass_csp: false
    # service_workers: allow       # allow | block
    # proxy_server: http://127.0.0.1:8080
    # proxy_bypass: localhost,127.0.0.1
    # proxy_username: fixture
    # proxy_password_env: AUA_WEB_PROXY_PASSWORD
```

`platforms.web.url` is an alternative to `device.serial`. AUA accepts only absolute HTTP(S)
URLs and refuses credentials embedded in a URL. Do not put secret query parameters in the target
URL: target identities appear in local lease and journal metadata. Use a private Playwright
`storage_state` file for authentication.

## Connections and sessions

### Parallel isolated sessions

For concurrent agents, configure the destination with `platforms.web.url` and omit
`device.serial` / `--serial`:

```yaml
device:
  platform: web
platforms:
  web:
    url: https://example.test/app
    headless: false
    context_slots: 4
```

Run `aua --config /absolute/path/web.yaml session start --goal "Check the catalog" --headed`.
Each slot is a separate browser context in its own warm daemon. AUA's normal lease registry
assigns a free slot to the calling agent and keeps later commands on that target. The session
records its owner and target; finish releases only that lease. Start each agent with a distinct
agent process, or the usual per-worker `AUA_CACHE__DIR` when workers share an ancestor process.
Keep the default shared lease registry. Do not impersonate an owner or force another agent's lease.

Increase `context_slots` if the pool is full, or wait for a session to finish. Slot identities are
stable for a configured URL, even when the page navigates. A URL passed as `--serial` remains an
explicit, exclusive target for backward compatibility. Attached modes below are also exclusive;
they never create a replacement window when the configured target is leased.

### Extension attachment

Use the extension mode when a task needs a login or browser state that already exists in your
normal Chrome profile. Isolated Playwright remains the default and is still the right mode for
repeatable QA, storage/network controls, and untrusted automation.

Install the native host and copy the bundled extension into a stable user directory:

```bash
aua browser extension install
# Then open chrome://extensions, enable Developer mode, choose Load unpacked,
# and select the extension_path printed by the command.
aua browser extension status
```

Configure a separate profile for attached operation:

```yaml
device:
  platform: web
  serial: existing-chrome

platforms:
  web:
    connection: existing-chrome
    attach_timeout_ms: 30000
    action_timeout_ms: 5000
```

Start `aua analyze` or `aua session start` in a terminal, then open the AUA extension on the exact
HTTP(S) tab you want to share and choose **Attach this tab**. The initial command waits up to
`attach_timeout_ms`; the warm daemon keeps that one attachment for later commands. The Chrome
debugger banner is expected. Choose **Detach** in the extension, run `session finish`, stop the
daemon, or close the AUA process to release it. A killed process also closes the authenticated
native bridge, which makes the extension detach automatically.

Attached mode intentionally has a narrower authority boundary:

- The extension receives `activeTab`, `debugger`, and `nativeMessaging`; it has no cookie,
  history, broad host, downloads, clipboard, or all-tabs permission.
- Only the tab approved from the extension popup can be analyzed, pictured, focused, navigated,
  clicked, typed into, or scrolled. `browser pages` reports only that tab.
- AUA cannot silently select another tab, close a personal tab, export/clear storage or cache,
  reset the profile, change offline/throttle/proxy/CORS rules, mock traffic, record HAR, or start a
  Playwright trace. Those calls return `platform_capability_unsupported`.
- Goal sessions do not snapshot or restore the personal profile. `session finish` only detaches
  the tab. Maps retain AUA's existing redacted structural skeleton; dynamic page content and form
  values are not made durable.
- Page actions are real user actions. Review any operation that posts, sends, buys, deletes,
  follows, or otherwise changes external state just as you would when driving the UI yourself.

The native bridge uses a freshly generated secret in a mode-0600 config file plus a mode-0600
Unix socket, and the native-host manifest accepts only the bundled extension's stable ID. Current
extension attachment supports Chrome/Chromium on macOS and Linux. It does not require Playwright
or a remote-debugging port, so it works with Chrome's normal default profile.

### Existing Electron or Chromium window

Enable a loopback remote-debugging port when starting the app, for example
`electron . --remote-debugging-port=9222`, then configure:

```yaml
device:
  platform: web
platforms:
  web:
    connection: existing-cdp
    cdp_endpoint: http://127.0.0.1:9222
    # Required if several eligible windows are open; an exact URL, not a navigation request:
    page_url: file:///absolute/path/to/example-app/index.html
```

Use the same `session start`, semantic actions and `session finish` commands. AUA reuses the
running app's window, preload bridge, login and workspace. It does not launch or restart the app,
open pages, change its viewport, restore storage, or close it at finish. Exactly one matching
HTTP(S)/file page is required; ambiguous or missing windows return a typed error. Once attached,
the connection stays on that page even when other windows open; closing it fails subsequent
actions instead of switching to another window. `browser pages` reports only the selected page.
Screenshots use CSS pixels to match DOM bounds on Retina displays.

The endpoint identifies one exclusive lease, regardless of the selected page. This avoids agents
changing different windows backed by the same app state concurrently. Network/storage mutation,
traces, native dialogs, recording and Electron guest `<webview>` targeting are not provided by this
attachment mode. Console and network diagnostics cover renderer browser activity; main-process
logs, Node requests and child-process output need separate instrumentation.

## Perception stack

In isolated mode, Playwright owns browser launch, navigation, input, and the native viewport
screenshot. Extension attachment uses Chrome's debugger API; CDP attachment uses Playwright
against the existing page. Both provide navigation, trusted input, a DOM snapshot and the native
viewport screenshot. AUA then uses the same layered perception stack as
its device adapters:

1. A DOM/ARIA snapshot supplies roles, accessible names and descriptions, visible text, state,
   stable test ids, parent relationships, and viewport-clipped bounds.
2. AUA normalizes those nodes into its existing response model and stable selector history.
3. `analyze --source auto` can reuse AUA's configured OCR, detector, and grounding providers on
   the Playwright screenshot for canvas, WebGL, image text, or incomplete accessibility markup.
4. The merged observation feeds the same memory, maps, flows, waits, and session evidence used by
   Android and iOS.

This split is deliberate: browser-native semantics stay the cheap primary source, while pixel
vision remains a fallback rather than replacing exact DOM evidence. Chromium-specific DevTools
features are not required, so the core contract also works with Firefox and WebKit.

## What AUA sees

- Rendered nodes intersecting the viewport become AUA elements. Off-screen DOM nodes do not make
  `has` pass and do not make `scroll-to` claim the target is already visible.
- `data-testid`, `data-test-id`, `data-test`, then HTML `id` become `resource_id`, so `--rid`
  produces the same stable `rid:…` identities used on Android and iOS.
- Visible text, form values/placeholders, associated labels, `aria-label`, roles, enabled/focused,
  checked/selected, password, and scrollable state map into the canonical element schema.
- Open shadow roots and iframe documents are traversed. Frame elements are reported with their
  page-viewport bounds, and `browser pages` reports the frame URL/name alongside each tab.
- Screenshots use the browser viewport at device scale factor 1, so DOM bounds and screenshot
  pixels share one coordinate space.
- `screen.package` is the page hostname and `screen.activity` is its path/query, preserving the
  existing output contract without exposing the complete URL as app identity.

AUA intentionally keeps its semantic selector contract rather than exposing CSS/XPath. Test ids,
accessible names, and visible text make flows portable and force the tested page to remain usable
through its accessibility surface.

## Browser lifetime and authentication

The default warm AUA daemon owns the browser context, so successive commands in one session share
page state, cookies, session storage, and local storage. The context is isolated and discarded when
the daemon/runtime closes. If daemon mode is disabled, each standalone CLI invocation starts at the
configured URL; use one `flow run` call for a multi-step journey.

`storage_state` seeds a context from a Playwright JSON file. AUA reads it but does not write browser
state back to disk, so ordinary web actions create no persistent device mutation and require no
device teardown ledger entry. A goal session snapshots cookies, local/IndexedDB state,
sessionStorage, the current URL, and AUA browser controls; `session finish` recreates the context
from that baseline, which also clears HTTP cache. Browser state remains process-local, so use the
default warm daemon for a session spanning several CLI calls.

## Browser lab controls

### Diagnostics included with observations

Every browser `analyze` response includes `meta.browser_diagnostics`; actions such as
`tap-and-analyze` include it in `observation.meta`. It carries ordinary console messages,
JavaScript errors, requests, HTTP statuses and failed requests, including cross-origin calls.
It survives compact/delta output and the default action projection. No separate `browser logs`
call is needed for this summary.

Standalone reads cover the preceding 30 seconds; actions cover their own window, including an
adopted `--until` wait. Windows can overlap so intermediate polls cannot consume evidence needed
by the final response. Only events observed after attachment are available. The default limit is
20 events (`logs.limit`, bounded to 100), prioritizing failures, with 500-character string fields.
`total_count`, `omitted_count` and `truncated` describe the retained window; `buffer_overflow`, when
available, signals older events lost from the transport buffer. Empty summaries confirm a
successful read; `unavailable` or `omitted` indicates missing diagnostics, not a silent app.

`logs.enabled: false` / `--no-app-logs` disables this enrichment. Request bodies and headers are
excluded, URL queries/fragments are stripped by the browser transport, and session evidence
bundles retain counts rather than raw console/network text. Existing native `app_logs` behavior
is unchanged.

### Explicit inspection and controls

`aua browser --help` exposes the browser-specific layer while ordinary UI commands retain the
same Android/iOS response model:

```bash
aua browser status
aua browser logs --kind console --kind page_error
aua browser storage                         # names/metadata only
aua browser storage --include-values        # explicit sensitive-value opt-in
aua browser storage-export .aua/login.json
aua browser storage-clear --kind local --kind indexeddb
aua browser cache-clear

aua browser offline                         # `online` restores connectivity
aua browser throttle --latency-ms 200 --download-kbps 750 --upload-kbps 250
aua browser cors-add --origin https://app.example.test --host api.example.test
aua browser proxy-set http://127.0.0.1:8080 --password-env AUA_WEB_PROXY_PASSWORD

aua browser har-start .aua/checkout.har     # stop writes the HAR
aua browser har-replay .aua/checkout.har --not-found abort
aua browser mock-add '**/api/orders' --status 201 --body '{"id":"fixture"}'
aua browser pages                           # page-1, page-2, frames…
aua browser page-select page-2
aua browser trace-start                     # stop into a Playwright trace.zip
```

The equivalent MCP surface is grouped into `browser_status`, `browser_logs`, `browser_storage`,
`browser_network`, `browser_cors`, `browser_proxy`, `browser_har`, `browser_mock`, `browser_pages`,
and `browser_trace`. Both interfaces call the same Engine operations.

For agents working on web apps, start a focused MCP server:

```bash
aua --platform web --config /absolute/path/aua-web.yaml mcp --tool-profile web
```

The `web` tool profile advertises session lifecycle, semantic UI actions and assertions,
screenshots, map/flow navigation and browser controls. It omits native device administration,
Android accessibility actions, app databases, helpers and other unrelated tool schemas. The
smaller catalogue reduces the context an agent processes each turn. Actions still return their
post-action observation; use it instead of calling `analyze_screen` again.

In an MCP client, append `"--tool-profile", "web"` to the server's `args`, or set
`AUA_MCP_TOOL_PROFILE=web` in its environment. The CLI option wins over the environment. The
default remains `full`, including when `--platform web` is selected, so existing clients keep
their complete catalogue. Restart the server/client session after changing profiles; a cached
call to a tool omitted from the active profile returns `tool_profile_unavailable` without
dispatching. The profile does not select the platform or grant capabilities: use `--platform web`
or a web config, and unsupported adapter operations retain their explicit capability errors.

- Logs include console records, page exceptions, requests, responses, failed requests, and
  WebSocket opens. They also feed AUA's existing `device.logs`/per-action diagnostic path.
- Storage inspection covers cookies, local/session storage, IndexedDB, CacheStorage, and service
  worker registrations. Values are withheld unless explicitly requested. Exports refuse to
  overwrite a file.
- CORS handling is scoped to target host globs and an allowed origin. It rewrites only matching
  responses/preflights; AUA never launches Chrome with global web-security disabled.
- Proxy changes recreate only this isolated browser context and preserve its storage/session
  state. Passwords are accepted through an environment-variable name, never printed in status.
- Bandwidth limits use Chromium DevTools. Firefox/WebKit support offline and latency controls but
  return an explicit unsupported-capability error for bandwidth limits.
- HAR replay and request mocks are deterministic context routes. Network status reports rule
  metadata but omits mock bodies and proxy secrets. HAR/storage exports may contain credentials
  or private response data; keep them outside git and publish only after sanitizing.
- `browser reset` returns storage and every control to configured startup state. `session finish`
  restores the state captured at session start instead.

## Capabilities and current boundary

| Works | Explicitly unsupported |
|---|---|
| `ui.tree`, `ui.input`, `ui.screenshot`, `device.logs` | app install/uninstall/lifecycle and private app files |
| `app.links` for HTTP(S), URL-aware maps/flows, iframe DOMs, popups/tabs | device shell, recording, clipboard and location |
| cookies, local/session storage, IndexedDB, CacheStorage, service workers | native database/datastore and feature-flag services |
| offline, throttle, scoped CORS, context proxy, HAR and request mocks in isolated mode | profile-wide storage/network controls in attached mode |
| browser traces and shared AUA screenshots/OCR/detection/grounding | physical-device controls and native radio profiles |
| Chromium, Firefox, WebKit isolated; existing Chrome/Chromium tab on macOS/Linux | Chromium-only bandwidth shaping when using Firefox/WebKit |

Unsupported operations return `platform_capability_unsupported`; web never imports or falls back to
Android tooling. Browser-only operations are declared named runtime capabilities and fail clearly
when another adapter does not provide them.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `web support needs the optional Playwright dependency` | Install `android-ui-analyser[web]` in the environment that owns the `aua` executable. |
| `Executable doesn't exist` or browser launch fails | Run `playwright install chromium`, or configure `browser: chromium` and `channel: chrome` for installed Google Chrome. |
| Every command starts from the initial URL | Keep the default daemon enabled, start a goal session, or put the journey in one `flow run`. |
| A control has no stable `rid:` | Add `data-testid` or an HTML `id`; text/ARIA labels still receive semantic stable keys. |
| A service worker bypasses a mock/HAR rule | Configure `service_workers: block` for deterministic request interception. |
| Bandwidth throttling is rejected | Use Chromium, or keep only `--latency-ms` on Firefox/WebKit. |
| `chrome_extension_not_attached` | Start the AUA command first, then open the extension on the target tab and choose **Attach this tab**. |
| The extension immediately detaches | Run `aua browser extension install`, restart Chrome after native-host installation, and confirm `aua browser extension status` is green. |
