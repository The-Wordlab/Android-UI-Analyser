# Web browsers

`aua` can launch an isolated Playwright browser page and drive it through the same semantic
surface as Android and iOS: `analyze`, `has`, waits, stable-id actions, screenshots, flows, maps,
and goal sessions. The browser DOM is normalized to AUA `Element` rows; callers do not need a
second selector or response format.

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

## Perception stack

Playwright owns browser launch, navigation, input, and the native viewport screenshot. AUA then
uses the same layered perception stack as its device adapters:

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
| offline, throttle, scoped CORS, context proxy, HAR and request mocks | attaching to an already-running browser or extension automation |
| browser traces and shared AUA screenshots/OCR/detection/grounding | physical-device controls and native radio profiles |
| Chromium, Firefox, WebKit; headless or headed | Chromium-only bandwidth shaping when using Firefox/WebKit |

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
