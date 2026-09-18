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
- Open shadow roots are traversed. Cross-origin and nested iframe documents are not included yet.
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
teardown ledger entry.

## Capabilities and current boundary

| Works | Explicitly unsupported |
|---|---|
| `ui.tree`, `ui.input`, `ui.screenshot` | app install/uninstall/lifecycle and private app files |
| `app.links` for HTTP(S) navigation | device shell, logs, recording, clipboard and location |
| `analyze`, `has`, waits, scroll-to, actions, flows, maps, screenshots | AUA network/offline/proxy controls and database/feature-flag services |
| Chromium, Firefox, WebKit; headless or headed | attaching to an already-running browser, popup/tab switching, iframe DOMs |

Unsupported operations return `platform_capability_unsupported`; web never imports or falls back to
Android tooling. Browser console/network/storage inspection can be added later as named optional
platform capabilities without changing the core action path.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `web support needs the optional Playwright dependency` | Install `android-ui-analyser[web]` in the environment that owns the `aua` executable. |
| `Executable doesn't exist` or browser launch fails | Run `playwright install chromium`, or configure `browser: chromium` and `channel: chrome` for installed Google Chrome. |
| Every command starts from the initial URL | Keep the default daemon enabled, start a goal session, or put the journey in one `flow run`. |
| A control has no stable `rid:` | Add `data-testid` or an HTML `id`; text/ARIA labels still receive semantic stable keys. |
| Content inside an iframe is missing | Iframe traversal is not in the first web adapter; test the iframe URL directly when possible. |
