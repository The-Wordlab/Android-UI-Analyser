# iOS simulators and physical-device screenshots

`aua` drives iOS simulators with the same agent-facing surface it offers for Android: `analyze`
returns elements with stable ids, and every `*-and-analyze` action, `has`, `wait`, `goto`, flows
and maps work unchanged. Start an agent-owned session on a free booted simulator. Keep the
platform on every CLI call (or set `AUA_PLATFORM=ios` for the process):

```bash
aua --platform ios session start --goal "Verify settings" --app com.example.app --no-provision-target
# Reuse the returned observation and recommended_call, then finish with the evidence.
aua --platform ios session finish
```

or in `~/.config/android-ui-analyser/config.yaml`:

```yaml
device:
  platform: ios
platforms:
  ios:
    axe_path: /opt/homebrew/bin/axe   # optional, when `axe` is not on PATH
    boot_timeout_s: 120               # optional, wait for a named simulator to boot
```

## What you need

| Requirement | Why |
|---|---|
| macOS with **Xcode 26 or newer** and at least one iOS simulator runtime | `xcrun simctl` owns simulator inventory, boot state, app lifecycle, screenshots, clipboard, links, permissions and location. |
| **AXe** (`aua --platform ios doctor --fix`, or `brew tap cameroncooke/axe && brew trust cameroncooke/axe && brew install axe`) | `axe describe-ui` reads the accessibility tree of a booted simulator; `axe tap/swipe/type/key/button` sends HID input. A current Homebrew refuses an untrusted tap, which is why the one-liner has the `trust` step. |
| A **booted simulator**, or a simulator name/UDID to boot | `aua --platform ios doctor` shows what is available and booted. |

A normal unpinned simulator session selects
a free booted simulator; it does not provision one. Omit `--headed`, which this adapter does not
support. Finish releases the lease without shutting down the simulator. For MCP, select iOS in
the server's config or environment before launch; `session_start` cannot switch platforms.

AXe drives simulators only. Physical iPhones and iPads use the separate screenshot-only
`ios-device` adapter below.

## Capture a physical iPhone or iPad

Requires a paired, connected device with Developer Mode enabled, and an Xcode/device
combination exposing `devicectl device capture screenshot`. Unlock the phone and accept
Trust This Computer if prompted. No helper is installed and the current app is left open.

```bash
aua --platform ios-device --serial <UDID> session start \
  --goal 'Capture current screen' --no-start-emulator
# View the returned observation.meta.raw_image. Capture a newer frame only if needed:
aua --platform ios-device screenshot /tmp/current.png
aua --platform ios-device session finish \
  --phase-done 'phase_1=Current screen captured and visually inspected'
```

CLI and MCP use the same `ui.screenshot` implementation, including normal session leasing,
evidence and cleanup. MCP callers select `device.platform: ios-device` and `device.serial`
through configuration, then use `session_start`, `screenshot`, and `session_finish`.
`aua --platform ios-device doctor` reports discovery status. Use an explicit UDID to
select the intended screen when multiple phones are connected.

This adapter exposes only screenshots. It cannot read the accessibility hierarchy, identify
the foreground app, tap/type, launch apps, or perform semantic UI assertions. An empty element
list means those capabilities are unavailable; it does not mean the screen is empty. Review
the PNG visually before completing a phase. Unsupported commands return capability errors.

Discovery accepts both current and older CoreDevice JSON layouts, excludes simulators and
non-iOS devices, and probes dormant paired connections before declaring them available.
Locked, disconnected, unsupported, missing-image and invalid-image failures never return an
older screenshot or switch to another device. Temporary CoreDevice output files are removed
on both success and failure; the requested screenshot/evidence file is retained.

## How it maps

- **Elements.** `type` is the accessibility type (`Button`, `StaticText`, `TextField`, `Switch`,
  `Cell`, …). `text` is the accessibility label; text entries and value-bearing controls (sliders,
  segmented controls, pickers) show their value as `text` and their caption as `content_desc`, so
  ids stay stable while the value moves. `resource_id` is the `accessibilityIdentifier`,
  so `--rid` and `has --by id` work when the app sets identifiers. Switches are `checkable` with
  `checked` read from their value. The keyboard is `window: ime`; the home screen is `system`.
- **Bounds are accessibility frames.** They are what AXe reports for the element, which can be
  larger than the view's layout frame: a decorative overlay that runs past the edge (a blurred
  rim, artwork bleeding out of a card's corner) or a container that combines its children
  widens the frame by a few points. Use them to locate and tap; for a pixel-exact layout
  assertion, measure the screenshot.
- **Coordinates.** AXe reports logical points; `aua` publishes screenshot pixels. The scale is
  measured once per connection from a screenshot and the accessibility root, so bounds, centers and
  taps line up with the PNG `analyze` returns.
- **Foreground app.** `screen.package` is the bundle identifier of the process that owns the
  screen (`com.apple.springboard` on the home screen).
- **Viewport checks.** `has`, selector waits and `scroll-to` reject accessibility nodes entirely
  outside the screen, including off-screen rows retained by SwiftUI. They use the same viewport
  bounds as `analyze`. Intersecting bounds are not proof that another view does not cover a node;
  use the screenshot for visual assertions.
- **Keys.** `enter`, `delete`/`backspace`, `tab`, `space`, `escape`, arrow keys, `home`, `lock`,
  `siri`, `side_button`, `apple_pay`, and `hid:<usage-code>` for anything else. There is no back
  key on iOS: `aua key-and-analyze back` performs the system back gesture (a swipe in from the left edge).
- **Typing.** ASCII goes through the HID keyboard. Anything else (accents, emoji, other scripts)
  is placed on the simulator pasteboard and pasted, so `input-and-analyze` accepts any text.
- **Connecting.** `session start` chooses a free booted simulator and retains its lease for
  later calls; omit `--serial`. An explicit UDID/unique name can select an existing simulator;
  naming a shut-down simulator boots it first (`boot_timeout_s`).
- **Apps.** `aua install <Build.app>` installs an `iphonesimulator` `.app` bundle (not an
  `.ipa`); `aua app launch|stop|clear|grant|exists` map to `simctl launch|terminate|…`.
  `aua app launch <bundle> --arg --uitesting --arg --feature-flag-x:on` hands the app its
  process arguments, the way a test build reads launch flags; add the global `--until
  rid:<landing>` to the same call to wait on the screen it opens instead of sleeping.
  `aua app uninstall <bundle-id> --yes` removes the app and its data.
  `clear` empties the app's data container and resets its permissions. Permission grants use
  `simctl privacy`, which covers calendar, contacts, photos, media library, microphone, motion,
  reminders, Siri and location; grants it cannot express (camera, notifications) are left to the app's
  own prompts.

## Capabilities

| Works | Not yet |
|---|---|
| `ui.tree`, `ui.input`, `ui.query_batch`, `ui.screenshot`, `ui.read_deadline`, `ui.peek` (dashboard tiles) | `device.logs` (unified log diagnostics) |
| `device.touch` (held touches, single-attempt taps), `device.drag` (AXe swipe; no hold before moving) | `device.recording` (`simctl io recordVideo`) |
| `app.lifecycle`, `app.links`, `app.status`, `app.install` | `virtual_targets` (`aua virtual-target …` boot/create/delete) |
| `device.clipboard`, `device.location` | `device.orientation`, `device.keyboard`, `device.clock`, `device.airplane`, `app.files` |
| `app_database` (SQLite), `feature_flags` (UserDefaults and configured deeplinks) | `proxy`, `network*`, `microphone`, `device_agent`, `webview` |

A missing capability fails with `platform_capability_unsupported`; nothing falls back to `adb`.

## App data and test setup

SQLite databases are discovered inside the installed app's data container. Use the relative
path returned by `db list`; encrypted stores, Keychain and shared app-group containers are not
included. Reads use a consistent SQLite snapshot including committed WAL data without stopping
the app. `--coherent` additionally stops it. Mutations accept a single data-only SQL statement,
require `--yes`, create a restore point, validate integrity and relaunch by default.

```bash
aua --platform ios db list com.example.app
aua --platform ios db query com.example.app Documents/example.sqlite 'SELECT * FROM items'
aua --platform ios db execute com.example.app Documents/example.sqlite \
  "UPDATE items SET label='Fixture' WHERE id=1" --yes
aua --platform ios db backups com.example.app Documents/example.sqlite
aua --platform ios db restore com.example.app Documents/example.sqlite <backup-id> --yes
```

Feature-flag deeplinks use the same `flags.templates` config as Android. Verification reads
UserDefaults through the simulator's preferences service (not a potentially stale plist file).
The default preference domain is the bundle id; `--prefs-file` can select another plist in that
app's preferences directory. An initial iOS "Open in…" prompt must be handled through the UI;
an unverified flag is reported as failure, not accepted as a test precondition.

For direct setup without an app-owned deeplink, use a flow with an explicit `.plist` filename:

```yaml
name: fixture_setup
app: com.example.app
steps:
  - prefs_write:
      file: com.example.app.plist
      values:
        fixture_enabled: true
```

Run it with `aua --platform ios flow run --file fixture_setup.yaml`. Only string, boolean and
number values are accepted; unrelated keys are preserved. The original preferences are recorded
before writing and restored by session cleanup, including after repeated writes. This does not
grant access to arbitrary server-side or remote-config flags: the app must use these values.

Offline simulation and physical iPhones remain unsupported by this adapter. AUA does not change
the Mac's network to simulate a disconnected iPhone.

## Compound wait performance

A predicate such as `--until 'rid:result,!rid:spinner,text:Done'` evaluates its accessibility
terms against one fresh AXe tree per poll. The snapshot is discarded after that poll; later
polls and post-action observations read again. This reduces repeated native calls and prevents
conditions from different transition frames being combined into one successful check. Existing
OCR checks for visible text and deadline enforcement still apply.

An alternating comparison on one warm simulator (Xcode 27.1, AXe 1.8.0, three pairs) measured a
three-condition wait including its final observation at a median **3.22 s before, 2.31 s after**
(28% lower). Full accessibility reads fell from four to two: one for the predicate, one for the
returned observation. All six comparison runs returned observations. These figures describe this
local workload, not every device or UI flow.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `AXe (axe) was not found on PATH` | `brew tap cameroncooke/axe && brew install axe`, or set `platforms.ios.axe_path`. |
| `no booted iOS simulator` | `xcrun simctl boot <udid>` (UDIDs from `aua --platform ios devices`), or pass `--serial <name>` to boot by name. |
| `multiple booted iOS simulators` | Pass `--serial <udid>`. |
| `simulator … exposes no accessibility root yet` | The simulator is still showing its boot screen; retry once the home screen is up. |
| Typed text is missing accents | Update AXe; `aua` already pastes non-ASCII text, but the field must accept paste. |
