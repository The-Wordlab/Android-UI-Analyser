# iOS simulators

`aua` drives iOS simulators with the same agent-facing surface it offers for Android: `analyze`
returns elements with stable ids, and every `*-and-analyze` action, `has`, `wait`, `goto`, flows
and maps work unchanged. Select the platform once:

```bash
aua --platform ios devices                 # or: export AUA_PLATFORM=ios
aua --platform ios --format compact analyze
```

or in `~/.config/android-ui-analyser/config.yaml`:

```yaml
device:
  platform: ios
  serial: "iPhone 17"          # optional: a UDID or a unique simulator name
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

Physical iPhones are not supported: AXe drives simulators only.

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
- **Connecting.** With no `--serial`, the single booted simulator is used; several booted ones
  need `--serial`. Naming a shut-down simulator boots it first (`boot_timeout_s`).
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
| `ui.tree`, `ui.input`, `ui.screenshot`, `ui.read_deadline`, `ui.peek` (dashboard tiles) | `device.logs` (unified log diagnostics) |
| `device.touch` (held touches, single-attempt taps) | `device.recording` (`simctl io recordVideo`) |
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

## Troubleshooting

| Symptom | Fix |
|---|---|
| `AXe (axe) was not found on PATH` | `brew tap cameroncooke/axe && brew install axe`, or set `platforms.ios.axe_path`. |
| `no booted iOS simulator` | `xcrun simctl boot <udid>` (UDIDs from `aua --platform ios devices`), or pass `--serial <name>` to boot by name. |
| `multiple booted iOS simulators` | Pass `--serial <udid>`. |
| `simulator … exposes no accessibility root yet` | The simulator is still showing its boot screen; retry once the home screen is up. |
| Typed text is missing accents | Update AXe; `aua` already pastes non-ASCII text, but the field must accept paste. |
