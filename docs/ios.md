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
| **AXe** (`brew tap cameroncooke/axe && brew install axe`) | `axe describe-ui` reads the accessibility tree of a booted simulator; `axe tap/swipe/type/key/button` sends HID input. |
| A **booted simulator**, or a simulator name/UDID to boot | `aua --platform ios doctor` shows what is available and booted. |

Physical iPhones are not supported: AXe drives simulators only.

## How it maps

- **Elements.** `type` is the accessibility type (`Button`, `StaticText`, `TextField`, `Switch`,
  `Cell`, …). `text` is the accessibility label; text entries and value-bearing controls (sliders,
  segmented controls, pickers) show their value as `text` and their caption as `content_desc`, so
  ids stay stable while the value moves. `resource_id` is the `accessibilityIdentifier`,
  so `--rid` and `has --by id` work when the app sets identifiers. Switches are `checkable` with
  `checked` read from their value. The keyboard is `window: ime`; the home screen is `system`.
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
| | Android-only services: `app_database`, `proxy`, `network*`, `feature_flags`, `microphone`, `device_agent`, `webview` |

A missing capability fails with `platform_capability_unsupported`; nothing falls back to `adb`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `AXe (axe) was not found on PATH` | `brew tap cameroncooke/axe && brew install axe`, or set `platforms.ios.axe_path`. |
| `no booted iOS simulator` | `xcrun simctl boot <udid>` (UDIDs from `aua --platform ios devices`), or pass `--serial <name>` to boot by name. |
| `multiple booted iOS simulators` | Pass `--serial <udid>`. |
| `simulator … exposes no accessibility root yet` | The simulator is still showing its boot screen; retry once the home screen is up. |
| Typed text is missing accents | Update AXe; `aua` already pastes non-ASCII text, but the field must accept paste. |
