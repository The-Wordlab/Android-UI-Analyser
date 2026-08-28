# Changelog

All notable, user-facing changes to `aua` (android-ui-analyser) are recorded in this file.

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and the
project follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

Every release is a git tag `vX.Y.Z` with a matching GitHub Release carrying that version's
notes, so you can check for a newer version — and read what changed — without pulling `main`.
`aua --version` prints the version you have installed.

## [Unreleased]

## [0.13.0] - 2026-08-28

First tagged release. AUA has been developed in the open since 2026-08-20 with no tags, no
releases and no changelog, so this entry is one honest summary of everything the tool does as
of the first tag rather than a reconstruction of the untagged versions it passed through.

### Breaking

- **`elements[].id` is now the stable element identity, and `stable_elements` is gone.** `id`
  used to be a frame-local reading-order ordinal, renumbered on every analyze, with the durable
  id sitting beside it in a second array. The stable id is now published as `id` on every
  surface — elements, `parent`, `next_actions`, `meta.element_diff`, and the `acting` element an
  action reports — so a published id pastes straight back into any command
  (`aua tap-and-analyze rid:continue_btn`); MCP accepts either kind. The shipped agent guidance
  was regenerated to match. **If you read `stable_elements`, or joined two lists to address one
  element, that code must change.**
- **`aua dashboard start` now publishes an mDNS name, binds every interface, and serves with no
  access token.** It used to bind loopback only and require a 43-character token on every network
  path, and it drives the device, streams logcat and queries app databases. The start result
  prints a warning naming exactly what is exposed. Restore the old shape with `--auth` (re-arm the
  token), `--local` (loopback only), `--name ""` (publish nothing) or `--port N`, or move the
  default itself in the new `dashboard` config block. Port 80 is a preference, not a requirement:
  where the kernel refuses it the dashboard falls back to 48765 and says so, while a `--port` you
  pinned is never moved.

### Added

- Versioned releases: every `vX.Y.Z` tag is tested and published as a GitHub Release with wheel,
  source archive, helper APK, and notes from this changelog. `aua update --check` compares the
  installed version with the latest release without pulling `main`; `--json` and exit code 10 make
  the same check usable from automation.
- `aua drive "<goal>"` — hand AUA a goal in plain words and it picks and taps each step itself,
  one host round trip per step (~1500 ms/step). It needs no root, no sideloaded service and no
  permission grant, so it works on retail phones and Play-image emulators. It also stops instead
  of looping: a control that was tapped and changed nothing is remembered per element, and the run
  ends with `no_progress` rather than pressing the same row until the budget runs out.
- `aua helper drive "<goal>"` — the same goal-driving rule running entirely on the device via the
  optional helper APK: observe, choose, act, observe, with no host round trip at all.
- `aua app exists|status|foreground|launch|launch-and-analyze|restart|restart-and-analyze|stop|kill|clear|grant <pkg>`
  for package presence, version and lifecycle, and `aua shell <argv…>` for leased, argv-quoted,
  read-only device diagnostics (each output stream capped at 256 KiB) — neither drops to raw adb,
  and `aua emulator start` plus the app lifecycle calls stream real progress instead of blocking
  silently.
- `aua dashboard start|status|open|qr|stop|run` — the browser dashboard is a persistent service you
  start once, ask the status of, reopen, hand to a phone as a QR code, and stop. With
  `--name aua` it publishes an mDNS record and answers at a typeable `http://aua.local/` from the
  host and from any device on the network, with no privilege and no `/etc/hosts` edit.
- The dashboard opens on a live grid of every device, discovers emulators that appear later, and
  shows each one's lease holder, the owner that started it, the idle-watchdog state and the
  remaining auto-stop time. A device detail view adds the agent I/O journal, logcat, the screen
  map, a database workspace, a proxy panel and local-model control — syntax-coloured and
  filterable, with a fails-only toggle on the journal — and the Lease chip gains an Unlease button
  that runs the same clean-then-release the CLI escape hatch does, so the next agent never inherits
  somebody else's proxy, clock or radio change.
- `aua mock rewrite` — patch a real response (status, headers, whole body, JSON field set/delete,
  literal substitution) instead of only stubbing it away, from the CLI, MCP or the dashboard's
  click-a-request panel, with `--host` and `--times` scoping.
- Every observed action now carries `app_logs`: what the app under test itself logged during that
  action's own window, scoped to its process, priority-filtered and line-budgeted. A crash still
  supersedes it with the fuller `crash_evidence` block.
- `aua logcat prefs show|set|reset --app <pkg>` — say once which log tags matter for an app and
  every later action in every later session honours it: `--ignore-tag`, `--keep-tag` (rescue a tag
  the built-in noise list hides), `--only-tag`, levels, line and per-tag caps. No filter can ever
  drop an `F` line.
- Locale awareness: every analyze reports `meta.device_locale`, `aua devices` lists each device's
  locale, and `aua has` / `wait-and-analyze --for` / `scroll-to-and-analyze` match a query written
  in one language against a device rendering another — reporting which string key and rendering
  matched, and naming the expected rendering on a miss.
- `input-and-analyze --send rid:<control>` types and taps the app's own semantic send control in
  one call, and every input result now reports `submitted` plus a `recommended_call`, so "the text
  was typed" is no longer mistaken for "the app accepted it".
- `aua capture sheet <path> --since last-action --max-frames N --timestamps` — a bounded,
  timestamped PNG contact sheet of the rolling frame buffer, with no ffmpeg on the host.
- `aua session start --animations` (or `--needs animations`, or animation words in the goal)
  enables the device's animation scales for motion and easing checks, and restores their exact
  prior values at `session finish`.
- `aua lease transfer <serial>` / `aua lease accept <token>` / `aua lease cancel-transfer` — hand a
  running device to a child agent without resetting it, via a one-time five-minute token.
- `aua session start` owns device selection: `--needs root,play,proxy,animations` picks a capable
  free target or boots a matching AVD, and when every compatible target is leased by a live agent
  it leaves them alone and provisions a unique read-only instance.
- AUA no longer reports a confident wrong screen. `meta.screen_moved` plus a `WARNING:` note say
  when an overlay, interstitial or dialog arrived between your last observation and this call, with
  `capture_hint` pointing at the frames that show it arriving; and when a tap lands on an activity
  that has started but not rendered, AUA waits for content and returns the real destination — or
  says `stale_risk` with an `arrival` verdict when it cannot.
- Coaching for runs that cannot succeed: a target that is not on screen hands back the screen AUA
  already looked at instead of telling you to go and analyze; a positive `rid:` no mapped screen has
  ever carried is named as impossible with the nearest real ids; three relaunches or the same call
  three times on one target earns a hint; and `session start` warns when the screen a goal is about
  was last seen empty, quoting its own words, and names device changes another run left behind.
- Every action accepts `stable_key` (`--key` on the CLI, `stable_key` on MCP, with optional bounds
  to pick between list rows sharing one key) — the safe way to act on an observation another process
  produced.

### Changed

- `aua db query` no longer stops the app: it copies the database plus WAL through `run-as` and reads
  it host-side, so the screen you were looking at is still there afterwards. **Script-breaking** —
  pass `--coherent` for the old stop-and-relaunch behaviour when you need transactional coherence.
- `aua session finish` returns a compact verdict by default (`--full`, or the returned
  `full_review_call`, for the whole timeline). **Script-breaking** — incomplete closure now exits
  nonzero, but keeps the session and the lease alive and returns the missing checkpoints plus one
  exact next call; `--allow-incomplete` now explicitly means "abandon the goal".
- One leased device is implicit: omit `--serial` from ordinary commands. Switching targets needs the
  warned `aua lease acquire <new> --replace`, which cleans and releases the old device, and a dead
  owner's lease is released immediately.
- The frame is kept by default (`with_image` is on), so `meta.raw_image` always points at a picture
  you can look at when the element tree does not explain itself. **Script-breaking** — `runs/` now
  accumulates frames, pruned to the newest N auto-named frames per device.
- An action result leads with the screen you asked for: `observation` is rendered in its declared
  position rather than appended behind a dozen diagnostics, and `note` sits above it.
- A stable key names exactly one element — colliding keys on a reusable row layout take an ordinal
  suffix, so `rid:row#2` is the second such row down the screen, while a bare key still returns the
  whole group — and selectors now accept the spelling AUA published, so `--rid rid:continue_btn` no
  longer misses, and the same holds for `text:` with `--text` and `desc:` with `--desc`.
- `goto` and arrival proof now treat context variants of one screen as the same screen family when
  the mapped `logical_name`, `state` and `surface` agree; loading shells and modals stay distinct.
- The on-device driver shows 28 actionable nodes per screen instead of 14. 14 truncated 13.2% of 638
  real harvested screens — including fixed bottom navigation bars, which are last in tree order,
  first in importance, and can never be scrolled into view. 28 truncates 2.0%.
- `aua --help` page 1 now lists every command instead of ~55 lines of global options, and an unknown
  command is answered with the real vocabulary instead of a hint naming nothing.
- Helper protocol 1 → 2 on both sides: an older helper APK left on a device is refused with a hint
  rather than answering without the stall fields, where a caller cannot tell "never stalled" from
  "does not report stalls".

### Removed

- `stable_elements` — its content is now `elements[].id`.
- `next_actions` from action responses by default (re-enable with `output.next_actions`); the learned
  per-control cost it carried moved onto the element itself as `Element.cost`.
- `meta.element_diff` from the default action observation — `--observe-meta all` or naming it brings
  it back, and `--format delta` keeps it unconditionally.
- `tier_used`, `via`, `path` and `duration_ms` from the action observation's `meta`, and `rid` from
  the default columns. None of the first four changes the next call and `analyze` still reports all
  of them; `rid` was a restatement of `id` on well over half the rows, `--fields id,rid` returns it,
  and `--where-rid` never depended on it.

### Fixed

- Two agents driving two emulators through the proxy at once no longer read each other's traffic:
  every piece of proxy state — rules, cassette record, flow log, bodies — is keyed by device serial,
  so one agent's rewrite rule cannot fire on the other's device and `mock clear` cannot wipe rules
  its owner never armed.
- The proxy panel no longer serves the app's bearer token, API key, device key or stream token into
  a page people paste into bug reports. Header and field names survive; the values do not.
- A session start can no longer kill or hijack another worker's emulator: port allocation consults
  the host-wide lease registry, a boot detects a serial collision before touching any device, every
  stop path refuses a serial whose live lease belongs to someone else and reports `skipped_leased`,
  and a device-bound warm daemon claims exactly its own device instead of acquiring a different free
  emulator and then refusing to use it — which used to strand a lease for its full TTL on a device
  the caller never touched.
- Parallel session startup, transport recovery and stale Android UiAutomation are fenced and
  recovered once rather than cascading into a failed run.
- The dashboard was broken outright and silently: a stray newline in the page script blanked the
  whole page behind a working header, and clicks coerced the element id to a number — never valid
  for a stable id — so a click on an element the page had just drawn came back as "needs a
  non-negative AUA element id". The ids the page drew are now recorded too, so clicking a box no
  longer validates against whichever screen last wrote the cache.
- Stable ids are published on all four boundaries, not one: the CLI's `--fields` / `--format tsv`
  path, MCP and the daemon used to hand back frame ordinals while the same payload carried stable ids
  elsewhere — one response, two id spaces, naming the same controls.
- `aua input-and-analyze rid:searchField "some text"` was refused with "with --rid/--desc, pass only
  the text to type" — typing by a published id, the whole point of publishing them, did not work.
- `meta.element_diff.changed` crashed the response with an unhashable-dict `TypeError` whenever
  something actually changed between two frames, which is why it surfaced as an intermittent
  `internal_error` on real taps.
- Dashboard grid tiles were pinned to the first frame they ever drew, and a dead capture served its
  last file forever; tiles now key on a frame token that moves with the bytes, fall through to a live
  screencap, and never take the UiAutomation slot from the agent that is driving.
- The dashboard's evidence panels are readable: a scrolled-away journal reader is no longer dragged
  down one row per event (a pill counts what you have not seen and clicking it returns you), the live
  frame column hugs the device instead of taking the wide column, logcat gets a full-width row and
  stops chopping identifiers mid-token, and the copy buttons work on `http://aua.local/` — not a
  browser secure context, so `navigator.clipboard` is simply absent.
- A dashboard-armed proxy rule left no undo record at all — the one mutation the device ledger exists
  to retract was the one it never saw — and `mock map` silently dropped the host and `--times` budget
  it was given, arming a stub against every host, forever.
- Clicking an older proxy exchange showed a newer one's headers and body, because rows were matched
  on a sequence number that restarts with every mitmdump process.
- A malformed `--set` / `--header` / `--replace` printed a Python traceback and exited 1 instead of
  one line of JSON and exit 2.
- Soft lints, screenshots and screen recording no longer bypass the session daemon or connect to the
  device on their own, and a headed session never lands on a physical device.

### Performance

- The screen returned by every action costs 68% less: 919 → 292 tokens on one real settings screen.
  `meta` dropped 16 empty keys and three hints no action asks for, and element rows stopped restating
  `clickable: false`, `enabled: true`, `checked: false` and `selected: false` on every row. `analyze`
  itself is untouched byte for byte, and a switch's `checked: false` — the one flag whose off state is
  the whole reading — always survives.
- A change is reported once rather than three times. On one real tap, change reporting was 473 of 921
  tokens (51%), of which `element_diff.removed` alone was 91% — ids of elements that are gone from the
  screen and cannot be tapped, read or asserted on. What remains answers the same question for a tenth
  of the price, and names the added and removed *text* rather than ids you would have to look up.
- Responses are ~350 tokens lighter with the derived `next_actions` list off by default: it re-listed
  the actionable subset of `elements` and cost more than the whole list it was filtering, while
  silently capping at 12 when 15 controls were clickable. The dashboard now trims what it serves the
  same way — one inspection 3365 B → 817 B, one tap 3498 B → 919 B, `meta` 29 keys → 14 — and what is
  stored is untouched, so every drawn box stays clickable.
- A miss is no longer the most expensive payload the tool emits: on one measured miss, the screen
  attached to "your target is not here" went from 147 rows / ~9200 tokens to 7 rows / ~277, in the row
  shape you already know. A healthy action is also 98 bytes cheaper, because `capture_hint` is attached
  only when something is actually wrong.
- The new evidence is close to free, each measured on one specific action: keeping the frame was
  247.2 ms against 254.7 ms without it (median of five on an emulator, because the screenshot was
  already captured and decoded and `false` only threw it away), `app_logs` costs +16 ms on a ~1050 ms
  action, and waiting for a real destination costs only the calls that were previously returning a
  wrong answer — a settled same-screen tap is 427 ms against 426 ms.

### Notes

- The optional on-device helper APK is **off by default** (`helper.enabled: false`). Turning that one
  switch on does everything: AUA probes rootability, installs the APK and enables the service itself.
  It needs `adb root`, so it cannot be used on retail phones or Play-image emulators — that is what
  `aua drive` is for.
- The published dashboard is unauthenticated on your network by default, and it drives the device,
  streams logcat and queries app databases. `--auth`, `--local`, or `auth: true` in config restores the
  guarded shape.
- The on-device and host driving lanes share one scoring rule (word overlap with a first-token bonus,
  not a language model), measured at 82.2% on 5,741 held-out rows and 17 of 19 reachable destinations
  on a live device. `done` and `no_progress`-style "is X on screen" goals are deliberately unsupported:
  a false "done" is a silent claim of success.
- Time travel (`clock set`) still invalidates auth; always `clock restore`. Use `network_offline`
  (never airplane mode) to prove offline behaviour, and `network_restore` or `session finish` to put it
  back.

[Unreleased]: https://github.com/The-Wordlab/Android-UI-Analyser/compare/v0.13.0...HEAD
[0.13.0]: https://github.com/The-Wordlab/Android-UI-Analyser/releases/tag/v0.13.0
