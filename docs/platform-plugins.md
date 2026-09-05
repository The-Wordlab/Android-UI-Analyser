# Platform adapter API v1

AUA selects one platform strategy from `device.platform`, `--platform`, or `AUA_PLATFORM`.
`android` is the built-in default. A third-party distribution can add another strategy without
changing AUA's Engine, CLI, MCP server, daemon, or state stores.

API v1 separates three scopes:

1. `PlatformAdapter` discovers and connects targets, normalizes native accessibility trees, and
   owns platform-wide operations.
2. `TargetRuntime` performs semantic operations on one connected target.
3. A named service returned by `PlatformAdapter.load_capability()` implements an optional
   host-wide feature such as `virtual_targets`.

Capability names are promises. A missing optional capability returns
`platform_capability_unsupported`; a declared but structurally incomplete one returns
`platform_capability_invalid`. A non-Android adapter never falls back to ADB or another Android
tool.

## Packaging and selection

Publish the adapter class as a Python entry point:

```toml
[project.entry-points."aua.platforms"]
ios = "my_aua_ios:IOSPlatform"
```

The entry-point name is the value users put in configuration:

```yaml
device:
  platform: ios
  serial: simulator-1       # optional adapter-local target id

platforms:
  ios:                      # opaque to AUA; validated by IOSPlatform
    endpoint_env: IOS_AUTOMATION_ENDPOINT
```

AUA indexes entry-point names without importing their modules. It imports only the selected
adapter, checks `platform_api_version`, constructs it, passes only `platforms.<selected-name>` to
`validate_options()`, and freezes the returned mapping as `adapter.options`. A broken unselected
plugin therefore cannot stop Android or another plugin from loading.

API v1 is exported as `PLATFORM_API_VERSION == 1`. These selection failures have stable codes:

| Condition | Error code |
| --- | --- |
| Entry point cannot import or adapter initialization fails | `platform_plugin_load_failed` |
| Plugin API version differs from AUA | `platform_api_incompatible` |
| More than one distribution publishes the selected name | `platform_plugin_ambiguous` |
| Capability declaration is incomplete | `platform_capability_invalid` |

`validate_options()` should reject unknown keys. Keep credentials out of configuration; accept an
environment-variable name and resolve it inside the adapter when needed.

Use explicit JSON-compatible routing options and `*_env` (or camel-case `*Env`) references.
Do not derive target identity from an undocumented ambient variable or process working directory.
The core transports exactly the selected JSON option mapping by inherited descriptor; it fingerprints
the full mapping and the current values of its named environment references. Those values remain
inside a local HMAC, never in the persisted digest. Startup/validation errors expose the exception
type, not arbitrary plugin exception text that may contain credentials. Plugins remain responsible
for keeping their own runtime logs and operation errors free of secrets.

Empty option mappings use a constant non-secret identity, so default Android recovery needs no
key file. Nonempty mappings use a local `.platform-options-hmac-key`: ledger/watchdog identity
lives in `~/.cache/android-ui-analyser/device-state`, and worker identity keys live under `cache.dir`.
Back up the ledger and its key together. Losing or corrupting that key cannot be repaired by
re-entering the same options: restore the original key as well. AUA retains the pending undo and
names the missing identity proof; `--force` does not authorize a different endpoint or boot.

## Attached-target contract

An adapter that passes the API-v1 attached-target profile declares:

```python
capabilities = frozenset({"ui.tree", "ui.input", "ui.screenshot"})
```

Its minimum shape is:

```python
from collections.abc import Mapping, Sequence

from android_ui_analyser.platforms import (
    AppContext,
    Element,
    DisplayGeometry,
    NormalizedTree,
    PLATFORM_API_VERSION,
    PlatformAdapter,
    ScreenImage,
    TargetInfo,
    TargetRuntime,
)


class IOSRuntime(TargetRuntime):
    target_id = "simulator-1"

    def window_size(self) -> tuple[int, int]: ...
    def display_geometry(self) -> DisplayGeometry: ...
    def dump_hierarchy(self, compressed: bool = False) -> str: ...
    def screenshot(self) -> ScreenImage: ...
    def current_app(self) -> AppContext: ...

    # All coordinates received or returned here are canonical screenshot pixels.
    def click(self, x: int, y: int) -> None: ...
    def long_click(self, x: int, y: int, duration_ms: int = 600) -> None: ...
    def send_text(self, text: str, *, clear: bool = True) -> None: ...
    def clear_text(self) -> None: ...
    def send_ime_action(self, action: str = "search") -> None: ...
    def swipe(self, x1: int, y1: int, x2: int, y2: int,
              duration_ms: int = 300) -> None: ...
    def press(self, key: str) -> None: ...
    def find_text(self, text: str, *, match="contains", ignore_case=False,
                  by="text"): ...


class IOSPlatform(PlatformAdapter):
    platform_api_version = PLATFORM_API_VERSION
    capabilities = frozenset({"ui.tree", "ui.input", "ui.screenshot"})

    def validate_options(self, options: Mapping[str, object]): ...

    def list_targets(self) -> list[TargetInfo]:
        return [
            TargetInfo(
                target_id="simulator-1",
                platform=self.name,
                os_name="ios",
                os_version="20.0",
            )
        ]

    def connect(self, target_id: str | None = None) -> TargetRuntime: ...

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        geometry: DisplayGeometry | None = None,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        # Parse native nodes and transform every native bound with
        # geometry.bounds_to_canonical(...).
        return NormalizedTree(elements=[Element(...)], app_id="org.example.app")

    def capture_screenshot(self, runtime: TargetRuntime) -> ScreenImage:
        return runtime.screenshot()
```

`list_targets()` returns neutral `TargetInfo` values. `connect()` returns a `TargetRuntime`; the
runtime's `target_id` is its stable adapter-local identity. `current_app()` returns
`AppContext(app_id, surface_id)`. Historical `serial`, `package`, and `activity` spellings remain
compatibility projections, not identities a new adapter needs to model as Android concepts.

Two platforms may return the same `target_id`. Shared coordination uses
`TargetRef(platform, target_id)`, so their leases, sessions, journals, captures, diagnostic marks,
and teardown records remain separate. Bare legacy serials are interpreted as Android only.
Plugin storage components are percent-encoded, including uppercase bytes on case-insensitive
hosts, and use a delimiter that cannot occur in Android's legacy keys.

## Coordinate contract

AUA has one public coordinate space: physical pixels in the screenshot for the current frame.
Element/OCR/detection bounds, annotations, action targets, and `window_size()` all use it.

If a native automation API uses logical points, a cropped viewport, or rotated coordinates, the
runtime returns an invertible `DisplayGeometry`:

```python
geometry = DisplayGeometry(
    native_size=(100.0, 200.0),
    canonical_size=(400, 200),
    # 2x clockwise rotation: native -> screenshot pixels
    native_to_canonical=(0.0, 2.0, -2.0, 0.0, 400.0, 0.0),
)
```

The adapter passes `geometry.bounds_to_canonical(native_bounds)` into each normalized `Element`.
The runtime accepts canonical points from the Engine and calls `geometry.to_native(point)` before
its native input transport. Do not infer scale or orientation in shared code.

## Capability scopes

Complete structural specifications live in `platforms/contracts.py` and
`platforms/services.py`. The API-v1 names are:

| Scope | Capability names |
| --- | --- |
| Runtime | `ui.tree`, `ui.input`, `app.lifecycle`, `app.files`, `app.links`, `device.keyboard`, `device.clipboard`, `device.location`, `device.orientation`, `device.airplane`, `device.media`, `device.recording`, `device.clock`, `device.accessibility`, `device.touch`, `device.proxy`, `device.shell` |
| Adapter | `ui.screenshot`, `app.status`, `app.install`, `device.logs` |
| Service | `app_database`, `developer_settings`, `device_agent`, `feature_flags`, `microphone`, `network`, `network_profiles`, `proxy`, `target_supervision`, `virtual_targets`, `webview` |

Declare only complete capabilities. Runtime calls are resolved with
`adapter.runtime_capability(name, runtime)`, adapter calls with
`adapter.adapter_capability(name)`, and services with `adapter.capability(name)`. Service loading is
lazy and cached:

```python
def load_capability(self, name: str):
    if name == "virtual_targets":
        from .simulators import IOSSimulatorService
        return IOSSimulatorService(self.options)
    return None
```

Native framework imports belong inside the selected adapter/runtime/service modules. Do not put
them in the Engine, CLI, MCP server, daemon, dashboard, or generic state modules.

`target_supervision` is optional lifecycle metadata for dashboard observability. Its
`target_supervision_status(target_id, cache_dir=...)` operation returns a
`TargetSupervisionStatus` (or `None` for an unmanaged target), allowing a platform to report the
owner, instance identity, idle-retirement policy, and monitor health without exposing native
process-record storage to shared code.

## Virtual targets

`virtual_targets` is the neutral optional provisioning service. Its public
`VirtualTargetsService` protocol and all typed requests/results are exported from the stable
`android_ui_analyser.platforms` facade. It implements
`list_virtual_targets`, `select_virtual_target`, `start_virtual_target`,
`provision_virtual_target`, `virtual_target_status`, `stop_virtual_targets`,
`stop_virtual_target_instance`, `reclaim_virtual_targets`, `create_virtual_target`, and
`delete_virtual_target`. Typed requests/results are in `platforms.virtual_targets`.

Keep these identities distinct:

- `definition_id`: reusable simulator definition;
- `target_id`: attached automation target;
- `instance_token`: exact process/boot AUA started.

A successful start/provision returns an opaque `instance_token`. Rollback calls
`stop_virtual_target_instance()` with that token, never a reusable target id that could now belong
to another worker. `virtual_devices` and `emulator` remain accepted capability aliases;
`aua emulator ...` remains an Android-compatible command alias. New integrations use
`virtual_targets`, `aua virtual-target ...`, and the `virtual_target_*` MCP tools. An attached-only
adapter simply omits this capability.

## Published conformance profile

Plugin test suites can run the executable attached-target profile without depending on pytest:

```python
from android_ui_analyser.platforms import (
    AttachedTargetCase,
    PlatformFactory,
    run_attached_target_conformance,
)

adapter = PlatformFactory(config).create()
report = run_attached_target_conformance(
    adapter,
    AttachedTargetCase(
        target_id="simulator-1",
        element_text="Continue",
        expected_bounds=(300, 20, 360, 60),
        expected_app_id="org.example.app",
        require_non_identity_geometry=True,
        input_element_text="Continue",       # optional text-input check
        key_name="back",                     # optional semantic-key check
        expected_scrollable_bounds=(0, 80, 400, 760),  # optional verified swipe
    ),
)
```

Use a deterministic fake or disposable prepared screen: the profile intentionally taps the center
of the named element. It verifies API version and capability declarations, discovery/connection,
absence of Android transport conveniences, canonical geometry, screenshot dimensions, hierarchy
normalization through the Engine, Engine-routed tap and wait, requested text/key/verified-swipe
checks, and a typed refusal for one omitted optional capability. The connected runtime is closed
before the report is returned.

An adapter that passes this profile has proved the attached UI/input foundation. It has not proved
every AUA feature or every process boundary: add focused tests for each optional runtime, adapter,
or service capability it declares, including CLI/MCP transport parity, detached-worker
configuration identity, and failure/ownership/rollback behavior for persistent mutations.

AUA's repository gate additionally builds the strict fixture as a wheel, discovers its real entry
point in a fresh isolated Python process, blocks Android imports and native subprocess execution,
and runs this profile. That gate proves the published surface is sufficient for an independently
packaged attached-target plugin; it is separate from a plugin's own native transport tests.

The contract does not itself implement iOS, web, or another native transport, and it does not
rename the `aua` command or project.
