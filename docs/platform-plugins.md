# Platform adapters

AUA selects a platform strategy from `device.platform` (or `--platform` / `AUA_PLATFORM`).
`android` remains the only built-in strategy and the default, so existing commands behave exactly
as before. Selection happens once for an engine/process; callers do **not** pass a platform on each
`tap`, `analyze`, `goto`, or other call.

A platform package implements `PlatformAdapter`: target discovery, connection to an object that
implements the current `Device` runtime contract, native UI-tree capture, and normalization into
AUA's canonical `Element` objects. The engine then reuses the same analysis, tap-and-analyze,
history, map, flow, and `goto` orchestration.

There are two replaceable layers:

1. `Device` is the per-target strategy: capture, input, app lifecycle, files, logs, proxy wiring,
   media, clock, and other semantic target actions. Its implementation may use XCUITest,
   Playwright, WebDriver, a remote service, or anything else; core does not know.
2. `PlatformAdapter.capability(name)` supplies optional host/platform services such as virtual
   devices, network shaping, proxy/CA management, feature flags, app databases, developer settings,
   and microphone injection. The stable names and required members are in
   `android_ui_analyser.platforms.services.CAPABILITY_METHODS`.

An adapter can ship incrementally. If it does not provide an optional service, AUA returns
`platform_capability_unsupported`; if it claims a service but omits part of its common interface,
AUA returns `platform_capability_invalid`. A selected non-Android platform never falls back to ADB.

Third-party Python distributions register their adapter through an entry point:

```toml
[project.entry-points."aua.platforms"]
ios = "my_aua_ios:IOSPlatform"
```

```python
from android_ui_analyser.platforms import NormalizedTree, PlatformAdapter

class IOSPlatform(PlatformAdapter):
    capabilities = frozenset(
        {"ui.tree", "ui.input", "ui.screenshot", "app.install", "virtual_devices"}
    )

    def connect(self, target_id=None): ...
    def list_targets(self): ...
    def normalize_tree(self, raw_tree, screen_size, *, ignored_app_ids=()):
        return NormalizedTree(elements=[...], app_id="com.example.app")

    def load_capability(self, name):
        if name == "virtual_devices":
            return IOSSimulatorService()  # list_avds/start/stop/status/... contract
        return None
```

`load_capability` is lazy and its returned object is cached for the adapter lifetime. Keep native
tool imports inside the platform runtime/service implementation, never in the engine, CLI, MCP,
daemon, dashboard, or capture sidecar. `tests/test_platform_boundary.py` enforces that separation.

The gate does not implement iOS or web and does not rename the `aua` command or project.
