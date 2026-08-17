# Platform adapters

AUA selects a platform strategy from `device.platform` (or `--platform` / `AUA_PLATFORM`).
`android` remains the only built-in strategy and the default, so existing commands behave exactly
as before.

A platform package implements `PlatformAdapter`: target discovery, connection to an object that
implements the current `Device` runtime contract, native UI-tree capture, and normalization into
AUA's canonical `Element` objects. The engine then reuses the same analysis, tap-and-analyze,
history, map, flow, and `goto` orchestration.

Third-party Python distributions register their adapter through an entry point:

```toml
[project.entry-points."aua.platforms"]
ios = "my_aua_ios:IOSPlatform"
```

```python
from android_ui_analyser.platforms import NormalizedTree, PlatformAdapter

class IOSPlatform(PlatformAdapter):
    capabilities = frozenset({"ui.tree", "ui.input", "ui.screenshot", "app.install"})

    def connect(self, target_id=None): ...
    def list_targets(self): ...
    def normalize_tree(self, raw_tree, screen_size, *, ignored_app_ids=()):
        return NormalizedTree(elements=[...], app_id="com.example.app")
```

The capability set is the extension point for features that are not universal. The initial gate
does not implement iOS or web and does not rename the `aua` command or project.
