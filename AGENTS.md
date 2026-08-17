# AUA repository instructions for coding agents

Read `CLAUDE.md` for the complete development and verification guide. The following architecture
rule is non-negotiable for every new device-facing feature.

## Keep platform operations behind the platform adapter

`PlatformAdapter` is the gateway between AUA's agent-oriented core and native automation tools.
Android is the default implementation today; it is not permission to make new core code Android-
specific.

- Never add a new direct `adb`, `adbutils`, `uiautomator2`, emulator-console, `dumpsys`, `logcat`,
  `run-as`, or other native-tool call to the engine, CLI, MCP server, daemon, or a generic service.
- Define the platform-neutral operation on `PlatformAdapter`, or on a focused capability protocol
  owned and returned by the adapter. The adapter may delegate to its platform runtime; it does not
  need to become one giant class.
- Implement the operation for Android in `platforms/android.py` or an Android-only module reached
  exclusively through `AndroidPlatform`. Declare the matching capability in the adapter.
- Core code calls the selected adapter. Do not add new `platform == "android"` branches in the
  engine, CLI, MCP, or daemon, and do not bypass `PlatformFactory`/the `aua.platforms` registry.
  The explicitly marked legacy monkeypatch shim is migration debt and must not be extended.
- An optional operation must fail with an explicit unsupported-capability result when the selected
  adapter does not provide it. It must never silently fall back to Android tooling.
- Keep CLI and MCP behavior on the same engine path. Do not implement a platform operation once for
  CLI and separately for MCP.
- Add a platform-neutral test with a fake adapter and an Android regression test. The neutral test
  must prove the feature does not invoke Android tooling directly.

Existing direct Android calls are migration debt, not examples to copy. When a feature touches one,
move the touched boundary behind the adapter when practical; do not expand the leak.

Before considering a device-facing feature complete, be able to answer:

1. What is its platform-neutral contract and capability name?
2. Where is the Android implementation?
3. What happens on an adapter that does not support it?
4. Do CLI and MCP reach the same implementation?
5. Does a fake-adapter test prove the core is platform-independent?
