# Native / AOT speedups roadmap

Tracked on `feat/aua-fast`. Goal: cut host cold-start and on-device dump cost
without rewriting the Python engine.

| Phase | Item | Status | Notes |
|---|---|---|---|
| **1** | **C thin client (`aua-fast`)** | **done** | Speaks existing daemon JSON-lines protocol; falls back to `aua`. |
| 2 | On-device native / incremental a11y | not started | Custom agent caching tree + AccessibilityEvent deltas. Largest payoff, largest risk (OEM variance, Play policy). |
| 3 | FlatBuffers / binary dump | not started | Requires phase 2 or a u2 fork; host + device must version together. |
| 4 | WebSocket push MCP | not started | Daemon pushes screen-changed events; agents stop polling. |
| 5 | Multi-emulator fanout | not started | Orchestration layer over N serials; product more than protocol. |
| 6 | CoreML / GPU vision | not started | Optional OCR/detection path on Apple Silicon; keep RapidOCR fallback. |
| 7 | Nuitka / freeze of Python CLI | not started | Alternative to thin client; heavier build, full flag coverage. Prefer phase 1. |

## Why phase 1 first

The daemon already holds the warm Engine. Cold cost is **process startup**, not perception.
A ~50 KB C binary that only does `connect(AF_UNIX)` + write/read one JSON line removes that
floor for `analyze` / `tap` / `has` / `key` when agents call them in a loop.

## Non-goals (for now)

- Replacing `uiautomator2` in-tree
- Shipping an APK / privileged AccessibilityService in the default install
- Breaking the existing Python CLI / MCP surface
