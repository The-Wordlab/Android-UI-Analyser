# Native / AOT speedups roadmap

Tracked on `feat/native-phases-2-6` (stacked on `feat/aua-fast`). Goal: cut host
cold-start and on-device dump cost without rewriting the Python engine.

| Phase | Item | Status | Notes |
|---|---|---|---|
| **1** | **C thin client (`aua-fast`)** | **done** | Speaks existing daemon JSON-lines protocol; falls back to `aua`. |
| **2** | **Incremental a11y (host)** | **done (host slice)** | `perf.skip_unchanged_analyze` + XML-hash reuse; `aua wait-and-analyze --changed` / MCP `wait_changed`. **No APK** — true AccessibilityEvent deltas still deferred (OEM/Play risk). |
| **3** | **Binary dump** | **done (host)** | `--format delta` + `--format msgpack` (AUA1 zlib frames). `schemas/hierarchy.fbs` draft for future on-device FlatBuffers. |
| **4** | **WebSocket push** | **done** | `daemon.push_ws_port` → localhost WS `screen_changed` events; MCP/CLI long-poll via `wait_changed`. |
| **5** | **Multi-emulator fanout** | **done** | Per-serial daemon sockets (`daemon.sock.<serial>`); `aua fanout [--serials …] <cmd>`. |
| **6** | **CoreML / GPU vision** | **done (defaults)** | Apple Vision default `recognition_level: fast` (Neural Engine); YOLO/OmniParser already default `device: mps`. |
| 7 | Nuitka / freeze of Python CLI | skipped | Prefer phase 1 thin client. |

## Honest limits

- Phase 2 does **not** ship an on-device AccessibilityService. Host still polls
  `dump_hierarchy`; skip-unchanged avoids re-parse when the XML hash matches.
- Phase 3 FlatBuffers on-device producer still needs phase 2 (or a u2 fork).
- Phase 4 push is still fingerprint-polled under the hood; WS removes *agent*
  polling, not the host↔device dump.

## Non-goals (for now)

- Replacing `uiautomator2` in-tree
- Shipping an APK / privileged AccessibilityService in the default install
- Breaking the existing Python CLI / MCP surface
- Nuitka freeze (phase 7)
