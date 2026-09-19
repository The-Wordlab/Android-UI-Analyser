"""Explicit MCP discovery profiles, independent of platform action dispatch."""

from enum import Enum


class ToolProfile(str, Enum):
    full = "full"
    web = "web"


# Deliberately enumerated: new native/admin tools must not silently grow this surface.
# Keep reusable navigation and memory alongside the basic browser interaction loop.
WEB_TOOL_NAMES = frozenset(
    {
        "configure",
        "session_start",
        "session_progress",
        "session_review",
        "session_finish",
        "analyze_screen",
        "has",
        "tap_and_analyze",
        "input_and_analyze",
        "clear_and_analyze",
        "swipe_and_analyze",
        "key_and_analyze",
        "scroll_to_and_analyze",
        "wait_and_analyze",
        "wait_changed_and_analyze",
        "wait_stable_and_analyze",
        "await_and_analyze",
        "expect_and_analyze",
        "screenshot",
        "inspect",
        "open_link_and_analyze",
        "orient",
        "reach",
        "goto",
        "map_find",
        "flow_list",
        "flow_run",
        "flow_save",
        "flow_delete",
        "browser_status",
        "browser_logs",
        "browser_storage",
        "browser_network",
        "browser_cors",
        "browser_proxy",
        "browser_har",
        "browser_mock",
        "browser_pages",
        "browser_trace",
    }
)

WEB_INSTRUCTIONS = (
    "This server advertises AUA's web tool profile. Start with session_start(goal); it "
    "observes once and recommends a verified route, saved flow, or analyzed action. "
    "Use its recommended_call when available. Analyzed actions already return a fresh "
    "observation and element ids: do not follow them with analyze_screen unless you need "
    "a different view. Put semantic arrival terms in until to fold verification into the "
    "action; include at least one positive arrival term. Use phase_done on the next call "
    "to acknowledge a goal checkpoint without an extra round trip. Use browser_pages "
    "for tabs/popups and frames, browser_logs for console/page/network diagnostics, and "
    "browser_network/browser_cors/browser_proxy/browser_har/browser_mock for browser "
    "controls. browser_storage values are sensitive and require explicit opt-in. "
    "Call session_finish to restore the starting browser context; session_review "
    "reports caller-visible calls, retries and run health. Unsafe or destructive effects "
    "require explicit authorization. This profile changes tool discovery, not the selected "
    "platform: unsupported adapter operations still fail explicitly. Only listed tools "
    "are available in this process; restart with --tool-profile full for other tools."
)
