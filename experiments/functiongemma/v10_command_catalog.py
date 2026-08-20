"""A do/don't matrix over the AUA command surface.

The V9 corpus taught twenty-two hand-written situations. That is enough to teach a handful of
discriminations and not enough to teach *what each command is for*: measured afterwards, the model
had learned "a non-tap command is on the menu, so choose it" rather than any notion of when each
command applies. Hand-writing one situation per command does not fix that either — a command only
becomes meaningful when the model has seen it be **right** in its own situation and **wrong** in
someone else's.

This module encodes each command once, as a precondition plus a role, and derives both directions
mechanically:

* the **do** case — the command's own precondition holds, so it is the oracle;
* the **don't** cases — it appears as a distractor in every other command's do case, where its
  precondition is explicitly false in the observation.

Because the distractor pool is every *other* command, each additional entry strengthens every
existing family rather than only adding one of its own. A command that is only ever correct with
explicit authorization (data deletion, configuration change, clock manipulation) is never an
oracle: it appears solely as a distractor, and the authorization families make refusal the answer.

Every string is fictional and app-agnostic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Command:
    """One AUA command, its precondition, and how it describes itself to the model."""

    tool: str
    #: Observation/outcome key that is true exactly when this command is the right next step.
    signal: str
    #: Short description used as the candidate purpose. ``{t}`` is the topic noun, ``{n}`` the tag.
    purpose: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    #: Group label, used to keep distractors plausible (a waiting command distracts a waiting one).
    group: str = "general"
    #: Never an oracle without explicit authorization; only ever offered as a distractor.
    requires_authorization: bool = False
    #: Reading the screen again when the frame is already fresh is redundant, not wrong.
    redundant_when_wrong: bool = False


def _a(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


# The catalogue. Preconditions are deliberately phrased as observation flags so a case can state
# the situation without naming the answer.
COMMANDS: tuple[Command, ...] = (
    # ---- direct interaction -------------------------------------------------------------
    Command(
        "tap_and_analyze",
        "target_visible",
        "Open the {t} {n} control.",
        _a(text="{t} {n}"),
        "interaction",
    ),
    Command(
        "long_press_and_analyze",
        "context_menu_required",
        "Long-press the {t} {n} entry to raise its context actions.",
        _a(text="{t} {n}"),
        "interaction",
    ),
    Command(
        "double_tap_and_analyze",
        "double_activation_required",
        "Double-tap the {t} {n} canvas area.",
        _a(text="{t} {n}"),
        "interaction",
    ),
    Command(
        "input_and_analyze",
        "text_entry_required",
        "Type the prepared value into the {t} field.",
        _a(rid="field{t}", text="{t}-{n}"),
        "interaction",
    ),
    Command(
        "clear_and_analyze",
        "field_holds_stale_text",
        "Clear the text currently in the {t} field.",
        _a(rid="field{t}"),
        "interaction",
    ),
    Command(
        "erase_and_analyze",
        "characters_must_be_removed",
        "Erase characters from the {t} field.",
        _a(rid="field{t}"),
        "interaction",
    ),
    Command(
        "paste_and_analyze",
        "clipboard_holds_required_value",
        "Paste the clipboard contents into the focused {t} field.",
        _a(),
        "interaction",
    ),
    Command(
        "copy",
        "value_must_be_captured",
        "Copy the {t} {n} value to the clipboard.",
        _a(text="{t} {n}"),
        "interaction",
    ),
    Command(
        "key_and_analyze",
        "hardware_key_required",
        "Press the device navigation key.",
        _a(name="back"),
        "interaction",
    ),
    Command(
        "hide_keyboard_and_analyze",
        "soft_input_visible",
        "Dismiss the soft keyboard.",
        _a(),
        "interaction",
    ),
    Command(
        "a11y_action_and_analyze",
        "accessibility_action_required",
        "Perform the named accessibility action on the {t} node.",
        _a(rid="node{t}", action="expand"),
        "interaction",
    ),
    # ---- movement -----------------------------------------------------------------------
    Command(
        "scroll_to_and_analyze",
        "target_below_viewport",
        "Scroll the list toward {t} {n}.",
        _a(text="{t} {n}"),
        "movement",
    ),
    Command(
        "scroll_and_analyze",
        "list_position_must_change",
        "Scroll the current container one page.",
        _a(direction="down"),
        "movement",
    ),
    Command(
        "swipe_and_analyze",
        "gesture_required",
        "Swipe across the {t} pane.",
        _a(direction="left"),
        "movement",
    ),
    Command(
        "a11y_scroll_and_analyze",
        "scrollable_node_known",
        "Scroll the {t} scrollable node through the accessibility tree.",
        _a(rid="list{t}"),
        "movement",
    ),
    # ---- observation --------------------------------------------------------------------
    Command(
        "analyze_screen",
        "outcome_unknown",
        "Take one fresh uncached observation of the current frame.",
        _a(source="hierarchy", no_cache=True),
        "observation",
        redundant_when_wrong=True,
    ),
    Command(
        "expect_screen",
        "assertion_only",
        "Assert that {t} {n} is present.",
        _a(text="{t} {n}"),
        "observation",
    ),
    Command(
        "has",
        "presence_probe_only",
        "Check whether {t} {n} is on the current frame.",
        _a(text="{t} {n}"),
        "observation",
    ),
    Command(
        "inspect",
        "element_attributes_required",
        "Print the full attributes of the {t} {n} element.",
        _a(text="{t} {n}"),
        "observation",
    ),
    Command(
        "target",
        "label_resolution_required",
        "Report what the {t} {n} label actually addresses.",
        _a(text="{t} {n}"),
        "observation",
    ),
    Command(
        "resolve",
        "cross_frame_binding_required",
        "Rebind the earlier {t} element onto the current frame.",
        _a(text="{t} {n}"),
        "observation",
    ),
    Command(
        "screenshot",
        "pixel_record_required",
        "Save a screenshot of the current frame.",
        _a(),
        "observation",
    ),
    Command(
        "logcat",
        "app_died_unexpectedly",
        "Read the device log around the last action.",
        _a(grep="FATAL|ANR"),
        "observation",
    ),
    # ---- waiting ------------------------------------------------------------------------
    Command(
        "wait_and_analyze",
        "short_settle_required",
        "Wait briefly for the frame to settle.",
        _a(text="{t} {n}"),
        "waiting",
    ),
    Command(
        "await_and_analyze",
        "condition_pending",
        "Wait until the {t} condition holds.",
        _a(predicate="text:{t} {n}"),
        "waiting",
    ),
    Command(
        "job_start_await",
        "long_operation_running",
        "Detach the long read-only wait and reconnect to it by job id.",
        _a(predicate="rid:ready{t}", timeout_ms=180000),
        "waiting",
    ),
    Command(
        "job_status",
        "detached_job_outstanding",
        "Reconnect to the outstanding job without restarting its wait.",
        _a(job_id="job-{n}"),
        "waiting",
    ),
    # ---- routing ------------------------------------------------------------------------
    Command(
        "goto",
        "verified_route_available",
        "Use the recorded route to {t} {n}, proving each hop.",
        _a(screen="{t}-{n}"),
        "routing",
    ),
    Command(
        "flow_run",
        "saved_flow_matches",
        "Replay the saved {t} journey.",
        _a(name="{t}-{n}"),
        "routing",
    ),
    Command(
        "open_and_analyze",
        "deeplink_offered",
        "Open the offered {t} deeplink.",
        _a(url="myapp://{t}/{n}"),
        "routing",
    ),
    Command(
        "back_until_and_analyze",
        "nested_return_required",
        "Return through the nested screens, stopping on {t} {n}.",
        _a(target="text:{t} {n}"),
        "routing",
    ),
    Command(
        "app_restart_and_analyze",
        "app_state_unrecoverable",
        "Restart the application to a known screen.",
        _a(package="com.example.app"),
        "routing",
    ),
    # ---- session ------------------------------------------------------------------------
    Command(
        "session_progress",
        "phase_state_required",
        "Report the ordered goal phases and the current phase.",
        _a(),
        "session",
    ),
    Command(
        "session_finish",
        "all_phases_proved",
        "Finish the session and restore session-owned state.",
        _a(session_id="session-{n}"),
        "session",
    ),
    Command(
        "session_review",
        "call_accounting_required",
        "Report call accounting for the session.",
        _a(session_id="session-{n}"),
        "session",
        redundant_when_wrong=True,
    ),
    # ---- device and infrastructure -------------------------------------------------------
    Command(
        "lease_acquire",
        "device_held_by_other_owner",
        "Claim the device for this agent.",
        _a(serial="device-under-test"),
        "infrastructure",
    ),
    Command(
        "lease_release",
        "work_complete_device_owned",
        "Release this agent's device lease.",
        _a(serial="device-under-test"),
        "infrastructure",
    ),
    Command(
        "devices", "device_inventory_required", "List the attached devices.", _a(), "infrastructure"
    ),
    Command(
        "emulator_recommend_proxy",
        "system_ca_not_writable",
        "Identify an image able to host a system certificate.",
        _a(),
        "infrastructure",
    ),
    Command(
        "emulator_start",
        "no_emulator_running",
        "Boot the configured emulator image.",
        _a(avd="test-avd"),
        "infrastructure",
    ),
    Command(
        "install",
        "build_not_present",
        "Install the build on the device.",
        _a(apk="build-{n}.apk"),
        "infrastructure",
    ),
    Command(
        "dev_anim",
        "animations_interfere",
        "Turn device animations off for stable timing.",
        _a(state="off"),
        "infrastructure",
    ),
    Command(
        "network_offline",
        "offline_state_required",
        "Take the device network offline.",
        _a(),
        "infrastructure",
    ),
    Command(
        "network_restore",
        "network_must_return",
        "Restore the device network.",
        _a(),
        "infrastructure",
    ),
    Command(
        "orientation_set",
        "orientation_must_change",
        "Set the device orientation.",
        _a(value="landscape"),
        "infrastructure",
    ),
    # ---- helper -------------------------------------------------------------------------
    Command(
        "helper_enable",
        "helper_not_bound",
        "Switch the on-device helper on and confirm it is bound.",
        _a(),
        "helper",
    ),
    Command(
        "helper_status",
        "helper_binding_unknown",
        "Report whether the helper is installed and bound.",
        _a(),
        "helper",
    ),
    Command(
        "helper_tree",
        "helper_bound",
        "Read the hierarchy through the on-device helper.",
        _a(),
        "helper",
    ),
    Command(
        "helper_watch",
        "pushed_events_required",
        "Stream pushed screen-change events from the helper.",
        _a(timeout_ms=30000),
        "helper",
    ),
    # ---- stored data --------------------------------------------------------------------
    Command(
        "db_list", "database_inventory_required", "List the private database files.", _a(), "data"
    ),
    Command(
        "db_schema",
        "table_shape_required",
        "Return the tables and columns of the {t} database.",
        _a(database="{t}.db"),
        "data",
    ),
    Command(
        "db_query",
        "stored_rows_must_be_read",
        "Run one read-only query against a host-side snapshot.",
        _a(database="{t}.db", sql="SELECT * FROM entries LIMIT 20"),
        "data",
    ),
    Command(
        "db_backup",
        "restore_point_required",
        "Create a restore point for the {t} database.",
        _a(database="{t}.db"),
        "data",
    ),
    # ---- capture ------------------------------------------------------------------------
    Command(
        "capture_on",
        "rolling_capture_required",
        "Resume the rolling capture buffer.",
        _a(),
        "capture",
    ),
    Command(
        "capture_last",
        "recent_frames_required",
        "Emit the recent capture timeline.",
        _a(),
        "capture",
    ),
    # ---- knowledge ----------------------------------------------------------------------
    Command(
        "orient",
        "app_playbook_required",
        "Report what the tool already knows about this application.",
        _a(),
        "knowledge",
    ),
    Command(
        "map",
        "known_layout_required",
        "Print the recorded layout for this application.",
        _a(),
        "knowledge",
    ),
    Command(
        "memory_update",
        "screen_should_be_recorded",
        "Record the current screen in the application map.",
        _a(),
        "knowledge",
    ),
    # ---- authorization-gated: never an oracle, always a distractor -----------------------
    Command(
        "db_execute",
        "never",
        "Delete the stored {t} rows.",
        _a(database="{t}.db", sql="DELETE FROM entries"),
        "data",
        requires_authorization=True,
    ),
    Command(
        "db_restore",
        "never",
        "Restore an earlier backup over the {t} database.",
        _a(database="{t}.db"),
        "data",
        requires_authorization=True,
    ),
    Command(
        "memory_forget",
        "never",
        "Clear the recorded map for this application.",
        _a(app="com.example.app"),
        "knowledge",
        requires_authorization=True,
    ),
    Command(
        "flags_set",
        "never",
        "Change a runtime flag and restart the application.",
        _a(query="{t}_{n}=true"),
        "infrastructure",
        requires_authorization=True,
    ),
    Command(
        "clock_set",
        "never",
        "Move the device clock forward.",
        _a(iso="2030-01-01T00:00:00Z"),
        "infrastructure",
        requires_authorization=True,
    ),
    Command(
        "helper_remove",
        "never",
        "Switch off and uninstall the on-device helper.",
        _a(),
        "helper",
        requires_authorization=True,
    ),
    Command(
        "memory_forget_screen",
        "never",
        "Clear one recorded screen from the map.",
        _a(app="com.example.app", screen="{t}-{n}"),
        "knowledge",
        requires_authorization=True,
    ),
    Command(
        "proxy_ca_install",
        "never_without_rootable_target",
        "Install the interception certificate into the system trust store.",
        _a(),
        "infrastructure",
        requires_authorization=True,
    ),
)

#: Commands eligible to be the oracle of a do-case.
ORACLE_COMMANDS: tuple[Command, ...] = tuple(
    command for command in COMMANDS if not command.requires_authorization
)

#: Commands that may only ever appear as distractors.
AUTHORIZATION_GATED: tuple[Command, ...] = tuple(
    command for command in COMMANDS if command.requires_authorization
)

BY_TOOL: dict[str, Command] = {command.tool: command for command in COMMANDS}


def render(command: Command, topic: str, tag: str) -> dict[str, Any]:
    """Materialise one command as a candidate payload for *topic*/*tag*."""

    def fill(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("{t}", topic).replace("{n}", tag)
        return value

    return {
        "tool": command.tool,
        "arguments": {key: fill(value) for key, value in command.arguments.items()},
        "purpose": fill(command.purpose),
    }
