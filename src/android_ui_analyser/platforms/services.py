"""Stable structural contracts for optional platform capability services.

The target runtime (`Device`) covers common screen/input/app operations. These named services
cover operations that are host-wide or not universal. A plugin may implement only the services
it supports, but claiming a service means implementing its complete structural surface below.
"""

from __future__ import annotations

from typing import Any

APP_DATABASE = "app_database"
DEVICE_AGENT = "device_agent"
DEVELOPER_SETTINGS = "developer_settings"
FEATURE_FLAGS = "feature_flags"
MICROPHONE = "microphone"
NETWORK = "network"
NETWORK_PROFILES = "network_profiles"
PROXY = "proxy"
VIRTUAL_DEVICES = "virtual_devices"
WEBVIEW = "webview"

CAPABILITY_METHODS: dict[str, frozenset[str]] = {
    APP_DATABASE: frozenset(
        {
            "backup_database",
            "database_schema",
            "execute_database",
            "list_backups",
            "list_databases",
            "query_database",
            "restore_database",
        }
    ),
    # An optional in-target agent: a platform-side process AUA can talk to instead of
    # polling from the host. Android backs it with an AccessibilityService APK; another
    # platform could back it with whatever its runtime offers, or not claim it at all.
    DEVICE_AGENT: frozenset(
        {
            "HelperUnavailableError",
            "disable",
            "enable",
            "install",
            "is_bound",
            "is_enabled",
            "is_installed",
            "open_channel",
            "release_uiautomation",
            "remove",
            "root_available",
            "rootable",
            "start_touch_capture",
            "status",
            "stop_touch_capture",
            "tree_to_xml",
            "uiautomation_held",
        }
    ),
    DEVELOPER_SETTINGS: frozenset(
        {"anim_off", "anim_restore", "crashes_set", "profile_ac", "profile_default", "read_state"}
    ),
    FEATURE_FLAGS: frozenset(
        {
            "build_uri",
            "dump_result",
            "load_flags_file",
            "parse_assignments",
            "read_context_flags",
            "read_prefs",
            "restore_prefs",
            "save_prefs_backup",
            "snapshot_prefs",
            "write_prefs",
        }
    ),
    MICROPHONE: frozenset(
        {
            "MicDeliveredReleaseError",
            "MicDeliveryUncertainError",
            "MicToggleStartUncertainError",
            "MicToggleStopUncertainError",
            "claim_injection_attempt",
            "inject_prepared",
            "inspect_pcm_wav",
            "prepare_injection",
            "synthesize_speech",
            "validate_control_mode",
        }
    ),
    NETWORK: frozenset(
        {
            "apply_offline_controls",
            "backup_path",
            "load_backup",
            "offline_verified",
            "read_network_state",
            "require_current_backup",
            "restore_controls",
            "restored_verified",
            "save_backup",
            "wait_for_state",
        }
    ),
    NETWORK_PROFILES: frozenset(
        {
            "PROFILE_NAMES",
            "apply_radio_profile",
            "load_profile",
            "normalize_profile",
            "prepare_loss",
            "profile_path",
            "profile_verified",
            "qdisc_evidence",
            "read_emulator_shape",
            "remove_loss",
            "require_current_profile",
            "restore_emulator_shape",
            "restore_radio_profile",
            "root_enabled",
            "safe_unroot_after_failed_apply",
            "save_profile",
            "set_emulator_shape",
            "set_loss",
            "shape_matches",
            "stale_profile",
            "wait_for_radio_profile",
        }
    ),
    # The ownership members are part of the contract, not an optional extra: the device's proxy
    # is a *device-global* setting pointing at a *non-persistent* host process, so a platform
    # claiming this capability must be able to say who owns it and whether that owner is dead.
    # Without them a parallel agent silently inherits a proxied device it cannot see or fix.
    PROXY: frozenset(
        {
            "backfill_rule_ids",
            "cassette_dir",
            "clear_record_window",
            "clear_rules",
            "clear_state",
            "diagnose_empty_recording",
            # Health-check trio: a platform claiming PROXY must be able to say — together, not
            # separately — whether the process is alive, the tunnel is reachable, and the
            # device setting points at it. See `proxy_mock.proxy_health`.
            "ensure_reverse_tunnel",
            "flow_matches",
            "guard_rule_scope",
            "install_system_ca",
            "load_cassette",
            "load_doc",
            "load_listen_port",
            "load_record",
            "load_record_window",
            "load_rules",
            "map_rule",
            "orphan_reason",
            # A device pointed at a proxy nobody owns is diagnosable *only* from the device's
            # own setting, so a platform claiming PROXY must be able to say what that setting
            # means — which host, which port, and whether the host is even on this machine.
            # Without it the generic layer cannot tell a black hole from a clean device.
            "parse_proxy_target",
            "proxy_health",
            "read_device_http_proxy",
            "read_flow_bodies",
            "read_flows_since",
            "read_state",
            "record_path",
            "reset_record",
            "reverse_tunnel_active",
            "rewrite_rule",
            "rules_path",
            "save_cassette",
            "save_doc",
            "save_record_window",
            "start_mitm",
            "stop_mitm",
            "tls_failures_in_log",
            "write_rules",
            "write_state",
        }
    ),
    VIRTUAL_DEVICES: frozenset(
        {
            # A platform that can boot throwaway targets must also be able to reclaim the ones
            # its own supervisor lost track of, or "aua started it" becomes "aua leaked it".
            "adopt_idle_watchdogs",
            "ensure_proxy_avd",
            "list_avds",
            "recommend_proxy_avd",
            "select_avd_for_session",
            "start",
            "status",
            "stop",
            # Rollback teardown scoped to one boot this process performed. A bare serial
            # can name a foreign device after a provisioning collision, so provisioning
            # rollbacks must be able to stop by owned instance/pid, never by serial.
            "stop_spawned_instance",
        }
    ),
    WEBVIEW: frozenset({"enrich", "should_try_webview"}),
}


def missing_members(capability: str, service: Any) -> list[str]:
    """Members absent from a service that claims *capability*."""

    return sorted(
        name for name in CAPABILITY_METHODS.get(capability, ()) if not hasattr(service, name)
    )
