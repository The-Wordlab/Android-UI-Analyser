"""Stable structural contracts for optional platform capability services.

The target runtime (`Device`) covers common screen/input/app operations. These named services
cover operations that are host-wide or not universal. A plugin may implement only the services
it supports, but claiming a service means implementing its complete structural surface below.
"""

from __future__ import annotations

from typing import Any

APP_DATABASE = "app_database"
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
    PROXY: frozenset(
        {
            "cassette_dir",
            "flow_matches",
            "install_system_ca",
            "load_cassette",
            "load_listen_port",
            "load_rules",
            "map_rule",
            "read_flows_since",
            "record_path",
            "rules_path",
            "save_cassette",
            "start_mitm",
            "stop_mitm",
            "tls_failures_in_log",
            "write_rules",
        }
    ),
    VIRTUAL_DEVICES: frozenset(
        {"ensure_proxy_avd", "list_avds", "recommend_proxy_avd", "start", "status", "stop"}
    ),
    WEBVIEW: frozenset({"enrich", "should_try_webview"}),
}


def missing_members(capability: str, service: Any) -> list[str]:
    """Members absent from a service that claims *capability*."""

    return sorted(
        name for name in CAPABILITY_METHODS.get(capability, ()) if not hasattr(service, name)
    )
