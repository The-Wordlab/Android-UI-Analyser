"""Developer-option profiles — anim scales, crash/ANR dialogs, don't-keep-activities.

No UI: pure ``settings`` / ``settings put`` via the device shell. Prior values are saved
under the cache dir so ``aua dev profile default`` always restores.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

# settings namespace → key
ANIM_KEYS = (
    ("global", "window_animation_scale"),
    ("global", "transition_animation_scale"),
    ("global", "animator_duration_scale"),
)
CRASH_KEYS = (
    ("secure", "anr_show_background"),  # Show background ANRs
    ("global", "hide_error_dialogs"),  # 0 = always show crash dialog (inverted!)
)
DONT_KEEP_KEY = ("global", "always_finish_activities")

ShellFn = Callable[[str], str]


def _settings_get(shell: ShellFn, namespace: str, key: str) -> str:
    raw = shell(f"settings get {namespace} {key}").strip()
    if raw in ("null", "None", ""):
        return "1" if namespace == "global" and "animation" in key else "0"
    return raw


def _settings_put(shell: ShellFn, namespace: str, key: str, value: str) -> None:
    shell(f"settings put {namespace} {key} {value}")


def read_state(shell: ShellFn) -> dict[str, Any]:
    """Snapshot of the knobs we manage."""
    anim = {
        key: _settings_get(shell, ns, key) for ns, key in ANIM_KEYS
    }
    # hide_error_dialogs: 0 = show dialogs; expose as crashes_visible bool for agents.
    hide = _settings_get(shell, "global", "hide_error_dialogs")
    anr = _settings_get(shell, "secure", "anr_show_background")
    dont_keep = _settings_get(shell, *DONT_KEEP_KEY)
    return {
        "anim": anim,
        "crashes_visible": hide in ("0", "null", ""),
        "hide_error_dialogs": hide,
        "anr_show_background": anr,
        "dont_keep_activities": dont_keep in ("1", "true"),
        "always_finish_activities": dont_keep,
    }


def save_backup(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_backup(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def anim_off(shell: ShellFn, backup_path: Path) -> dict[str, Any]:
    state = read_state(shell)
    save_backup(backup_path, state)
    for ns, key in ANIM_KEYS:
        _settings_put(shell, ns, key, "0")
    return read_state(shell)


def anim_on(shell: ShellFn, backup_path: Path) -> dict[str, Any]:
    """Enable all animation scales while preserving the exact prior values."""
    state = read_state(shell)
    save_backup(backup_path, state)
    for ns, key in ANIM_KEYS:
        _settings_put(shell, ns, key, "1")
    return read_state(shell)


def anim_restore(shell: ShellFn, backup_path: Path) -> dict[str, Any]:
    backup = load_backup(backup_path)
    if backup is None:
        for ns, key in ANIM_KEYS:
            _settings_put(shell, ns, key, "1")
    else:
        anim = backup.get("anim") or {}
        for ns, key in ANIM_KEYS:
            _settings_put(shell, ns, key, str(anim.get(key, "1")))
    state = read_state(shell)
    backup_path.unlink(missing_ok=True)
    return state


def crashes_set(shell: ShellFn, enabled: bool, backup_path: Path) -> dict[str, Any]:
    state = read_state(shell)
    save_backup(backup_path, state)
    # hide_error_dialogs=0 → show; =1 → hide
    _settings_put(shell, "global", "hide_error_dialogs", "0" if enabled else "1")
    _settings_put(shell, "secure", "anr_show_background", "1" if enabled else "0")
    return read_state(shell)


def dont_keep_set(shell: ShellFn, enabled: bool, backup_path: Path) -> dict[str, Any]:
    state = read_state(shell)
    save_backup(backup_path, state)
    _settings_put(shell, *DONT_KEEP_KEY, "1" if enabled else "0")
    return read_state(shell)


def profile_ac(shell: ShellFn, backup_path: Path) -> dict[str, Any]:
    """AC-friendly: animations off, crash/ANR dialogs visible."""
    state = read_state(shell)
    save_backup(backup_path, state)
    for ns, key in ANIM_KEYS:
        _settings_put(shell, ns, key, "0")
    _settings_put(shell, "global", "hide_error_dialogs", "0")
    _settings_put(shell, "secure", "anr_show_background", "1")
    return read_state(shell)


def profile_default(shell: ShellFn, backup_path: Path) -> dict[str, Any]:
    """Restore everything from the backup file (or sane defaults)."""
    backup = load_backup(backup_path)
    if backup is None:
        for ns, key in ANIM_KEYS:
            _settings_put(shell, ns, key, "1")
        _settings_put(shell, "global", "hide_error_dialogs", "1")
        _settings_put(shell, "secure", "anr_show_background", "0")
        _settings_put(shell, *DONT_KEEP_KEY, "0")
    else:
        anim = backup.get("anim") or {}
        for ns, key in ANIM_KEYS:
            _settings_put(shell, ns, key, str(anim.get(key, "1")))
        _settings_put(
            shell,
            "global",
            "hide_error_dialogs",
            str(backup.get("hide_error_dialogs", "1")),
        )
        _settings_put(
            shell,
            "secure",
            "anr_show_background",
            str(backup.get("anr_show_background", "0")),
        )
        _settings_put(
            shell,
            *DONT_KEEP_KEY,
            str(backup.get("always_finish_activities", "0")),
        )
        backup_path.unlink(missing_ok=True)
    return read_state(shell)


__all__ = [
    "anim_off",
    "anim_restore",
    "crashes_set",
    "dont_keep_set",
    "load_backup",
    "profile_ac",
    "profile_default",
    "read_state",
    "save_backup",
]
