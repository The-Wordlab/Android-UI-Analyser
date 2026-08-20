"""Architecture guard: a new persistent device change must register how to undo it.

The point is not this list. The point is that the *next* device-facing feature cannot leak
silently. A mutation whose undo lives only in the mutating process is lost the moment that
process is killed, and the person who finds out is the next agent, on a device that reports
"Offline" or 401s every login for reasons that have nothing to do with the app under test.

So: touching persistent device state means adding an entry to
``device_ledger.MUTATION_CATALOGUE`` (with an undo op, or an explicit reason it needs none) and
recording an entry from the code that performs it. This test fails until you do.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from android_ui_analyser import device_ledger

ROOT = Path(__file__).parents[1] / "src" / "android_ui_analyser"

# Where native device mutations are allowed to live at all (the platform boundary keeps them
# out of the generic layers; this guard asks a different question about the same files).
BACKENDS = (
    "device.py",
    "device_agent.py",
    "devopts.py",
    "emulator.py",
    "flags.py",
    "mic.py",
    "network.py",
    "network_profiles.py",
    "proxy_mock.py",
    "app_database.py",
    "platforms/android.py",
)

# Commands that change state the device keeps after the command returns.
MUTATING = re.compile(
    r"settings\s+put\b"
    r"|settings\s+delete\b"
    r"|svc\s+(?:wifi|data|power|bluetooth|nfc)\b"
    r"|\bsetprop\b"
    r"|pm\s+(?:grant|revoke|disable|enable|clear)\b"
    r"|cmd\s+netpolicy\b"
    r"|am\s+set-debug-app\b"
    r"|dumpsys\s+deviceidle\s+(?:force-idle|unforce|step)\b"
)

# Functions that exist to *reverse* a mutation, or to read one back. They touch the same
# commands, so the scanner sees them; they are the cure, not the disease.
UNDO_OR_READ_SITES = frozenset(
    {
        "device.py:adb_reverse_remove",
        "device.py:remove_reverse_port",
        "device_agent.py:disable",
        "device_agent.py:remove",
        "devopts.py:anim_restore",
        "devopts.py:read_state",
        "network.py:restore_controls",
        "network.py:read_network_state",
        "network_profiles.py:restore_radio_profile",
        "network_profiles.py:restore_emulator_shape",
        "network_profiles.py:remove_loss",
        "network_profiles.py:safe_unroot_after_failed_apply",
    }
)

# Mutations confined to a device AUA itself created and will destroy, or to a scratch overlay
# that a reboot discards. Listed with a reason so "considered and safe" is distinguishable from
# "forgotten" — which is the whole failure mode this file exists to prevent.
NEEDS_NO_UNDO = {
    "emulator.py": "an emulator AUA booted is stopped by AUA; its whole disk goes with it",
    "flags.py": "app-private prefs inside the app under test, reset by `app clear`",
    "mic.py": "audio injection is a transient stream, not a stored setting",
}


def _sites(path: Path, *, skip_lines: set[int]) -> set[str]:
    """``<module>:<function>`` for every mutating command literal in *path*.

    Docstring lines are skipped: a docstring saying "wipes app data (``pm clear``)" is
    documentation. Counting it would train the next author to delete the explanation rather
    than register the mutation, which is exactly backwards.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(line, node.name)
    name = path.name if path.parent == ROOT else f"{path.parent.name}/{path.name}"
    found: set[str] = set()
    for node in ast.walk(tree):
        # Only string literals that are actually built into a command, never comments or
        # docstrings — a docstring mentioning `settings put` is documentation, not a mutation.
        if getattr(node, "lineno", None) in skip_lines:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if MUTATING.search(node.value):
                found.add(f"{name}:{owner.get(node.lineno, '<module>')}")
        elif isinstance(node, ast.JoinedStr):
            literal = "".join(
                part.value
                for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            if MUTATING.search(literal):
                found.add(f"{name}:{owner.get(node.lineno, '<module>')}")
    return found


def _docstring_lines(path: Path) -> set[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


def test_every_mutating_call_site_is_registered_with_an_undo() -> None:
    registered = {m.site for m in device_ledger.MUTATION_CATALOGUE.values()}
    unregistered: list[str] = []
    for relative in BACKENDS:
        path = ROOT / relative
        if not path.is_file():  # pragma: no cover — the file list moved
            continue
        for site in _sites(path, skip_lines=_docstring_lines(path)):
            module = site.split(":", 1)[0]
            if module in NEEDS_NO_UNDO:
                continue
            if site in UNDO_OR_READ_SITES or site in registered:
                continue
            unregistered.append(site)
    assert not sorted(unregistered), (
        "these call sites change persistent device state with no registered undo:\n  "
        + "\n  ".join(sorted(unregistered))
        + "\n\nRegister each in device_ledger.MUTATION_CATALOGUE (kind, site, undo_op) and call "
        "Engine.record_device_change *before* performing it — otherwise the change survives "
        "the agent that made it and the next agent inherits a dirty device. If it genuinely "
        "needs no undo, add it to the catalogue with undo_op=None and say why, or list it in "
        "this test's UNDO_OR_READ_SITES / NEEDS_NO_UNDO with a reason."
    )


def test_every_catalogued_undo_op_actually_exists() -> None:
    """A catalogue naming a handler that was renamed away is worse than no catalogue."""
    assert device_ledger.catalogue_gaps() == []


def test_every_undo_op_is_reachable_from_the_catalogue() -> None:
    """An op nothing records is dead code pretending to be a safety net."""
    used = {m.undo_op for m in device_ledger.MUTATION_CATALOGUE.values() if m.undo_op}
    orphans = sorted(set(device_ledger.UNDO_OPS) - used)
    assert not orphans, (
        f"undo ops no catalogued mutation uses: {orphans}. Either register the mutation they "
        "reverse, or delete them — an unreachable undo reads as coverage it does not provide."
    )


def test_recording_an_unknown_op_is_refused_loudly() -> None:
    """A typo'd op must fail at record time, not at reap time on a dirty device."""
    import pytest

    with pytest.raises(ValueError, match="unknown undo op"):
        device_ledger.record(
            "emulator-0000", key="k", kind="k", op="restore_the_vibes_somehow"
        )


def test_every_registered_undo_declares_whether_it_needs_the_target() -> None:
    """The reaper uses this to clean up host residue for a device that is unplugged."""
    for name, op in device_ledger.UNDO_OPS.items():
        assert isinstance(op.needs_device, bool), name
        assert op.summary, f"{name} has no human-readable summary for `aua teardown status`"
