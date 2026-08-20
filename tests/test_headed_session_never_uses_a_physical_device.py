"""A headed QA session means a visible emulator window, never a USB handset.

Measured 2026-08-20: ``session start --start-emulator --headed --fresh`` considered a physical
phone capable of ``headed`` work and therefore uninstalled/reinstalled the QA app there instead
of opening the requested emulator window.  A stricter retry with ``--needs emulator`` then
booted an emulator but rejected it because the runtime probe did not publish that capability.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from android_ui_analyser.platforms import android


def test_physical_target_never_satisfies_emulator_window_requirements() -> None:
    assert android._runtime_emulator_capabilities("R5CWC1QRVGR") == {  # noqa: SLF001
        "emulator": False,
        "headed": False,
        "audio": False,
    }


def test_headed_emulator_explicitly_satisfies_emulator_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        android.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="/sdk/emulator/qemu-system -avd Medium_Phone -port 5558\n"
        ),
    )

    assert android._runtime_emulator_capabilities("emulator-5558") == {  # noqa: SLF001
        "emulator": True,
        "headed": True,
        "audio": True,
    }


def test_headless_emulator_is_still_an_emulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        android.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=(
                "/sdk/emulator/qemu-system -avd Medium_Phone -port 5558 "
                "-no-window -no-audio\n"
            )
        ),
    )

    assert android._runtime_emulator_capabilities("emulator-5558") == {  # noqa: SLF001
        "emulator": True,
        "headed": False,
        "audio": False,
    }
