"""A clear_data step's settle-barrier warning must not abort the flow it belongs to.

``Device.clear_app`` returns a non-fatal warning (rather than raising) when the post-wipe
settle barrier could not be *proven* quiescent in time, even though the wipe itself already
happened and cannot be repeated (see ``device.py``). Silently discarding that return value
would make the warning invisible to the caller; raising instead would abort an otherwise-
successful flow mid-setup over a barrier that only failed to prove a negative. This proves the
actual wiring all the way to a flow result: the flow completes, and the warning is visible on
the ``clear_data`` step's own row rather than swallowed.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.engine import Engine
from conftest import FakeDevice, make_config

P = "com.example.app"
WARNING = (
    f"cleared app data for {P}, but its removed task did not settle; quiescence could not be "
    "proven within 12s. The wipe already happened and will not be repeated."
)


class _UnprovenClearDevice(FakeDevice):
    """Models ``Device.clear_app`` returning a warning instead of raising (see device.py)."""

    def clear_app(self, package: str) -> str | None:
        self.calls.append(("clear_app", (package,)))
        return WARNING


def _engine(tmp_path: Path) -> Engine:
    return Engine(
        make_config(
            memory={"enabled": True, "dir": str(tmp_path / "memory")},
            cache={"dir": str(tmp_path / "cache")},
            daemon={"enabled": False},
        ),
        device=_UnprovenClearDevice(package=P, serial="unproven-clear"),
    )


def test_a_flow_completes_despite_an_unproven_clear_barrier(tmp_path: Path) -> None:
    flow_path = tmp_path / "setup.yaml"
    flow_path.write_text(f"name: setup\napp: {P}\nsteps:\n  - clear_data\n", encoding="utf-8")

    result = _engine(tmp_path).flow_run(file=str(flow_path))

    assert result["ok"] is True
    [clear_step] = result["steps_run"]
    assert clear_step["warning"] == f"{P} — {WARNING}"


def test_the_standalone_app_clear_action_also_surfaces_the_warning(tmp_path: Path) -> None:
    """Same contract outside a flow: `aua app clear` must not swallow it either."""
    engine = _engine(tmp_path)

    result = engine.app("clear", package=P, confirmed=True, observe=False)

    assert result.ok is True
    assert result.detail is not None
    assert WARNING in result.detail
