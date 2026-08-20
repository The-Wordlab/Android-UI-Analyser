"""A `clear_data`-first setup flow must be runnable more than once.

`flow run` refuses to start unless the flow's owning app is already in the foreground — a
precondition that exists to stop a flow from being replayed against the wrong app. But
`clear_data` (the only way a flow can wipe app data and start fresh) always kills the app and
drops the device on the launcher: after any run of a flow shaped ``[clear_data, launch_app,
...]`` completes, the device is exactly where the *next* run's precondition will refuse to
start from, unless the device happened to still be sitting on the flow's own app already.

Before the fix, the "this flow already establishes its own foreground" bypass
(``Engine._flow_leading_launch_establishes_origin``) only recognized a *bare* ``launch_app`` as
the very first step. A setup flow's first step is unavoidably ``clear_data``, so the bypass
never applied and the precondition raised — every single time the device was not already on
the flow's own app, which past the first run is nearly always.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from conftest import FakeDevice, make_config
from test_memory import HOME, P, _hier, _node

LAUNCHER = "com.example.launcher"
LAUNCHER_SCREEN = _hier(
    _node("android.widget.TextView", text="Launcher", rid="launcher:id/title", pkg=LAUNCHER)
)


class _ClearDropsToLauncherDevice(FakeDevice):
    """Models real Android: `pm clear` leaves the device on the launcher, not the app."""

    def clear_app(self, package: str) -> str | None:
        self.calls.append(("clear_app", (package,)))
        self._pkg = LAUNCHER
        self._xml = LAUNCHER_SCREEN
        return None

    def launch_app(self, package: str, *, activity: str | None = None) -> None:
        super().launch_app(package, activity=activity)
        self._xml = HOME


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    return Engine(
        make_config(
            memory={"enabled": True, "dir": str(tmp_path / "memory")},
            cache={"dir": str(tmp_path / "cache")},
            daemon={"enabled": False},
        ),
        device=device,
    )


def _write_setup_flow(tmp_path: Path) -> Path:
    flow_path = tmp_path / "setup.yaml"
    flow_path.write_text(
        f"name: setup\napp: {P}\nsteps:\n  - clear_data\n  - launch_app: {P}\n",
        encoding="utf-8",
    )
    return flow_path


def test_a_clear_data_first_setup_flow_runs_twice_in_a_row(tmp_path: Path) -> None:
    device = _ClearDropsToLauncherDevice(
        hierarchy_xml=LAUNCHER_SCREEN, package=LAUNCHER, serial="setup-twice"
    )
    engine = _engine(tmp_path, device)
    flow_path = _write_setup_flow(tmp_path)

    for attempt in (1, 2):
        # Exactly the reported symptom: whatever the previous run left behind, the device is
        # sitting on the launcher (not the flow's own app) when the next run starts.
        device._pkg = LAUNCHER
        device._xml = LAUNCHER_SCREEN
        result = engine.flow_run(file=str(flow_path))
        assert result["ok"] is True, f"attempt {attempt} failed: {result}"
        # `launch_app` (the flow's second step) brought its own app back to the foreground.
        assert device._pkg == P, f"attempt {attempt} did not reach {P}"


def test_a_flow_that_still_needs_the_right_starting_app_is_unaffected(tmp_path: Path) -> None:
    """The precondition must still fire for a flow that does NOT establish its own origin."""
    device = FakeDevice(hierarchy_xml=LAUNCHER_SCREEN, package=LAUNCHER, serial="unrelated-flow")
    engine = _engine(tmp_path, device)
    flow_path = tmp_path / "no_setup.yaml"
    flow_path.write_text(f"name: no_setup\napp: {P}\nsteps:\n  - wait_stable\n", encoding="utf-8")

    try:
        engine.flow_run(file=str(flow_path))
    except UsageError as exc:
        assert "foreground package" in str(exc)
    else:
        raise AssertionError("expected the foreground precondition to still fire")
