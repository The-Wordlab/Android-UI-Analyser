"""A ``stop_app``-first flow must be runnable from wherever the device is sitting.

``flow run`` refuses to start unless the flow's own app is already in the foreground. The
bypass for a flow that establishes its own origin
(``Engine._flow_leading_launch_establishes_origin``) recognised a leading ``clear_data`` run
followed by ``launch_app``, and its stated reason is that ``clear_data`` "always kills the app
and drops the device on the launcher — so *by design* the very first run of such a setup flow
leaves nothing in the foreground for a *second* run to match".

That reasoning holds verbatim for ``stop_app`` on the flow's own package, which was simply
missing from the list. Found 2026-09-01 by a QA runner whose flow was refused outright from a
cold launcher; **27 of 54 committed derived flows in that suite open with ``stop_app``**, so
they passed whenever a previous scenario happened to leave the app in the foreground and were
refused whenever a lane started cold. Naming the package explicitly did not help, because the
loop returned 0 before ever reaching the ``launch_app`` behind it.
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


class _StopDropsToLauncherDevice(FakeDevice):
    """Models real Android: `am force-stop` leaves the device on the launcher, not the app."""

    def stop_app(self, package: str) -> None:
        self.calls.append(("stop_app", (package,)))
        self._pkg = LAUNCHER
        self._xml = LAUNCHER_SCREEN

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


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_a_bare_stop_app_first_flow_runs_from_the_launcher(tmp_path: Path) -> None:
    """The regression itself, in the shape the suite's flows actually use."""

    device = _StopDropsToLauncherDevice(
        hierarchy_xml=LAUNCHER_SCREEN, package=LAUNCHER, serial="stop-first"
    )
    engine = _engine(tmp_path, device)
    flow_path = _write(tmp_path, "cold", f"name: cold\napp: {P}\nsteps:\n  - stop_app\n  - launch_app: {P}\n")

    result = engine.flow_run(file=str(flow_path))

    assert result["ok"] is True, result
    assert device._pkg == P


def test_naming_the_package_explicitly_also_runs(tmp_path: Path) -> None:
    """`stop_app: <pkg>` for the flow's own app is the same claim, spelled out."""

    device = _StopDropsToLauncherDevice(
        hierarchy_xml=LAUNCHER_SCREEN, package=LAUNCHER, serial="stop-named"
    )
    engine = _engine(tmp_path, device)
    flow_path = _write(
        tmp_path, "named", f"name: named\napp: {P}\nsteps:\n  - stop_app: {P}\n  - launch_app: {P}\n"
    )

    assert engine.flow_run(file=str(flow_path))["ok"] is True


def test_stop_then_clear_then_launch_still_runs(tmp_path: Path) -> None:
    """The two kinds mix: a leading run of either, ending in the flow's own launch."""

    device = _StopDropsToLauncherDevice(
        hierarchy_xml=LAUNCHER_SCREEN, package=LAUNCHER, serial="stop-clear"
    )
    engine = _engine(tmp_path, device)
    flow_path = _write(
        tmp_path,
        "both",
        f"name: both\napp: {P}\nsteps:\n  - stop_app\n  - clear_data\n  - launch_app: {P}\n",
    )

    assert engine.flow_run(file=str(flow_path))["ok"] is True


def test_stopping_another_app_still_has_to_prove_the_precondition(tmp_path: Path) -> None:
    """The bypass is about the flow's OWN app. Stopping a stranger proves nothing."""

    device = _StopDropsToLauncherDevice(
        hierarchy_xml=LAUNCHER_SCREEN, package=LAUNCHER, serial="stop-foreign"
    )
    engine = _engine(tmp_path, device)
    flow_path = _write(
        tmp_path,
        "foreign",
        f"name: foreign\napp: {P}\nsteps:\n  - stop_app: com.example.other\n  - launch_app: {P}\n",
    )

    try:
        engine.flow_run(file=str(flow_path))
    except UsageError as exc:
        assert "foreground package" in str(exc)
    else:
        raise AssertionError("expected the foreground precondition to still fire")


def test_a_stop_app_with_no_launch_behind_it_still_refuses(tmp_path: Path) -> None:
    """Killing the app is not a route to it. Only a leading run ending in `launch_app` counts."""

    device = _StopDropsToLauncherDevice(
        hierarchy_xml=LAUNCHER_SCREEN, package=LAUNCHER, serial="stop-only"
    )
    engine = _engine(tmp_path, device)
    flow_path = _write(
        tmp_path, "stoponly", f"name: stoponly\napp: {P}\nsteps:\n  - stop_app\n  - wait_stable\n"
    )

    try:
        engine.flow_run(file=str(flow_path))
    except UsageError as exc:
        assert "foreground package" in str(exc)
    else:
        raise AssertionError("expected the foreground precondition to still fire")
