"""A goal could start behind another run's proxy and nothing said so.

``set_http_proxy`` is a *device* setting. It survives the process that set it, it survives the
app being force-stopped, and it survives a reinstall — so "restart the app at the start of a
goal" does not clear it, and never could. The ledger already records it with an undo, and the
watchdog reaps it once no holder is left; but a session starting while a stale or foreign entry
is still live inherits it silently, and every observation after that is taken through somebody
else's network.

``teardown_status`` already knew. Nothing called it on the path an agent actually takes. So
``session start`` now names the device changes it did not make, because a proxy you know about
is a variable and a proxy you don't is a mystery bug.

Changes this session made itself are not reported: it will undo them in ``session finish``, and
warning about your own bookkeeping is noise.
"""

from __future__ import annotations

from android_ui_analyser.session import inherited_device_state_warning

_SERIAL = "emulator-5554"
_ME = "claude-1:100"
_SOMEONE_ELSE = "codex-4782-5:142026"


def _row(serial: str, *changes: dict[str, object]) -> dict[str, object]:
    return {"serial": serial, "reapable": False, "why": "held", "changes": list(changes)}


def _change(kind: str, owner: str) -> dict[str, object]:
    return {"kind": kind, "key": f"{kind}:1", "detail": "127.0.0.1:8080", "owner": owner}


def test_a_clean_device_says_nothing() -> None:
    assert inherited_device_state_warning([], _SERIAL, _ME) is None


def test_a_foreign_proxy_on_this_device_is_named() -> None:
    rows = [_row(_SERIAL, _change("set_http_proxy", _SOMEONE_ELSE))]
    warning = inherited_device_state_warning(rows, _SERIAL, _ME)
    assert warning is not None
    assert "set_http_proxy" in warning
    # The whole point: say the thing an app restart cannot fix, so nobody tries that instead.
    assert "restart" in warning.lower()


def test_changes_this_session_made_are_not_reported_back_to_it() -> None:
    rows = [_row(_SERIAL, _change("set_http_proxy", _ME))]
    assert inherited_device_state_warning(rows, _SERIAL, _ME) is None


def test_another_devices_leftovers_are_not_our_problem() -> None:
    rows = [_row("emulator-5556", _change("set_http_proxy", _SOMEONE_ELSE))]
    assert inherited_device_state_warning(rows, _SERIAL, _ME) is None


def test_every_inherited_kind_is_listed_once() -> None:
    rows = [
        _row(
            _SERIAL,
            _change("set_http_proxy", _SOMEONE_ELSE),
            _change("set_clock", _SOMEONE_ELSE),
            _change("set_http_proxy", None),
        )
    ]
    warning = inherited_device_state_warning(rows, _SERIAL, _ME)
    assert warning is not None
    assert warning.count("set_http_proxy") == 1
    assert "set_clock" in warning


def test_an_unowned_leftover_counts_as_inherited() -> None:
    # A crashed run leaves no owner. That is exactly the state nobody is coming back to clean.
    rows = [_row(_SERIAL, _change("set_http_proxy", None))]
    assert inherited_device_state_warning(rows, _SERIAL, _ME) is not None


def test_the_warning_names_how_to_clear_it() -> None:
    rows = [_row(_SERIAL, _change("set_http_proxy", _SOMEONE_ELSE))]
    warning = inherited_device_state_warning(rows, _SERIAL, _ME) or ""
    assert "aua teardown" in warning


# --------------------------------------------------------------- wired into session start


def test_session_start_reports_a_foreign_proxy(tmp_path, monkeypatch) -> None:
    from android_ui_analyser import device_ledger
    from android_ui_analyser.engine import Engine
    from android_ui_analyser.providers.registry import ProviderFactory
    from conftest import FakeDevice, make_config

    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    device = FakeDevice(package="com.example.app")
    engine = Engine(cfg, device=device, factory=ProviderFactory(cfg))

    # Patch the module global, not the instance: `teardown_status` imports `device_ledger`
    # inside the call, so the instance attribute is never consulted.
    monkeypatch.setattr(
        device_ledger,
        "status",
        lambda **_kwargs: [
            _row(device.serial, _change("set_http_proxy", _SOMEONE_ELSE)),
        ],
    )
    out = engine.session_start("open the catalog")
    assert any("set_http_proxy" in warning for warning in out.get("warnings") or []), (
        f"no inherited-state warning in {out.get('warnings')}"
    )
