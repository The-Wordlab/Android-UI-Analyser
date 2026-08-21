"""How ``aua dashboard start`` decides where to serve and whether to ask for a token.

The shipped default is deliberately the convenient one: publish ``aua`` over mDNS, bind
every interface, serve without a token, so the dashboard is a URL you type. That is a
strong default for a tool that drives a device and streams logcat, so the parts that keep
it honest are pinned here:

* the default is what you get with no flags, and it says out loud what it exposes;
* every part of it is overridable, and a typed flag always beats config — the flags are
  tri-state so ``--auth`` is distinguishable from a default that happens to agree;
* ``--local`` really means loopback: a *configured* name must not drag the bind back onto
  the network, while a name typed alongside ``--local`` is a contradiction and is reported;
* port 80 is a preference, not a requirement. macOS lets an ordinary user bind it and
  Linux does not, so an unbindable default port falls back instead of failing the start —
  but a port the caller pinned is never quietly moved.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import dashboard as dash
from android_ui_analyser.cli import _dashboard_access
from android_ui_analyser.config import Config
from android_ui_analyser.errors import UsageError


def _config(tmp_path: Path) -> Any:
    cfg = Config()
    cfg.cache.dir = str(tmp_path)
    return cfg


class _Proc:
    pid = 4242

    def poll(self) -> int | None:
        return None


def _start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **kwargs: Any
) -> tuple[dict[str, Any], list[str]]:
    """Run ``start_service`` with the detached child faked out, returning result and argv."""
    monkeypatch.setattr(dash, "service_status", lambda *a, **k: {"running": False})
    name = str(kwargs.get("hostname") or "")
    monkeypatch.setattr(
        dash,
        "_dashboard_health",
        lambda port: {"pid": 4242, "name": name, "name_resolved": bool(name)},
    )
    seen: dict[str, list[str]] = {}
    monkeypatch.setattr(
        subprocess, "Popen", lambda cmd, **kw: (seen.__setitem__("cmd", cmd), _Proc())[1]
    )
    result = dash.start_service(_config(tmp_path), **kwargs)
    return result, seen["cmd"]


# ------------------------------------------------------------------ shipped defaults


def test_the_shipped_default_is_a_typeable_unauthenticated_dashboard() -> None:
    defaults = Config().dashboard
    assert defaults.name == "aua"
    assert defaults.lan is True
    assert defaults.auth is False
    assert defaults.port is None


def test_no_flags_at_all_resolves_to_that_default() -> None:
    port, lan, name, auth = _dashboard_access(
        Config(), port=None, lan=None, hostname=None, auth=None
    )
    assert (port, lan, name, auth) == (None, True, "aua", False)


def test_an_unauthenticated_network_dashboard_always_says_what_it_exposes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Pin the bind decision: this test is about what an unauthenticated start *says*, and the
    # portless URL only holds when port 80 was actually claimed. Left to the real kernel it
    # asserts a fact about the host instead — it fails on Linux without CAP_NET_BIND_SERVICE,
    # and on any machine where something already holds :80, including another AUA dashboard.
    monkeypatch.setattr(dash, "_can_bind", lambda host, port: True)
    result, cmd = _start(monkeypatch, tmp_path, lan=True, hostname="aua", auth=False)
    assert result["authenticated"] is False
    assert "--no-auth" in cmd
    # A credential nothing checks must not be left lying in the state file.
    assert dash._read_service_state(tmp_path)["access_token"] == ""
    assert "unauthenticated" in result["warning"]
    assert result["access_url"] == "http://aua.local/"
    assert "?token=" not in " ".join(str(v) for v in result.values())


# ---------------------------------------------------------------------- overrides


@pytest.mark.parametrize(
    "flags, expected",
    [
        # (port, lan, hostname, auth) -> (port, lan, name, auth)
        ({"auth": True}, (None, True, "aua", True)),
        ({"lan": False}, (None, False, None, False)),
        ({"hostname": ""}, (None, True, None, False)),
        ({"hostname": "box"}, (None, True, "box", False)),
        ({"port": 48765}, (48765, True, "aua", False)),
        # An explicit --local plus an explicit --name is a contradiction, not a merge:
        # both survive so start_service can report it rather than silently picking one.
        ({"lan": False, "hostname": "aua"}, (None, False, "aua", False)),
    ],
)
def test_a_typed_flag_always_beats_the_configured_default(
    flags: dict[str, Any], expected: tuple[Any, ...]
) -> None:
    call = {"port": None, "lan": None, "hostname": None, "auth": None, **flags}
    assert _dashboard_access(Config(), **call) == expected


def test_config_can_restore_the_guarded_shape_without_any_flag() -> None:
    cfg = Config()
    cfg.dashboard.auth = True
    cfg.dashboard.lan = False
    cfg.dashboard.name = None
    assert _dashboard_access(cfg, port=None, lan=None, hostname=None, auth=None) == (
        None,
        False,
        None,
        True,
    )


def test_local_plus_a_typed_name_is_refused_rather_than_silently_resolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dash, "service_status", lambda *a, **k: {"running": False})
    with pytest.raises(UsageError, match="needs network access"):
        dash.start_service(_config(tmp_path), lan=False, hostname="aua")


def test_auth_is_meaningless_on_loopback_so_nothing_is_passed_or_warned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, cmd = _start(monkeypatch, tmp_path, lan=False, auth=False)
    assert result["authenticated"] is False
    assert "--no-auth" not in cmd
    assert "warning" not in result


# --------------------------------------------------------------------- port choice


def test_an_unbindable_default_port_falls_back_instead_of_failing_the_start(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Linux refuses 0.0.0.0:80 without CAP_NET_BIND_SERVICE. The default must degrade to a
    # port with a number in the URL, never to a dashboard that will not start.
    monkeypatch.setattr(dash, "_can_bind", lambda host, port: False)
    result, cmd = _start(monkeypatch, tmp_path, lan=True, hostname="aua")
    assert result["port"] == dash.DEFAULT_DASHBOARD_PORT
    assert cmd[cmd.index("--port") + 1] == str(dash.DEFAULT_DASHBOARD_PORT)
    assert "could not be bound" in result["port_fallback"]


def test_a_bindable_default_port_is_used_and_reported_without_noise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dash, "_can_bind", lambda host, port: True)
    result, cmd = _start(monkeypatch, tmp_path, lan=True, hostname="aua")
    assert result["port"] == 80
    assert "port_fallback" not in result


def test_a_pinned_port_is_never_quietly_moved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Falling back from a port the caller chose would invalidate their bookmark silently.
    monkeypatch.setattr(dash, "_can_bind", lambda host, port: False)
    result, _ = _start(monkeypatch, tmp_path, lan=True, hostname="aua", port=8080)
    assert result["port"] == 8080
    assert "port_fallback" not in result


def test_can_bind_reports_a_real_kernel_answer() -> None:
    # A port already held by this test cannot be claimed twice; an ephemeral one can.
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = int(held.getsockname()[1])
        assert dash._can_bind("127.0.0.1", taken) is False
    assert dash._can_bind("127.0.0.1", 0) is True


# --------------------------------------------------------------------------- reuse


@pytest.mark.parametrize(
    "running_auth, requested",
    [(True, {"lan": True, "auth": False}), (False, {"lan": True, "auth": True})],
)
def test_a_running_dashboard_is_never_reused_under_a_different_auth_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, running_auth: bool, requested: dict[str, Any]
) -> None:
    monkeypatch.setattr(
        dash,
        "service_status",
        lambda *a, **k: {
            "ok": True,
            "running": True,
            "status": "running",
            "lan": True,
            "name": "",
            "authenticated": running_auth,
        },
    )
    with pytest.raises(UsageError, match="different authentication setting"):
        dash.start_service(_config(tmp_path), **requested)


# ----------------------------------------------------------------------- plumbing


def test_the_detached_child_honours_no_auth_only_on_a_network_bind() -> None:
    source = inspect.getsource(dash._service_main)
    assert '"--no-auth"' in source
    # Order matters: the bind check stays, so --no-auth can never *add* exposure.
    assert 'require_auth=args.bind == "0.0.0.0" and not args.no_auth' in source


def test_the_handler_skips_every_check_only_when_auth_is_off() -> None:
    source = inspect.getsource(dash._make_handler)
    assert "if not state.require_auth:\n                return True" in source


def test_every_start_flag_is_tri_state_so_config_can_be_overridden_both_ways() -> None:
    from android_ui_analyser import cli

    for command in (cli.dashboard_cmd, cli.dashboard_start_cmd, cli.dashboard_run_cmd):
        params = inspect.signature(command).parameters
        for flag in ("auth", "lan", "hostname", "port"):
            assert params[flag].default.default is None, f"{command.__name__}.{flag}"
        source = inspect.getsource(command)
        assert '"--auth/--no-auth"' in source, command.__name__
        assert '"--lan/--local"' in source, command.__name__
