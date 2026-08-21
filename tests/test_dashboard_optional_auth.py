"""Token authentication on the LAN dashboard is the default, and opting out is explicit.

The dashboard is not a status page: it drives the device, streams logcat, and queries app
databases. Bound to ``0.0.0.0`` it therefore demands a token by default. ``--no-auth``
removes that for a network the operator vouches for, and these tests pin the parts that
make the opt-out safe to have at all:

* it is never implied — ``--lan`` and ``--name`` still mint a token on their own;
* it never applies silently — an unauthenticated LAN start returns a warning;
* it never half-applies — no token is minted, the child is actually told, and a running
  dashboard refuses to be reused under a different setting;
* it changes nothing for a loopback dashboard, which never had a token to begin with.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import dashboard as dash
from android_ui_analyser.errors import UsageError


def _config(tmp_path: Path) -> Any:
    from android_ui_analyser.config import Config

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


def test_a_lan_dashboard_still_mints_a_token_unless_you_say_otherwise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, cmd = _start(monkeypatch, tmp_path, lan=True)
    assert result["authenticated"] is True
    assert "--no-auth" not in cmd
    assert dash._read_service_state(tmp_path)["access_token"]
    assert "warning" not in result


def test_a_named_dashboard_does_not_quietly_drop_authentication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # --name is a convenience flag; it must not also be a security flag.
    result, cmd = _start(monkeypatch, tmp_path, lan=True, hostname="aua")
    assert result["authenticated"] is True
    assert "--no-auth" not in cmd


def test_no_auth_mints_no_token_tells_the_child_and_says_so_out_loud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, cmd = _start(monkeypatch, tmp_path, lan=True, hostname="aua", auth=False)

    assert result["authenticated"] is False
    assert "--no-auth" in cmd
    # A token left lying in the state file would be a credential nothing checks.
    assert dash._read_service_state(tmp_path)["access_token"] == ""
    assert dash._read_service_state(tmp_path)["auth"] is False
    assert "unauthenticated" in result["warning"]
    # Without a token there is nothing to append, so the URL is the plain one.
    assert result["access_url"] == "http://aua.local/"
    assert "?token=" not in " ".join(str(v) for v in result.values())


def test_no_auth_on_a_loopback_dashboard_changes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 127.0.0.1 never had a token, so there is no warning to give and nothing to tell the
    # child: the flag would only be noise in the argv.
    result, cmd = _start(monkeypatch, tmp_path, lan=False, auth=False)
    assert result["authenticated"] is False
    assert "--no-auth" not in cmd
    assert "warning" not in result


@pytest.mark.parametrize(
    "running_auth, requested",
    [(True, {"lan": True, "auth": False}), (False, {"lan": True, "auth": True})],
)
def test_a_running_dashboard_is_never_reused_under_a_different_auth_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, running_auth: bool, requested: dict[str, Any]
) -> None:
    # Returning "already_running" here would report a scope the caller did not ask for.
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


def test_the_detached_child_honours_no_auth_only_on_a_network_bind() -> None:
    source = inspect.getsource(dash._service_main)
    assert '"--no-auth"' in source
    # Order matters: the bind check stays, so --no-auth can never *add* exposure.
    assert 'require_auth=args.bind == "0.0.0.0" and not args.no_auth' in source


def test_the_handler_skips_every_check_only_when_auth_is_off() -> None:
    source = inspect.getsource(dash._make_handler)
    assert "if not state.require_auth:\n                return True" in source


def test_the_cli_offers_the_opt_out_on_every_command_that_starts_a_dashboard() -> None:
    from android_ui_analyser import cli

    for command in (cli.dashboard_cmd, cli.dashboard_start_cmd, cli.dashboard_run_cmd):
        assert "auth" in inspect.signature(command).parameters, command.__name__
        source = inspect.getsource(command)
        assert '"--auth/--no-auth"' in source, command.__name__
        # Authenticated has to be what you get by not thinking about it.
        assert inspect.signature(command).parameters["auth"].default.default is True
