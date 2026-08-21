"""The dashboard's URL has to be typeable, not merely reachable.

``http://192.168.8.240:48765/?token=zK4…`` is a URL you scan, never one you type. The
``--name`` flag publishes an unprivileged mDNS host record so the same page also answers
to ``http://aua.local/``. Four things have to hold for that to be worth having, and each
one is asserted below:

* the published name must be reachable, so it may only be offered when the server is
  actually bound to the network rather than to loopback;
* the URL must lose its port, so a named dashboard claims port 80 unless the caller pins
  one — and ``status``/``stop``/``qr`` must then still find it without being told;
* the first visit must land on the *name*, because the access token is exchanged for an
  origin-bound cookie and whichever origin spends the token is the one that keeps working;
* the record must die with the dashboard, or a stopped dashboard leaves a name on the
  network pointing at a closed port.

Nothing here needs a publisher binary or a network: the argv is built by a pure function
and the child process is injected.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser import bonjour
from android_ui_analyser import dashboard as dash
from android_ui_analyser.errors import UsageError


class _FakeProc:
    """Stands in for the publisher child, recording how it was asked to stop."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.terminated = False
        self.killed = False
        self._alive = True

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:  # pragma: no cover - only on a wedged publisher
        self.killed = True
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        return 0


# --------------------------------------------------------------------------- names


@pytest.mark.parametrize(
    "given, expected",
    [
        ("aua", "aua"),
        ("AUA", "aua"),
        ("  aua  ", "aua"),
        ("aua.local", "aua"),
        ("aua.local.", "aua"),
        ("my-aua-2", "my-aua-2"),
    ],
)
def test_hostname_accepts_the_forms_a_person_actually_types(given: str, expected: str) -> None:
    assert bonjour.normalise_hostname(given) == expected
    assert bonjour.hostname_for(given) == f"{expected}.local"


@pytest.mark.parametrize(
    "given",
    ["", "   ", "a b", "aua.box", "-aua", "aua-", "aua_box", "a" * 64, "aua/../etc"],
)
def test_hostname_rejects_anything_that_is_not_one_dns_label(given: str) -> None:
    with pytest.raises(UsageError):
        bonjour.normalise_hostname(given)


def test_named_url_drops_the_default_http_port_and_keeps_any_other() -> None:
    assert bonjour.hostname_url("aua", 80) == "http://aua.local/"
    assert bonjour.hostname_url("aua", 48765) == "http://aua.local:48765/"


# ---------------------------------------------------------------- publisher command


def test_macos_publishes_a_proxy_record_not_just_a_service_advert() -> None:
    argv = bonjour.publisher_command(
        hostname="aua",
        port=80,
        address="192.0.2.10",
        platform="darwin",
        which=lambda tool: f"/usr/bin/{tool}",
    )
    # ``-R`` would advertise a browsable service whose name a browser still cannot resolve.
    # Only ``-P`` publishes the A record that makes http://aua.local/ work.
    assert argv == [
        "/usr/bin/dns-sd", "-P", "aua", "_http._tcp", "local", "80", "aua.local", "192.0.2.10",
    ]


def test_linux_publishes_the_same_name_through_avahi() -> None:
    argv = bonjour.publisher_command(
        hostname="aua",
        port=80,
        address="192.0.2.10",
        platform="linux",
        which=lambda tool: f"/usr/bin/{tool}",
    )
    assert argv == ["/usr/bin/avahi-publish", "-a", "-R", "aua.local", "192.0.2.10"]


@pytest.mark.parametrize(
    "platform, which",
    [
        ("win32", lambda tool: f"/usr/bin/{tool}"),  # no publisher for this platform
        ("darwin", lambda tool: None),  # publisher not installed
        ("linux", lambda tool: None),
    ],
)
def test_a_host_without_a_publisher_is_a_soft_no_not_an_error(platform: str, which: Any) -> None:
    assert (
        bonjour.publisher_command(
            hostname="aua", port=80, address="192.0.2.10", platform=platform, which=which
        )
        is None
    )


def test_advertise_without_a_publisher_returns_none_so_the_dashboard_still_starts() -> None:
    assert bonjour.advertise(hostname="aua", port=80, address="192.0.2.10", command=[]) is None


# -------------------------------------------------------------------- advertisement


def test_advertise_hands_back_a_handle_that_retires_the_name() -> None:
    spawned: list[_FakeProc] = []

    def _spawn(argv: list[str]) -> Any:
        proc = _FakeProc(argv)
        spawned.append(proc)
        return proc

    advert = bonjour.advertise(
        hostname="aua",
        port=80,
        address="192.0.2.10",
        command=["/usr/bin/dns-sd", "-P", "aua"],
        spawn=_spawn,
        wait=False,
    )
    assert advert is not None
    assert advert.url == "http://aua.local/"
    assert advert.info()["hostname"] == "aua.local"
    assert advert.resolved is False

    advert.stop()
    assert spawned[0].terminated is True
    advert.stop()  # idempotent: stopping twice must not raise or re-signal
    assert spawned[0].killed is False


def test_a_publisher_that_dies_immediately_is_reported_unresolved() -> None:
    class _DeadProc(_FakeProc):
        def poll(self) -> int | None:
            return 1

    advert = bonjour.advertise(
        hostname="aua",
        port=80,
        address="192.0.2.10",
        command=["/usr/bin/dns-sd"],
        spawn=lambda argv: _DeadProc(argv),
        wait=True,
    )
    assert advert is not None and advert.resolved is False


def test_a_publisher_that_cannot_be_spawned_does_not_break_the_dashboard() -> None:
    def _boom(argv: list[str]) -> Any:
        raise OSError("no such tool")

    assert (
        bonjour.advertise(
            hostname="aua", port=80, address="192.0.2.10", command=["x"], spawn=_boom
        )
        is None
    )


# ----------------------------------------------------------------------- URL surface


def test_a_resolved_name_becomes_the_url_callers_open() -> None:
    urls = dash._service_urls(
        port=80, lan=True, access_token="secret", hostname="aua", name_resolved=True
    )
    assert urls["name"] == "aua.local"
    assert urls["name_url"] == "http://aua.local/"
    assert urls["name_access_url"] == "http://aua.local/?token=secret"
    # The token buys an origin-bound cookie. Opening 127.0.0.1 would spend it on the wrong
    # origin and leave http://aua.local/ demanding a token nobody has any more.
    assert urls["access_url"] == "http://aua.local/?token=secret"


def test_an_unresolved_name_is_reported_but_never_opened() -> None:
    urls = dash._service_urls(
        port=80, lan=True, access_token="secret", hostname="aua", name_resolved=False
    )
    assert urls["name_url"] == "http://aua.local/"
    assert urls["name_resolved"] is False
    assert urls["access_url"] == "http://127.0.0.1/?token=secret"


def test_port_80_disappears_from_every_url_not_just_the_named_one() -> None:
    # A dashboard on port 80 that still prints ":80" defeats the point of moving there.
    urls = dash._service_urls(port=80, lan=False, access_token=None)
    assert urls["url"] == "http://127.0.0.1/"
    assert dash._http_url("192.0.2.10", 80) == "http://192.0.2.10/"
    assert dash._http_url("192.0.2.10", 48765) == "http://192.0.2.10:48765/"


def test_an_unnamed_dashboard_keeps_exactly_its_old_url_surface() -> None:
    urls = dash._service_urls(port=48765, lan=False, access_token=None)
    assert urls == {
        "url": "http://127.0.0.1:48765/",
        "access_url": "http://127.0.0.1:48765/",
        "lan_urls": [],
        "lan_access_urls": [],
    }


# ---------------------------------------------------------------------------- ports


def test_a_named_dashboard_is_found_again_without_repeating_its_port(tmp_path: Path) -> None:
    # --name moves the dashboard to port 80. `aua dashboard stop` must not then look at
    # 48765, decide nothing is running, and leave the dashboard alive.
    assert dash._resolve_service_port(tmp_path, None) == dash.DEFAULT_DASHBOARD_PORT
    dash._write_service_state(tmp_path, {"service": "aua-dashboard-v1", "port": 80})
    assert dash._resolve_service_port(tmp_path, None) == 80
    assert dash._resolve_service_port(tmp_path, 48765) == 48765


@pytest.mark.parametrize("recorded", [0, -1, 70000, "80", None])
def test_a_corrupt_recorded_port_falls_back_to_the_default(tmp_path: Path, recorded: Any) -> None:
    dash._write_service_state(tmp_path, {"port": recorded})
    assert dash._resolve_service_port(tmp_path, None) == dash.DEFAULT_DASHBOARD_PORT


# -------------------------------------------------------------------- start contract


def _config(tmp_path: Path) -> Any:
    from android_ui_analyser.config import Config

    cfg = Config()
    cfg.cache.dir = str(tmp_path)
    return cfg


def test_a_name_without_network_access_is_refused_rather_than_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # aua.local resolves to this machine's LAN address, so a loopback-bound server would
    # answer the published name with a refused connection.
    monkeypatch.setattr(dash, "service_status", lambda *a, **k: {"running": False})
    with pytest.raises(UsageError, match="needs network access"):
        dash.start_service(_config(tmp_path), lan=False, hostname="aua")


def test_starting_under_a_different_name_is_refused_instead_of_silently_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    running = {
        "ok": True,
        "running": True,
        "status": "running",
        "lan": True,
        "name": "aua.local",
        "authenticated": True,
    }
    monkeypatch.setattr(dash, "service_status", lambda *a, **k: dict(running))
    assert (
        dash.start_service(_config(tmp_path), lan=True, hostname="aua")["status"]
        == "already_running"
    )
    with pytest.raises(UsageError, match="different hostname"):
        dash.start_service(_config(tmp_path), lan=True, hostname="other")
    with pytest.raises(UsageError, match="different hostname"):
        dash.start_service(_config(tmp_path), lan=True)


def test_a_named_start_claims_port_80_and_forwards_the_name_to_the_detached_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dash, "service_status", lambda *a, **k: {"running": False})
    monkeypatch.setattr(dash, "_dashboard_health", lambda port: {"pid": 4242, "name": "aua"})
    seen: dict[str, Any] = {}

    class _Proc:
        pid = 4242

        def poll(self) -> int | None:
            return None

    def _popen(cmd: list[str], **kwargs: Any) -> Any:
        seen["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(subprocess, "Popen", _popen)
    result = dash.start_service(_config(tmp_path), lan=True, hostname="AUA.local")

    assert result["port"] == 80
    assert "--hostname" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--hostname") + 1] == "aua"
    assert seen["cmd"][seen["cmd"].index("--bind") + 1] == "0.0.0.0"
    # The port the child was told to use and the port recorded for `stop` must agree.
    assert seen["cmd"][seen["cmd"].index("--port") + 1] == "80"
    assert dash._read_service_state(tmp_path)["port"] == 80
    assert dash._read_service_state(tmp_path)["hostname"] == "aua"


def test_an_explicit_port_still_wins_over_the_named_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dash, "service_status", lambda *a, **k: {"running": False})
    monkeypatch.setattr(dash, "_dashboard_health", lambda port: {"pid": 1, "name": "aua"})
    seen: dict[str, Any] = {}

    class _Proc:
        pid = 1

        def poll(self) -> int | None:
            return None

    monkeypatch.setattr(
        subprocess, "Popen", lambda cmd, **kw: (seen.__setitem__("cmd", cmd), _Proc())[1]
    )
    result = dash.start_service(_config(tmp_path), lan=True, hostname="aua", port=48765)
    assert result["port"] == 48765
    assert seen["cmd"][seen["cmd"].index("--port") + 1] == "48765"


def test_the_detached_child_accepts_the_hostname_flag() -> None:
    # A flag start_service passes but the child rejects would make every named start fail
    # with a bare argparse exit in the log file.
    import inspect

    source = inspect.getsource(dash._service_main)
    assert '"--hostname"' in source
    assert "hostname=args.hostname or None" in source


def test_the_serve_loop_retires_the_name_when_the_dashboard_is_stopped() -> None:
    import inspect

    stop_sources = inspect.getsource(dash.run) + inspect.getsource(dash.shutdown)
    # Both exit paths — Ctrl-C on the foreground server and shutdown() on a threaded one —
    # have to stop the publisher, or the name outlives the port it points at.
    assert stop_sources.count("advert.stop()") == 2


# ------------------------------------------------------------------------------ CLI


def test_the_cli_exposes_name_on_every_command_that_starts_a_dashboard() -> None:
    import inspect

    from android_ui_analyser import cli

    for command in (cli.dashboard_cmd, cli.dashboard_start_cmd, cli.dashboard_run_cmd):
        assert "hostname" in inspect.signature(command).parameters, command.__name__
        assert '"--name"' in inspect.getsource(command), command.__name__

    for command in (
        cli.dashboard_status_cmd,
        cli.dashboard_stop_cmd,
        cli.dashboard_open_cmd,
        cli.dashboard_qr_cmd,
    ):
        # None, not 48765: these have to follow a dashboard that --name moved to port 80.
        assert inspect.signature(command).parameters["port"].default.default is None
