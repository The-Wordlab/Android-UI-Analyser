"""A screen mid-load and an idle screen are the same hierarchy; only the network tells them apart.

Measured on a real run: told nothing about the network, the navigator pressed a login button, was
handed back a screen that looked unchanged, and pressed it again. Told `POST /v1/auth/login` had
not answered, the same model on the same screen chose to wait instead.

The window is the load-bearing part. This app -- any chat app -- holds a streamed connection open
by design, and AUA's own manual says so ("Still not network idle: this app never is"). Reporting
every unanswered call would therefore report one on every screen forever, and a signal that is
always on is not a signal.
"""

from __future__ import annotations

import time
from typing import Any

from android_ui_analyser import engine_analyze
from conftest import make_engine


class _Journal:
    """Stands in for the proxy capability, holding whatever the addon would have written."""

    def __init__(self, entries: list[dict[str, Any]] | None = None, *, broken: bool = False) -> None:
        self.entries = entries or []
        self.broken = broken
        self.asked: list[float] = []

    def read_flows_in_flight(self, cache_dir: Any, since_ts: float, serial: Any = None) -> list[dict]:
        if self.broken:
            raise RuntimeError("proxy is not running")
        self.asked.append(since_ts)
        return [e for e in self.entries if float(e.get("ts") or 0) > since_ts]


def _engine(tmp_path: Any, journal: _Journal):  # noqa: ANN202
    engine = make_engine(cache={"dir": str(tmp_path / "cache")})
    real = engine.platform.capability

    def capability(name: str) -> Any:
        return journal if name == "proxy" else real(name)

    engine.platform.capability = capability  # type: ignore[method-assign]
    return engine


def _open(path: str, *, method: str = "POST", ago: float = 0.2) -> dict[str, Any]:
    return {"flow": 1, "ts": time.time() - ago, "method": method, "path": path, "open": True}


def test_a_call_that_has_not_answered_is_named_on_the_observation(tmp_path: Any) -> None:
    engine = _engine(tmp_path, _Journal([_open("/v1/auth/login")]))
    assert engine_analyze.network_in_flight(engine) == ["POST /v1/auth/login"]


def test_a_quiet_screen_says_nothing_at_all(tmp_path: Any) -> None:
    """`Meta` drops falsey values, so a healthy response must pay nothing for this field."""
    assert engine_analyze.network_in_flight(_engine(tmp_path, _Journal([]))) is None


def test_a_proxy_that_is_not_running_is_not_an_error(tmp_path: Any) -> None:
    """Far more runs have no proxy than have one; none of them may fail because of this."""
    assert engine_analyze.network_in_flight(_engine(tmp_path, _Journal(broken=True))) is None


def test_a_platform_with_no_proxy_capability_is_not_an_error(tmp_path: Any) -> None:
    engine = make_engine(cache={"dir": str(tmp_path / "cache")})
    assert engine_analyze.network_in_flight(engine) is None


def test_a_connection_held_open_since_before_this_screen_is_not_reported(tmp_path: Any) -> None:
    """The streamed chat connection is open on every screen; it is not what "loading" means."""
    journal = _Journal([_open("/v1/stream", method="GET", ago=600.0), _open("/v1/auth/login")])
    assert engine_analyze.network_in_flight(engine := _engine(tmp_path, journal)) == [
        "POST /v1/auth/login"
    ]
    assert engine is not None


def test_the_window_asked_for_is_recent_not_the_whole_run(tmp_path: Any) -> None:
    journal = _Journal([])
    before = time.time()
    engine_analyze.network_in_flight(_engine(tmp_path, journal))
    assert journal.asked and before - journal.asked[0] <= engine_analyze.IN_FLIGHT_WINDOW_S + 1


def test_the_calls_are_named_semantically_and_carry_no_numbers(tmp_path: Any) -> None:
    """jev-1.13 is documented to read opaque and numeric values worse than semantic ones."""
    journal = _Journal([_open("/v1/auth/login"), _open("/v1/profile", method="GET")])
    named = engine_analyze.network_in_flight(_engine(tmp_path, journal))
    assert named == ["POST /v1/auth/login", "GET /v1/profile"]
    assert all(isinstance(item, str) for item in named)
