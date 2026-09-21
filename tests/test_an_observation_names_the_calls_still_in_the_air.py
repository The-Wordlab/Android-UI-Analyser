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
        # The real reader pairs starts with completions; a fake that skipped that would report
        # every answered call as still waiting and quietly agree with a broken implementation.
        from android_ui_analyser.proxy_mock import flows_in_flight

        return flows_in_flight([e for e in self.entries if float(e.get("ts") or 0) > since_ts])

    def read_flows_since(self, cache_dir: Any, since_ts: float, serial: Any = None) -> list[dict]:
        if self.broken:
            raise RuntimeError("proxy is not running")
        self.asked.append(since_ts)
        return [e for e in self.entries
                if not e.get("open") and float(e.get("ts") or 0) > since_ts]


def _engine(tmp_path: Any, journal: _Journal, *, app_hosts: list[str] | None = None):  # noqa: ANN202
    engine = make_engine(cache={"dir": str(tmp_path / "cache")},
                         network={"app_hosts": app_hosts or []})
    real = engine.platform.capability

    def capability(name: str) -> Any:
        return journal if name == "proxy" else real(name)

    engine.platform.capability = capability  # type: ignore[method-assign]
    return engine


_SEQ = iter(range(1, 10_000))


def _open(path: str, *, method: str = "POST", ago: float = 0.2,
          host: str = "api.example.test") -> dict[str, Any]:
    return {"flow": next(_SEQ), "ts": time.time() - ago, "method": method, "path": path,
            "host": host, "open": True}


def _answered(path: str, *, status: int = 200, method: str = "POST", ago: float = 0.2,
              host: str = "api.example.test") -> dict[str, Any]:
    """A call that started and came back -- what the journal holds most of the time."""
    flow = next(_SEQ)
    return [{"flow": flow, "ts": time.time() - ago, "method": method, "path": path,
             "host": host, "open": True},
            {"flow": flow, "ts": time.time() - ago + 0.05, "method": method, "path": path,
             "host": host, "status": status}]


def test_a_call_that_has_not_answered_is_named_on_the_observation(tmp_path: Any) -> None:
    engine = _engine(tmp_path, _Journal([_open("/v1/auth/login")]), app_hosts=["example.test"])
    assert engine_analyze.network_calls(engine) == ["POST /v1/auth/login -> no answer yet"]


def test_a_quiet_screen_says_nothing_at_all(tmp_path: Any) -> None:
    """`Meta` drops falsey values, so a healthy response must pay nothing for this field."""
    assert engine_analyze.network_calls(_engine(tmp_path, _Journal([]))) is None


def test_a_proxy_that_is_not_running_is_not_an_error(tmp_path: Any) -> None:
    """Far more runs have no proxy than have one; none of them may fail because of this."""
    assert engine_analyze.network_calls(_engine(tmp_path, _Journal(broken=True))) is None


def test_a_platform_with_no_proxy_capability_is_not_an_error(tmp_path: Any) -> None:
    engine = make_engine(cache={"dir": str(tmp_path / "cache")})
    assert engine_analyze.network_calls(engine) is None


def test_a_connection_held_open_since_before_this_screen_is_not_reported(tmp_path: Any) -> None:
    """The streamed chat connection is open on every screen; it is not what "loading" means."""
    journal = _Journal([_open("/v1/stream", method="GET", ago=600.0), _open("/v1/auth/login")])
    engine = _engine(tmp_path, journal, app_hosts=["example.test"])
    assert engine_analyze.network_calls(engine) == [
        "POST /v1/auth/login -> no answer yet"
    ]
    assert engine is not None


def test_the_window_asked_for_is_recent_not_the_whole_run(tmp_path: Any) -> None:
    journal = _Journal([])
    before = time.time()
    engine_analyze.network_calls(_engine(tmp_path, journal, app_hosts=["theapp.test"]))
    assert journal.asked and before - journal.asked[0] <= engine_analyze.NETWORK_WINDOW_S + 1


def test_the_journal_is_not_even_read_when_no_backend_is_named(tmp_path: Any) -> None:
    """The default costs nothing: no file is opened on any of the runs that never configure this."""
    journal = _Journal([_open("/v1/chat")])
    assert engine_analyze.network_calls(_engine(tmp_path, journal)) is None
    assert journal.asked == []


def test_the_calls_are_named_semantically_and_carry_no_numbers(tmp_path: Any) -> None:
    """jev-1.13 is documented to read opaque and numeric values worse than semantic ones."""
    journal = _Journal([_open("/v1/auth/login"), _open("/v1/profile", method="GET")])
    named = engine_analyze.network_calls(_engine(tmp_path, journal, app_hosts=["example.test"]))
    assert named == ["POST /v1/auth/login -> no answer yet",
                     "GET /v1/profile -> no answer yet"]
    assert all(isinstance(item, str) for item in named)


# ------------------------------------------------- only the app's own backend is worth hearing

def test_nothing_is_reported_until_the_caller_names_their_backend(tmp_path: Any) -> None:
    """Measured on a real app: every call the proxy caught was a third-party SDK.

    Google push registration, a Firebase remote-config stream, RevenueCat, Facebook's SDK, a
    `POST /a1` analytics beacon. None of them hold a screen up. Replayed through the model they
    never produced a false `wait`, but they cost confidence on every screen that carried one --
    1.00 to 0.74 and 0.93 to 0.69, both of which fall below the gate, so two ready screens would
    have been handed back to the expensive model for nothing. Silence is the honest default.
    """
    journal = _Journal([_open("/v1/firelog/legacy/batchlog", host="crashlyticsreports-pa.googleapis.com")])
    assert engine_analyze.network_calls(_engine(tmp_path, journal)) is None


def test_only_the_named_backend_is_reported(tmp_path: Any) -> None:
    journal = _Journal([
        _open("/c2dm/register3", host="android.googleapis.com"),
        _open("/v1/chat", host="api.theapp.test"),
        _open("/v16.0/app", method="GET", host="graph.facebook.com"),
    ])
    named = engine_analyze.network_calls(_engine(tmp_path, journal, app_hosts=["api.theapp.test"]))
    assert named == ["POST /v1/chat -> no answer yet"]


def test_a_named_host_covers_its_subdomains(tmp_path: Any) -> None:
    """A backend is named once and reached at several subdomains: staging, api, eu1."""
    journal = _Journal([_open("/v1/chat", host="api.staging.theapp.test")])
    assert engine_analyze.network_calls(
        _engine(tmp_path, journal, app_hosts=["theapp.test"])) == ["POST /v1/chat -> no answer yet"]


def test_a_named_host_does_not_match_a_lookalike(tmp_path: Any) -> None:
    """`theapp.test` must not swallow `nottheapp.test`; the boundary is a dot, not a substring."""
    journal = _Journal([_open("/v1/chat", host="nottheapp.test")])
    assert engine_analyze.network_calls(
        _engine(tmp_path, journal, app_hosts=["theapp.test"])) is None


def test_a_call_with_no_host_is_not_guessed_at(tmp_path: Any) -> None:
    journal = _Journal([{"flow": 1, "ts": time.time(), "method": "GET", "path": "/x", "open": True}])
    assert engine_analyze.network_calls(
        _engine(tmp_path, journal, app_hosts=["theapp.test"])) is None


# ------------------------------------------------- it has to survive the trip to the caller


def test_an_action_keeps_the_field_on_its_folded_observation() -> None:
    """Every action trims `meta` to the `changed` preset, and a key not named there is gone.

    Measured on a real run: AUA computed the field and returned it, and the harness's copy of the
    very same observation -- same fingerprint -- did not have it, on every tap. The engine half of
    this feature was working and no caller could see it, which is the same shape as the `checked`
    flag that went missing and made a whole class of contract bullet unverifiable.
    """
    from android_ui_analyser.projection import OBSERVATION_META_PRESETS, Projection

    assert "network_calls" in OBSERVATION_META_PRESETS["changed"]
    view = Projection.for_observation(None, meta="changed")
    assert view is not None
    kept = view.apply({
        "observation": {
            "screen": {"package": "com.example.app"},
            "elements": [{"id": "el:a", "text": "Sign in", "clickable": True,
                          "bounds": [0, 0, 10, 10]}],
            "meta": {"fingerprint": "abc", "duration_ms": 3,
                     "network_calls": ["POST /v1/auth/login -> 200"]},
        }
    })
    assert (kept["observation"]["meta"].get("network_calls")
            == ["POST /v1/auth/login -> 200"]), kept["observation"]["meta"]


def test_one_backend_can_be_named_from_the_environment(monkeypatch: Any) -> None:
    """Naming one host is the normal case, and a bare value must not arrive as a string.

    The env reader splits on commas, so a single host would be a scalar and fail the list
    validator outright -- `AUA_NETWORK__APP_HOSTS=theapp.test` raised rather than configuring
    anything, which is the whole feature refusing to turn on for its most ordinary caller.
    """
    from android_ui_analyser.config import load_config

    monkeypatch.setenv("AUA_NETWORK__APP_HOSTS", "theapp.test")
    assert load_config().network.app_hosts == ["theapp.test"]
    monkeypatch.setenv("AUA_NETWORK__APP_HOSTS", "theapp.test,other.test")
    assert load_config().network.app_hosts == ["theapp.test", "other.test"]


# ------------------------------------ what the app asked for, and what came back

def test_a_call_that_came_back_is_reported_with_its_status(tmp_path: Any) -> None:
    """The half that was missing. Reporting only unanswered calls threw away the whole journal.

    Measured on a real run of a real app: 25 backend calls across the run and not one of them was
    unanswered at the moment an observation was taken, because that backend answers in 55 ms and
    AUA settles the screen before handing anything back. Reported: nothing. Yet those windows held
    a `PUT /v1/push-token -> 401` and its retry, and the `PUT /v1/profile -> 200` that
    *was* the change under test -- the judge called that clause unevidenced while the proof sat in
    the journal.
    """
    journal = _Journal(_answered("/v1/session", status=201))
    assert engine_analyze.network_calls(
        _engine(tmp_path, journal, app_hosts=["example.test"])) == ["POST /v1/session -> 201"]


def test_an_unanswered_call_says_so_in_words(tmp_path: Any) -> None:
    """Not a number: this model reads a status code as a label, not as arithmetic."""
    journal = _Journal([_open("/v1/login")])
    assert engine_analyze.network_calls(
        _engine(tmp_path, journal, app_hosts=["example.test"])) == [
            "POST /v1/login -> no answer yet"]


def test_a_failure_and_its_retry_both_survive(tmp_path: Any) -> None:
    """A 401 followed by a 200 is the app recovering; either alone tells the wrong story."""
    journal = _Journal(_answered("/v1/push-token", status=401, method="PUT", ago=0.4)
                       + _answered("/v1/push-token", status=200, method="PUT"))
    assert engine_analyze.network_calls(
        _engine(tmp_path, journal, app_hosts=["example.test"])) == [
            "PUT /v1/push-token -> 401",
            "PUT /v1/push-token -> 200"]


def test_calls_are_reported_oldest_first(tmp_path: Any) -> None:
    journal = _Journal(_answered("/v1/config", method="GET", ago=0.9)
                       + _answered("/v1/items", method="GET", ago=0.5)
                       + [_open("/v1/send", ago=0.1)])
    assert engine_analyze.network_calls(
        _engine(tmp_path, journal, app_hosts=["example.test"])) == [
            "GET /v1/config -> 200",
            "GET /v1/items -> 200",
            "POST /v1/send -> no answer yet"]


def test_a_tap_that_asked_the_backend_for_nothing_says_nothing(tmp_path: Any) -> None:
    """Silence is itself an answer -- three taps in the measured run moved no traffic at all."""
    assert engine_analyze.network_calls(_engine(tmp_path, _Journal([]), app_hosts=["example.test"])) is None


def test_a_vendor_call_is_still_excluded_once_it_answers(tmp_path: Any) -> None:
    journal = _Journal(_answered("/v1/firelog/legacy/batchlog", host="crashlyticsreports-pa.googleapis.com"))
    assert engine_analyze.network_calls(
        _engine(tmp_path, journal, app_hosts=["example.test"])) is None


def test_the_window_starts_where_the_last_observation_ended(tmp_path: Any) -> None:
    """"Since the last call" is the question -- not "open at this instant", which was the bug."""
    journal = _Journal(_answered("/v1/first", ago=0.4))
    engine = _engine(tmp_path, journal, app_hosts=["example.test"])
    assert engine_analyze.network_calls(engine) == ["POST /v1/first -> 200"]
    # the same call must not be reported twice: the next observation starts after this one
    assert engine_analyze.network_calls(engine) is None
