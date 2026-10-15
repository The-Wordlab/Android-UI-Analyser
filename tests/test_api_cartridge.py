"""`aua api run`: a cartridge sends a backend the calls a client app makes, with no device.

The fake backend below serves the fictional API of the shipped sample, so the sample an agent
starts from is itself exercised end to end.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from android_ui_analyser import api_cartridge
from android_ui_analyser.api_cartridge import compare, run, sample_text, shape, write_expect
from android_ui_analyser.cli import app
from android_ui_analyser.errors import UsageError

ENV = {"EXAMPLE_CLIENT_KEY": "client-key-123"}


class NotesBackend:
    """The sample's backend. ``change`` alters a response the way a backend PR might."""

    def __init__(self, change: dict[str, Any] | None = None) -> None:
        self.change = change or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("X-Client-Key") != ENV["EXAMPLE_CLIENT_KEY"]:
            return httpx.Response(403, json={"error": "bad client key"})
        path = request.url.path
        if path == "/v1/sessions/anonymous":
            install = request.headers["X-Install-Id"]
            return httpx.Response(
                200,
                json={
                    "access_token": f"tok-{install}",
                    "expires_in": 3600,
                    "user": {"id": f"u-{install[:8]}", "anonymous": True},
                },
            )
        if not request.headers.get("Authorization", "").startswith("Bearer tok-"):
            return httpx.Response(401, json={"error": "no session"})
        if path == "/v1/notes" and request.method == "POST":
            sent = json.loads(request.content)
            return httpx.Response(201, json=self._note(sent["text"]))
        if path.startswith("/v1/notes/"):
            if path.rsplit("/", 1)[1] != "n-1":
                return httpx.Response(404, json={"error": "no such note"})
            return httpx.Response(200, json={**self._note("Buy oat milk"), "tags": ["shopping"]})
        if path == "/v1/assistant/replies":
            sent = json.loads(request.content)
            done = {"reply_id": "r-9", "tokens": self.change.get("tokens", 12)}
            stream = (
                'event: delta\ndata: {"text": "You need "}\n\n'
                f'event: delta\ndata: {{"text": "{sent["message"][:4]}"}}\n\n'
                f"event: done\ndata: {json.dumps(done)}\n\n"
            )
            return httpx.Response(
                200, content=stream.encode(), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(404)

    def _note(self, text: str) -> dict[str, Any]:
        note: dict[str, Any] = {
            "id": "n-1",
            "text": text,
            "pinned": False,
            "created_at": "2026-01-01T00:00:00Z",
        }
        for key, value in self.change.get("note", {}).items():
            if value is _GONE:
                note.pop(key)
            else:
                note[key] = value
        return note


_GONE = object()


@pytest.fixture
def cartridge(tmp_path: Path) -> Path:
    path = tmp_path / "notes.yaml"
    path.write_text(sample_text(), encoding="utf-8")
    return path


def _run(path: Path, backend: NotesBackend, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("env", ENV)
    return run(path, transport=httpx.MockTransport(backend), **kwargs)


def test_the_sample_runs_as_a_client_would(cartridge: Path) -> None:
    backend = NotesBackend()
    result = _run(cartridge, backend)

    assert result["ok"] is True, result
    assert [s["id"] for s in result["steps"]] == ["session", "create_note", "read_note", "ask"]
    assert all(s["ok"] for s in result["steps"])
    assert result["breaks"] == 0 and result["warnings"] == 0
    session, create, read, ask = backend.requests
    # The token one response handed out authorises the later calls, and the saved id is the path.
    assert create.headers["Authorization"] == f"Bearer tok-{session.headers['X-Install-Id']}"
    assert read.url.path == "/v1/notes/n-1"
    # One fresh install id for the whole run, and the input's default in the body.
    assert len({r.headers["X-Install-Id"] for r in backend.requests}) == 1
    assert json.loads(create.content) == {"text": "Buy oat milk and coffee", "pinned": False}
    assert json.loads(ask.content) == {"note_id": "n-1", "message": "What do I need to buy?"}
    stream = result["steps"][3]
    assert stream["events"] == 3 and "first_byte_ms" in stream
    assert stream["saved"] == ["reply_id"]


def test_an_input_changes_the_request_and_each_run_is_a_new_install(cartridge: Path) -> None:
    first, second = NotesBackend(), NotesBackend()
    _run(cartridge, first)
    _run(cartridge, second, inputs={"question": "Is it raining?"})

    assert json.loads(second.requests[3].content)["message"] == "Is it raining?"
    assert first.requests[0].headers["X-Install-Id"] != second.requests[0].headers["X-Install-Id"]
    with pytest.raises(UsageError, match="no input `questoin`"):
        _run(cartridge, NotesBackend(), inputs={"questoin": "x"})


def test_a_missing_secret_is_named_before_anything_is_sent(cartridge: Path) -> None:
    backend = NotesBackend()
    with pytest.raises(UsageError, match="EXAMPLE_CLIENT_KEY"):
        _run(cartridge, backend, env={})
    assert backend.requests == []


def test_secrets_and_tokens_never_reach_the_result(cartridge: Path) -> None:
    result = json.dumps(_run(cartridge, NotesBackend()))

    assert ENV["EXAMPLE_CLIENT_KEY"] not in result
    assert "tok-" not in result
    assert "You need" in result  # the reply itself is shown


def test_a_count_named_like_a_token_is_still_shown() -> None:
    shown = api_cartridge._redact({"token": "abc", "usage": {"totalTokens": 42}}, [])

    assert shown == {"token": "***", "usage": {"totalTokens": 42}}


def test_a_failed_step_stops_the_run_and_says_why(cartridge: Path) -> None:
    result = _run(cartridge, NotesBackend(), env={"EXAMPLE_CLIENT_KEY": "wrong"})

    assert result["ok"] is False
    assert result["steps"][0]["error"] == "status 403, expected 2xx"
    assert [s["error"] for s in result["steps"][1:]] == ["not run: an earlier step failed"] * 3


def test_a_value_the_response_lacks_cannot_be_saved(cartridge: Path) -> None:
    result = _run(cartridge, NotesBackend({"note": {"id": _GONE}}))

    create = result["steps"][1]
    assert create["error"] == "could not save `note_id`: the response has no `body.id`"
    assert result["steps"][2]["error"] == "not run: an earlier step failed"


def test_now_is_the_current_utc_time(tmp_path: Path) -> None:
    """Backends reject client timestamps far from their clock, so a fixed input goes stale."""
    flow = tmp_path / "events.yaml"
    step = {"id": "send", "method": "POST", "path": "/events", "json": {"at": "{{now}}"}}
    flow.write_text(yaml.safe_dump({"cartridge": 1, "base_url": "https://x.test", "steps": [step]}))
    sent: list[Any] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content)["at"])
        return httpx.Response(200, json={})

    assert run(flow, env={}, transport=httpx.MockTransport(handle))["ok"] is True
    at = datetime.strptime(sent[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    assert abs((datetime.now(UTC) - at).total_seconds()) < 5


def test_a_null_cannot_be_saved_as_the_next_steps_id(cartridge: Path) -> None:
    """A null id used to be sent on as the string "null" (`/v1/notes/null`)."""
    result = _run(cartridge, NotesBackend({"note": {"id": None}}))

    create = result["steps"][1]
    assert create["error"] == "could not save `note_id`: `body.id` is null"
    assert result["steps"][2]["error"] == "not run: an earlier step failed"


def test_carried_values_keep_one_install_and_skip_its_sign_in(cartridge: Path) -> None:
    saved: dict[str, Any] = {}
    install: dict[str, str] = {}
    first, second = NotesBackend(), NotesBackend()
    _run(cartridge, first, saved=saved, install=install)
    result = _run(cartridge, second, saved=saved, install=install)

    assert result["ok"] is True, result
    assert result["steps"][0] == {
        "id": "session",
        "method": "POST",
        "path": "/v1/sessions/anonymous",
        "ok": True,
        "skipped": "once: token already known",
    }
    # The second run is the same install, signed in with the token the first run got.
    assert [r.url.path for r in second.requests] == [
        "/v1/notes",
        "/v1/notes/n-1",
        "/v1/assistant/replies",
    ]
    assert second.requests[0].headers["X-Install-Id"] == first.requests[0].headers["X-Install-Id"]
    assert second.requests[0].headers["Authorization"] == first.requests[1].headers["Authorization"]
    assert install == {"id": first.requests[0].headers["X-Install-Id"]}
    assert set(saved) == {"token", "note_id", "reply_id"}


def test_once_needs_something_to_save(tmp_path: Path) -> None:
    flow = tmp_path / "once.yaml"
    step = {"id": "sign_in", "method": "POST", "path": "/sign-in", "once": True}
    flow.write_text(yaml.safe_dump({"cartridge": 1, "steps": [step]}))

    with pytest.raises(UsageError, match="`once: true` needs `save:`"):
        api_cartridge.load(flow)


def test_a_step_can_save_what_it_sent(tmp_path: Path) -> None:
    """A later run continues a conversation with the question this one asked."""
    flow = tmp_path / "ask.yaml"
    step = {
        "id": "ask",
        "method": "POST",
        "path": "/ask",
        "json": {"question": "{{input.q}}"},
        "save": {"asked": "request.json.question"},
    }
    doc = {"cartridge": 1, "base_url": "https://x.test", "inputs": {"q": "Why?"}, "steps": [step]}
    flow.write_text(yaml.safe_dump(doc))
    saved: dict[str, Any] = {}

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    assert run(flow, env={}, transport=transport, saved=saved)["ok"] is True
    assert saved == {"asked": "Why?"}


def test_a_flow_step_runs_a_shared_cartridge_in_its_place(tmp_path: Path) -> None:
    """A sign-in written once: every flow starts with it, and it also runs on its own."""
    sign_in = {
        "cartridge": 1,
        "env": ["EXAMPLE_CLIENT_KEY"],
        "inputs": {"who": "guest"},
        "headers": {"X-Client-Key": "{{env.EXAMPLE_CLIENT_KEY}}"},
        "steps": [
            {
                "id": "sign_in",
                "method": "POST",
                "path": "/sign-in",
                "json": {"as": "{{input.who}}"},
                "once": True,
                "save": {"token": "body.token"},
            }
        ],
    }
    add_note = {
        "cartridge": 1,
        "base_url": "https://notes.example.test",
        "steps": [
            {"flow": "sign-in.yaml"},
            {
                "id": "add",
                "method": "POST",
                "path": "/notes",
                "headers": {"Authorization": "Bearer {{token}}"},
            },
        ],
    }
    for name, doc in (("sign-in.yaml", sign_in), ("add-note.yaml", add_note)):
        (tmp_path / name).write_text(yaml.safe_dump(doc))
    sent: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"token": "tok-1"})

    transport = httpx.MockTransport(backend)
    flow = tmp_path / "add-note.yaml"
    result = run(flow, env=ENV, inputs={"who": "tester"}, transport=transport)

    assert result["ok"] is True, result
    assert [s["id"] for s in result["steps"]] == ["sign_in", "add"]
    # The shared step sends what it sends on its own: its headers, its input.
    assert sent[0].headers["X-Client-Key"] == ENV["EXAMPLE_CLIENT_KEY"]
    assert json.loads(sent[0].content) == {"as": "tester"}
    assert "X-Client-Key" not in sent[1].headers
    assert sent[1].headers["Authorization"] == "Bearer tok-1"
    with pytest.raises(UsageError, match="EXAMPLE_CLIENT_KEY"):
        run(flow, env={}, transport=transport)
    alone = run(tmp_path / "sign-in.yaml", env=ENV, base_url="https://x.test", transport=transport)
    assert [s["id"] for s in alone["steps"]] == ["sign_in"]


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"flow": "loop.yaml"}, "includes itself"),
        ({"flow": "../sign-in.yaml"}, "in the same folder"),
        ({"flow": "common/sign-in.yaml"}, "in the same folder"),
        ({"flow": "sign-in.yaml", "once": True}, "in the same folder"),
        ({"flow": "missing.yaml"}, "no cartridge at"),
    ],
)
def test_a_flow_step_names_one_cartridge_beside_it(
    tmp_path: Path, step: dict[str, Any], message: str
) -> None:
    flow = tmp_path / "loop.yaml"
    flow.write_text(yaml.safe_dump({"cartridge": 1, "steps": [step]}))

    with pytest.raises(UsageError, match=message):
        api_cartridge.load(flow)


def test_a_log_holds_every_call_in_full_with_no_secret_or_token(cartridge: Path) -> None:
    exchanges: list[dict[str, Any]] = []
    _run(cartridge, NotesBackend(), exchanges=exchanges, max_chars=10)

    assert [e["step"] for e in exchanges] == ["session", "create_note", "read_note", "ask"]
    create = exchanges[1]
    assert create["request"]["method"] == "POST"
    assert create["request"]["url"] == "https://api.example.com/v1/notes"
    assert create["request"]["json"] == {"text": "Buy oat milk and coffee", "pinned": False}
    assert create["request"]["headers"]["Authorization"] == "***"
    assert create["response"]["status"] == 201
    assert create["response"]["body"]["text"] == "Buy oat milk and coffee"
    # Not shortened like the printed preview: the whole stream is there.
    assert [e["event"] for e in exchanges[3]["response"]["events"]] == ["delta", "delta", "done"]
    logged = json.dumps(exchanges)
    assert ENV["EXAMPLE_CLIENT_KEY"] not in logged
    assert "tok-" not in logged


def test_a_backend_change_a_client_would_feel_is_a_break(cartridge: Path) -> None:
    changed = NotesBackend({"note": {"pinned": _GONE, "created_at": 1700000000}, "tokens": "12"})
    result = _run(cartridge, changed)

    assert result["ok"] is False
    found = {(f["where"], f["change"]) for s in result["steps"] for f in s.get("findings", [])}
    assert ("body.pinned", "missing") in found
    assert ("body.created_at", "now number, was string") in found
    assert ("events.done.tokens", "now string, was number") in found
    assert result["breaks"] == 5  # pinned and created_at in two steps, tokens once


def test_new_fields_are_fine_and_a_null_is_only_a_warning() -> None:
    assert compare({"id": "string"}, {"id": "string", "added": "number"}) == []
    assert compare({"id": "string"}, {"id": "null"}) == [
        {"level": "warn", "where": "body.id", "change": "now null, was string"}
    ]
    assert compare({"items": [{"id": "string"}]}, {"items": []}) == []
    assert shape({"a": [1, None, 2.5], "b": [{"x": 1}, {"y": "s"}]}) == {
        "a": ["number"],
        "b": [{"x": "number", "y": "string"}],
    }


def test_save_expect_records_shapes_and_keeps_the_file_as_written(tmp_path: Path) -> None:
    text = sample_text()
    path = tmp_path / "fresh.yaml"
    path.write_text(text[: text.index("# Written by")], encoding="utf-8")

    first = _run(path, NotesBackend(), save_expect=True)
    saved = path.read_text(encoding="utf-8")

    assert first["expect"].startswith("saved 4 response shapes")
    assert saved.startswith("# AUA API cartridge")  # comments above the block survive
    assert yaml.safe_load(saved)["expect"] == yaml.safe_load(text)["expect"]
    # Saving again rewrites the block instead of appending a second one.
    _run(path, NotesBackend(), save_expect=True)
    assert path.read_text(encoding="utf-8") == saved
    later = _run(path, NotesBackend({"note": {"text": _GONE}}))
    assert later["breaks"] == 2


def test_the_sample_is_exactly_what_save_expect_writes(cartridge: Path) -> None:
    _run(cartridge, NotesBackend(), save_expect=True)

    assert cartridge.read_text(encoding="utf-8") == sample_text()


def test_a_failed_run_is_never_saved_as_the_baseline(cartridge: Path) -> None:
    before = cartridge.read_text(encoding="utf-8")
    result = _run(cartridge, NotesBackend(), env={"EXAMPLE_CLIENT_KEY": "wrong"}, save_expect=True)

    assert result["expect"].startswith("not saved")
    assert cartridge.read_text(encoding="utf-8") == before


def test_expect_must_stay_last_to_be_rewritten(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("cartridge: 1\nexpect: {}\nsteps: []\n", encoding="utf-8")
    with pytest.raises(UsageError, match="not the last key"):
        write_expect(path, {})


def test_a_lone_placeholder_keeps_its_type_in_a_json_body(tmp_path: Path) -> None:
    path = tmp_path / "typed.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "cartridge": 1,
                "base_url": "https://api.example.com",
                "steps": [
                    {"id": "a", "method": "GET", "path": "/count", "save": {"n": "body.n"}},
                    {
                        "id": "b",
                        "method": "POST",
                        "path": "/echo",
                        "json": {"n": "{{n}}", "s": "#{{n}}"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    sent: list[Any] = []

    def backend(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/count":
            return httpx.Response(200, json={"n": 7})
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={})

    assert run(path, env={}, transport=httpx.MockTransport(backend))["ok"] is True
    assert sent == [{"n": 7, "s": "#7"}]


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ({"steps": []}, "not an AUA API cartridge"),
        ({"cartridge": 1, "steps": []}, "has no steps"),
        ({"cartridge": 1, "steps": [{"id": "a", "method": "GET", "path": "x"}]}, "starts with `/`"),
        (
            {"cartridge": 1, "steps": [{"id": "a", "method": "GET", "path": "/x"}] * 2},
            "two steps are called `a`",
        ),
        (
            {
                "cartridge": 1,
                "steps": [{"id": "a", "method": "GET", "path": "/{{input.q}}"}],
                "base_url": "http://h",
            },
            "does not declare: q",
        ),
    ],
)
def test_mistakes_in_a_cartridge_are_usage_errors(
    tmp_path: Path, doc: dict[str, Any], message: str
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(UsageError, match=message):
        run(path, env={})


def test_a_stream_reports_when_each_kind_of_event_first_arrived(tmp_path: Path) -> None:
    stream = (
        'event: start\ndata: {"id": "r"}\n\n'
        'event: delta\ndata: {"text": "a"}\n\n'
        'event: delta\ndata: {"text": "b"}\n\n'
        'event: done\ndata: {"tokens": 3}'  # the last event has no blank line after it
    )
    flow = tmp_path / "stream.yaml"
    flow.write_text(
        json.dumps(
            {
                "cartridge": 1,
                "base_url": "https://api.example.com",
                "steps": [{"id": "ask", "method": "POST", "path": "/v1/replies"}],
            }
        )
    )
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200, content=stream.encode(), headers={"content-type": "text/event-stream"}
        )
    )

    step = run(flow, env={}, transport=transport)["steps"][0]

    assert list(step["event_ms"]) == ["start", "delta", "done"]
    assert all(isinstance(ms, int) and ms >= 0 for ms in step["event_ms"].values())


class _Server:
    """A real local backend for the CLI, whose one answer the test can change."""

    def __init__(self) -> None:
        self.body: dict[str, Any] = {"id": "x", "name": "n"}
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server's hook name
                data = json.dumps(server.body).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


def test_the_cli_exits_8_when_a_backend_change_breaks_the_cartridge(tmp_path: Path) -> None:
    server = _Server()
    path = tmp_path / "item.yaml"
    path.write_text(
        "cartridge: 1\nsteps:\n  - id: item\n    method: GET\n    path: /items/1\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    try:
        first = runner.invoke(
            app, ["api", "run", str(path), "--base-url", server.url, "--save-expect"]
        )
        assert first.exit_code == 0, first.output
        server.body = {"id": "x"}
        second = runner.invoke(app, ["api", "run", str(path), "--base-url", server.url])
    finally:
        server.httpd.shutdown()

    assert second.exit_code == 8, second.output
    result = json.loads(second.output)
    assert result["steps"][0]["findings"] == [
        {"level": "break", "where": "body.name", "change": "missing"}
    ]


def test_the_cli_prints_the_sample() -> None:
    result = CliRunner().invoke(app, ["api", "sample"])

    assert result.exit_code == 0
    assert result.output == api_cartridge.sample_text()
