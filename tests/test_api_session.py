"""`aua api run --session`: one user and install kept across runs, logged, and handed over."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from android_ui_analyser import api_session
from android_ui_analyser.cli import app
from android_ui_analyser.errors import UsageError

ENV = {"EXAMPLE_KEY": "client-key-123"}

CHAT = {
    "cartridge": 1,
    "base_url": "https://chat.example.test",
    "env": ["EXAMPLE_KEY"],
    "inputs": {"text": "Hello"},
    "headers": {"X-Key": "{{env.EXAMPLE_KEY}}", "X-Install-Id": "{{install.device}}"},
    "steps": [
        {
            "id": "sign_in",
            "method": "POST",
            "path": "/v1/sign-in",
            "once": True,
            "save": {"token": "body.token"},
        },
        {
            "id": "send",
            "method": "POST",
            "path": "/v1/messages",
            "headers": {"Authorization": "Bearer {{token}}"},
            "json": {"text": "{{input.text}}"},
            "save": {"thread": "body.thread"},
        },
    ],
}


def _continue_chat() -> dict[str, Any]:
    doc = json.loads(json.dumps(CHAT))
    doc["steps"][1]["json"]["thread"] = "{{thread}}"
    return doc


class ChatBackend:
    """A user is a token on one install; a chat is a thread its messages pile up in."""

    def __init__(self) -> None:
        self.users: dict[str, str] = {}
        self.threads: dict[str, list[str]] = {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        install = request.headers["X-Install-Id"]
        if request.url.path == "/v1/sign-in":
            token = f"tok-{len(self.users)}"
            self.users[token] = install
            return httpx.Response(200, json={"token": token})
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if self.users.get(token) != install:
            return httpx.Response(401, json={"error": "no such user on this install"})
        sent = json.loads(request.content)
        thread = sent.get("thread") or f"t-{len(self.threads)}"
        self.threads.setdefault(thread, []).append(sent["text"])
        return httpx.Response(200, json={"thread": thread, "messages": len(self.threads[thread])})


def _flow(folder: Path, doc: dict[str, Any], name: str = "chat.yaml") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def _run(root: Path, flow: Path, backend: ChatBackend, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("env", ENV)
    return api_session.run(
        "bug-123", flow, root=root, transport=httpx.MockTransport(backend), **kwargs
    )


def test_the_next_run_is_the_same_user_in_the_same_chat(tmp_path: Path) -> None:
    root, backend = tmp_path / "sessions", ChatBackend()
    chat = _flow(tmp_path / "flows", CHAT)
    later = _flow(tmp_path / "flows", _continue_chat(), "continue-chat.yaml")

    first = _run(root, chat, backend)
    second = _run(root, later, backend, inputs={"text": "Still there?"})

    assert first["ok"] and second["ok"], second
    assert second["steps"][0]["skipped"] == "once: token already known"
    assert len(backend.users) == 1
    assert backend.threads == {"t-0": ["Hello", "Still there?"]}
    assert second["session"] == {
        "name": "bug-123",
        "new": False,
        "kept": ["thread", "token"],
        "log": str(root / "bug-123" / "runs" / "002-continue-chat.json"),
    }


def test_a_session_goes_on_against_the_backend_it_started_on(tmp_path: Path) -> None:
    root, backend = tmp_path / "sessions", ChatBackend()
    chat = _flow(tmp_path / "flows", CHAT)

    _run(root, chat, backend, base_url="http://localhost:8000")
    result = _run(root, chat, backend)

    assert result["base_url"] == "http://localhost:8000"
    assert {str(r.url.host) for r in backend.requests} == {"localhost"}


def test_each_run_is_logged_beside_its_flow_and_map(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    version = tmp_path / "map" / "notes" / "1.0.0"
    version.mkdir(parents=True)
    meta = {"client": "notes", "version": "1.0.0", "tag": "v1.0.0", "frozen": {"x": "y"}}
    (version / "version.yaml").write_text(yaml.safe_dump(meta))
    schema = {"endpoints": {"POST /v1/messages": {"response": {"body": {"thread": "string"}}}}}
    (version / "schema.yaml").write_text(yaml.safe_dump(schema))
    chat = _flow(version / "flows", CHAT)

    result = _run(root, chat, ChatBackend())

    folder = root / "bug-123"
    log = json.loads((folder / "runs" / "001-chat.json").read_text())
    assert log["flow"] == "map/notes/1.0.0/flows/chat.yaml"
    assert log["result"]["ok"] is True
    assert [c["step"] for c in log["calls"]] == ["sign_in", "send"]
    assert log["calls"][1]["response"]["body"] == {"thread": "t-0", "messages": 1}
    copy = folder / "map" / "notes" / "1.0.0"
    assert (copy / "flows" / "chat.yaml").read_text() == chat.read_text()
    assert (copy / "schema.yaml").read_text() == (version / "schema.yaml").read_text()
    # The copy holds one of the version's flows: a freeze fingerprint would call the rest edits.
    assert "frozen" not in yaml.safe_load((copy / "version.yaml").read_text())
    assert result["session"]["new"] is True
    # The sign-in is kept for the next run, and nowhere else; the client key is never written.
    for path in folder.rglob("*"):
        if path.is_file():
            text = path.read_text()
            assert ENV["EXAMPLE_KEY"] not in text, path
            assert ("tok-0" in text) == (path.name == "session.json"), path
    assert oct(folder.stat().st_mode & 0o777) == "0o700"


def test_a_handed_over_session_continues_the_same_chat(tmp_path: Path) -> None:
    mine, theirs, backend = tmp_path / "mine", tmp_path / "theirs", ChatBackend()
    _run(mine, _flow(tmp_path / "flows", CHAT), backend)

    zipped = api_session.export("bug-123", tmp_path / "bug.zip", root=mine)
    imported = api_session.import_zip(zipped["zip"], root=theirs)
    flow = Path(imported["flows"][0])
    result = _run(theirs, _flow(flow.parent, _continue_chat(), "continue-chat.yaml"), backend)

    assert zipped["files"] == 3 and "live sign-in" in zipped["warning"]
    assert imported["session"]["runs"] == 1
    assert flow == theirs / "bug-123" / "flows" / "chat.yaml"
    assert result["ok"] is True, result
    assert len(backend.users) == 1
    assert backend.threads == {"t-0": ["Hello", "Hello"]}


def test_an_import_never_writes_outside_its_session(tmp_path: Path) -> None:
    zipped = tmp_path / "evil.zip"
    with zipfile.ZipFile(zipped, "w") as zf:
        zf.writestr("session.json", json.dumps({"format": 1, "name": "evil"}))
        zf.writestr("../escaped.txt", "x")

    with pytest.raises(UsageError, match="outside its session"):
        api_session.import_zip(zipped, root=tmp_path / "sessions")
    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path / "sessions" / "evil").exists()


def test_an_import_never_replaces_a_session(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    _run(root, _flow(tmp_path / "flows", CHAT), ChatBackend())
    zipped = api_session.export("bug-123", tmp_path / "bug.zip", root=root)["zip"]

    with pytest.raises(UsageError, match="already here"):
        api_session.import_zip(zipped, root=root)


def test_a_session_unused_for_two_weeks_is_removed(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    _run(root, _flow(tmp_path / "flows", CHAT), ChatBackend())
    path = root / "bug-123" / "session.json"
    data = json.loads(path.read_text())
    data["last_used"] = "2020-01-01T00:00:00Z"
    path.write_text(json.dumps(data))
    (root / "not-a-session").mkdir()

    listed = api_session.list_sessions(root)

    assert listed["removed_unused"] == ["bug-123"]
    assert listed["sessions"] == []
    assert not (root / "bug-123").exists()
    assert (root / "not-a-session").exists()  # only sessions are ever removed


def test_a_session_name_cannot_leave_the_sessions_folder(tmp_path: Path) -> None:
    with pytest.raises(UsageError, match="not a session name"):
        api_session.open_session("../x", tmp_path)


def test_the_cli_keeps_a_session_and_lists_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    tokens: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server's hook name
            self.rfile.read(int(self.headers.get("content-length") or 0))
            if self.path == "/v1/sign-in":
                tokens.append(f"tok-{len(tokens)}")
                body = {"token": tokens[-1]}
            else:
                body = {"thread": "t-0", "used": self.headers["Authorization"]}
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    monkeypatch.setattr(api_session, "ROOT", tmp_path / "sessions")
    monkeypatch.setenv("EXAMPLE_KEY", ENV["EXAMPLE_KEY"])
    flow = str(_flow(tmp_path / "flows", CHAT))
    runner = CliRunner()
    try:
        first = runner.invoke(app, ["api", "run", flow, "--base-url", url, "--session", "s1"])
        second = runner.invoke(app, ["api", "run", flow, "--session", "s1"])
    finally:
        httpd.shutdown()
    listed = runner.invoke(app, ["api", "session", "list"])

    assert first.exit_code == 0 and second.exit_code == 0, second.output
    assert tokens == ["tok-0"]
    assert json.loads(second.output)["steps"][0]["ok"] is True
    (session,) = json.loads(listed.output)["sessions"]
    assert session["name"] == "s1" and session["runs"] == 2 and session["base_url"] == url
    deleted = runner.invoke(app, ["api", "session", "delete", "s1"])
    assert deleted.exit_code == 0 and not os.path.exists(session["dir"])
