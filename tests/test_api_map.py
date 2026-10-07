"""`aua api map`: per-version schemas and flows that an agent writes from a client's source.

The client below is a fictional notes app with two releases and unreleased work on `main`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from android_ui_analyser.api_cartridge import check_contract, check_response, run
from android_ui_analyser.api_map import freeze, new_version, status
from android_ui_analyser.errors import UsageError


def _git(repo: Path, *args: str, date: str = "2026-01-01T00:00:00") -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com"}
    env |= {"GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    env |= {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
    ).stdout


def _commit(repo: Path, files: dict[str, str], date: str, tag: str | None = None) -> None:
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", date, date=date)
    if tag:
        _git(repo, "tag", tag, date=date)


def _api(*calls: str) -> str:
    body = "".join(
        f'    @{c.split()[0]}("{c.split()[1]}")\n    suspend fun call{i}(): Any\n'
        for i, c in enumerate(calls)
    )
    return "interface NotesApi {\n" + body + "}\n"


@pytest.fixture
def config(tmp_path: Path) -> Path:
    app = tmp_path / "notes-android"
    app.mkdir()
    _git(app, "init", "-q", "-b", "main")
    _commit(
        app,
        {
            "src/NotesApi.kt": _api("POST v1/notes", "GET v1/notes/{id}"),
            "src/Models.kt": "data class Note(val id: String)\n",
        },
        "2026-01-01T00:00:00",
        tag="v1.0.0",
    )
    _commit(
        app,
        {
            "src/NotesApi.kt": _api(
                "POST v1/notes", "GET v1/notes/{id}", "POST v1/assistant/replies"
            ),
            "src/Models.kt": "data class Note(val id: String, val pinned: Boolean = false)\n",
        },
        "2026-02-01T00:00:00",
        tag="v1.1.0",
    )
    _commit(
        app,
        {"src/NotesApi.kt": _api("POST v1/notes", "GET v1/notes/{id}", "DELETE v1/notes/{id}")},
        "2026-03-01T00:00:00",
    )
    backend = tmp_path / "notes-backend"
    backend.mkdir()
    path = tmp_path / "aua-api.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "backend": {"repo": str(backend)},
                "clients": [
                    {
                        "name": "android",
                        "repo": str(app),
                        "scanner": "retrofit",
                        "base_path": "/api/",
                        "tags": "v*",
                        "version": {"tag_pattern": r"^v(\d+\.\d+\.\d+)"},
                    }
                ],
            }
        )
    )
    return path


def _new(config: Path, version: str, **kwargs: Any) -> dict[str, Any]:
    return new_version(config, config.parent / "cache", "android", version, fetch=False, **kwargs)


def _status(config: Path) -> dict[str, Any]:
    clients = status(config, config.parent / "cache", fetch=False)["clients"]
    return {v["version"]: v for v in clients[0]["versions"]} | {"_": clients[0]}


SCHEMA = {
    "endpoints": {
        "POST /v1/notes": {
            "response": {
                "body": {
                    "id": "string",
                    "state": "enum(draft, published)",
                    "pinned?": "boolean",
                    "tags": ["string"],
                }
            }
        },
        "GET /v1/notes/{id}": {"response": {"body": {"id": "string"}}},
    }
}

FLOW = {
    "cartridge": 1,
    "base_url": "https://api.example.com/api",
    "steps": [
        {
            "id": "create",
            "screen": "Editor",
            "method": "POST",
            "path": "/v1/notes",
            "save": {"id": "body.id"},
        },
        {"id": "open", "screen": "Note", "method": "GET", "path": "/v1/notes/{{id}}"},
        {"id": "share", "method": "POST", "path": "/v1/notes/{{id}}/share"},
    ],
}


def _fill(folder: Path) -> Path:
    (folder / "schema.yaml").write_text(yaml.safe_dump(SCHEMA))
    flow = folder / "flows" / "notes.yaml"
    flow.write_text(yaml.safe_dump(FLOW, sort_keys=False))
    return flow


def test_a_new_version_names_its_tag_commit_and_the_calls_to_map(config: Path) -> None:
    result = _new(config, "1.0.0")

    folder = Path(result["path"])
    meta = yaml.safe_load((folder / "version.yaml").read_text())
    assert meta["tag"] == "v1.0.0" and meta["released"] is True and meta["frozen"] is None
    assert (
        meta["commit"] == _git(Path(config.parent / "notes-android"), "rev-parse", "v1.0.0").strip()
    )
    # Paths are relative to the client's base path, as the flows write them.
    assert result["calls"] == [
        {"call": "GET /v1/notes/{}", "source": "src/NotesApi.kt:4"},
        {"call": "POST /v1/notes", "source": "src/NotesApi.kt:2"},
    ]
    assert "endpoints: {}" in (folder / "schema.yaml").read_text()
    assert (folder / "flows").is_dir()
    assert "show v1.0.0:<file>" in result["next"]


def test_the_next_release_starts_from_the_last_and_names_what_changed(config: Path) -> None:
    first = Path(_new(config, "1.0.0")["path"])
    _fill(first)

    result = _new(config, "1.1.0", from_version="1.0.0")

    second = Path(result["path"])
    assert (second / "schema.yaml").read_text() == (first / "schema.yaml").read_text()
    assert (second / "flows" / "notes.yaml").is_file()
    assert result["changes_since"] == {
        "version": "1.0.0",
        "calls_added": [{"call": "POST /v1/assistant/replies", "source": "src/NotesApi.kt:6"}],
        "calls_removed": [],
        "api_files_changed": ["src/NotesApi.kt"],
        "source_files_changed": 2,
    }
    assert yaml.safe_load((second / "version.yaml").read_text())["copied_from"] == "1.0.0"


def test_versions_that_do_not_exist_or_already_exist_are_refused(config: Path) -> None:
    _new(config, "1.0.0")
    with pytest.raises(UsageError, match="already mapped"):
        _new(config, "1.0.0")
    with pytest.raises(UsageError, match="no android release tag has version 9.9.9") as err:
        _new(config, "9.9.9")
    assert "--ref" in (err.value.hint or "")
    with pytest.raises(UsageError, match="0.1.0 is not mapped"):
        _new(config, "1.1.0", from_version="0.1.0")


def test_status_shows_what_each_version_still_lacks(config: Path) -> None:
    _fill(Path(_new(config, "1.0.0")["path"]))
    _fill(Path(_new(config, "1.1.0", from_version="1.0.0")["path"]))

    found = _status(config)

    assert found["1.0.0"]["calls_not_in_schema"] == []
    assert found["1.1.0"]["calls_not_in_schema"] == ["POST /v1/assistant/replies"]
    assert found["1.0.0"]["flows"] == ["notes"] and found["1.0.0"]["schema_endpoints"] == 2
    assert found["_"]["releases_not_mapped"] == []


def test_a_home_relative_cache_dir_is_expanded_not_created_in_the_cwd(
    config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default `cache.dir` is `~/.cache/…`; unexpanded, it left a literal `~` folder."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    new_version(config, "~/.cache/aua", "android", "1.0.0", fetch=False)
    status(config, "~/.cache/aua", fetch=False)

    assert not (work / "~").exists()
    assert (tmp_path / "home" / ".cache" / "aua" / "api-usage").is_dir()


def test_a_frozen_version_can_grow_but_reports_any_edit(config: Path) -> None:
    folder = Path(_new(config, "1.0.0")["path"])
    with pytest.raises(UsageError, match="no schema endpoints or no flows"):
        freeze(config, "android", "1.0.0")
    _fill(folder)

    assert freeze(config, "android", "1.0.0")["frozen"] == 3  # two endpoints and one flow
    assert _status(config)["1.0.0"]["changed_since_freeze"] == []

    # A comment is not an edit; mapping one more endpoint is an addition, not an edit.
    schema = dict(SCHEMA["endpoints"], **{"GET /v1/tags": {"response": {"body": "any"}}})
    (folder / "schema.yaml").write_text("# notes\n" + yaml.safe_dump({"endpoints": schema}))
    found = _status(config)["1.0.0"]
    assert found["changed_since_freeze"] == []
    assert found["added_since_freeze"] == ["endpoint GET /v1/tags"]
    assert freeze(config, "android", "1.0.0")["added"] == ["endpoint GET /v1/tags"]

    # Changing what a frozen endpoint requires is an edit.
    schema["GET /v1/notes/{id}"] = {"response": {"body": {"id": "number"}}}
    (folder / "schema.yaml").write_text(yaml.safe_dump({"endpoints": schema}))
    assert _status(config)["1.0.0"]["changed_since_freeze"] == ["endpoint GET /v1/notes/{}"]
    with pytest.raises(UsageError, match="changed after it was frozen: endpoint GET /v1/notes/"):
        freeze(config, "android", "1.0.0")
    assert freeze(config, "android", "1.0.0", force=True)["refrozen"] == [
        "endpoint GET /v1/notes/{}"
    ]


def test_unreleased_code_is_mapped_by_ref_but_never_frozen(config: Path) -> None:
    result = _new(config, "next", ref="main")

    assert [c["call"] for c in result["calls"]] == [
        "DELETE /v1/notes/{}",
        "GET /v1/notes/{}",
        "POST /v1/notes",
    ]
    _fill(Path(result["path"]))
    with pytest.raises(UsageError, match="not a release"):
        freeze(config, "android", "next")
    assert _status(config)["next"]["moved"] is False


def _backend(note: dict[str, Any]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/api/v1/notes":
            return httpx.Response(201, json=note)
        return httpx.Response(200, json={"id": "n-1"})

    return httpx.MockTransport(handle)


def test_a_flow_in_a_version_folder_is_checked_against_that_version(config: Path) -> None:
    flow = _fill(Path(_new(config, "1.0.0")["path"]))
    note = {"id": "n-1", "state": "archived", "tags": ["a", 7]}

    result = run(flow, env={}, transport=_backend(note))

    assert result["ok"] is False
    assert result["version"]["version"] == "1.0.0" and result["version"]["tag"] == "v1.0.0"
    create, opened, share = result["steps"]
    assert (create["screen"], create["schema"]) == ("Editor", "POST /v1/notes")
    assert {(f["where"], f["level"], f["effect"]) for f in create["findings"]} == {
        ("body.state", "break", "decode_fails"),
        ("body.tags[]", "break", "decode_fails"),
    }
    # An optional field the backend left out is listed, not counted as a warning.
    assert create["optional_absent"] == ["body.pinned"] and result["warnings"] == 0
    assert opened["ok"] is True and "findings" not in opened
    assert share["schema"] is None
    assert result["not_in_schema"] == ["POST /v1/notes/{{id}}/share"]


def test_a_version_the_backend_still_serves_passes(config: Path) -> None:
    flow = _fill(Path(_new(config, "1.0.0")["path"]))
    note = {"id": "n-1", "state": "draft", "pinned": None, "tags": [], "extra": 1}

    assert run(flow, env={}, transport=_backend(note))["breaks"] == 0


def test_running_a_frozen_version_that_was_edited_warns(config: Path) -> None:
    folder = Path(_new(config, "1.0.0")["path"])
    flow = _fill(folder)
    freeze(config, "android", "1.0.0")
    flow.write_text(flow.read_text().replace("Editor", "Composer"))

    result = run(flow, env={}, transport=_backend({"id": "n", "state": "draft", "tags": []}))

    assert result["version"]["changed_since_freeze"] == ["flow notes"]
    assert "flow notes" in result["warning"]


@pytest.mark.parametrize(
    ("schema", "value", "expected"),
    [
        ({"a": "string"}, {"a": None}, [("body.a", "break")]),
        ({"a?": "string"}, {"a": None}, []),
        ({"a": "any"}, {"a": [1, {"b": 2}]}, []),
        ({"a": "number"}, {"a": True}, [("body.a", "break")]),
        ({"a": "boolean"}, {"a": False}, []),
        ({"a": ["number"]}, {"a": "x"}, [("body.a", "break")]),
        ({"a": {"b": "string"}}, {"a": {"b": "x", "c": 1}}, []),
        ({"a": "string?"}, {"a": None}, []),
        ({"a": "string?"}, {}, [("body.a", "break")]),
        ({"a": "string?"}, {"a": 3}, [("body.a", "break")]),
    ],
)
def test_contract_rules(schema: Any, value: Any, expected: list[tuple[str, str]]) -> None:
    assert [(f["where"], f["level"]) for f in check_contract(schema, value)] == expected


def test_a_needed_event_must_arrive_and_an_optional_one_may_not() -> None:
    contract = {"events": {"done": {"id": "string"}, "tool?": {"name": "string"}}}

    assert check_response(contract, None, [("done", {"id": "r"})]) == []
    assert [(f["where"], f["change"]) for f in check_response(contract, None, [])] == [
        ("events.done", "never sent: this version needs it")
    ]
    found = check_response(contract, None, [("done", {"id": "r"}), ("tool", {})])
    assert [f["where"] for f in found] == ["events.tool.name"]


def test_a_schema_yaml_cannot_parse_is_an_error_not_an_empty_schema(config: Path) -> None:
    flow = _fill(Path(_new(config, "1.0.0")["path"]))
    (flow.parent.parent / "schema.yaml").write_text(
        'endpoints:\n  "GET /x": {response: {body: {a?: string}}}\n'
    )

    with pytest.raises(UsageError, match="not valid YAML") as err:
        run(flow, env={}, transport=_backend({}))
    assert '{"name?": string, other: "string?"}' in (err.value.hint or "")


def test_an_unknown_schema_type_is_a_usage_error() -> None:
    with pytest.raises(UsageError, match="unknown schema type 'text'"):
        check_contract({"a": "text"}, {"a": "x"})
