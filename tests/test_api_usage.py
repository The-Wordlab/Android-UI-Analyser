"""`aua api`: which client versions call a backend operation, and what a spec change touches."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from android_ui_analyser.api_usage import check, load_config, usage
from android_ui_analyser.errors import UsageError


def _op(fields: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": fields or {}}
    if required:
        schema["required"] = required
    return {"responses": {"200": {"content": {"application/json": {"schema": schema}}}}}


def _spec(
    paths: dict[str, dict[str, Any]], components: dict[str, Any] | None = None
) -> dict[str, Any]:
    spec: dict[str, Any] = {"openapi": "3.1.0", "paths": paths}
    if components:
        spec["components"] = {"schemas": components}
    return spec


def _git(repo: Path, *args: str, date: str | None = None) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com"}
    env |= {"GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    if date:
        env |= {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
    ).stdout


def _repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    return path


def _commit(repo: Path, files: dict[str, str], date: str, tag: str | None = None) -> None:
    for name, text in files.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"at {date}", date=date)
    if tag:
        _git(repo, "tag", tag, date=date)


def _retrofit(*calls: str) -> str:
    lines = [
        f'    @{c.split()[0]}("{c.split()[1]}")\n    suspend fun f{i}(): Any\n'
        for i, c in enumerate(calls)
    ]
    return "interface ExampleApi {\n" + "".join(lines) + "}\n"


def _write(path: Path, data: Any) -> str:
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)
    return str(path)


def _client_history(root: Path) -> Path:
    """1.0 calls `legacy`; 2.0 drops it and adds an order call; the branch adds item detail."""

    app = _repo(root / "example-android")
    _commit(
        app,
        {
            "app/build.gradle.kts": 'versionName = "1.0.0"\n',
            "data/Api.kt": _retrofit("GET items", "GET legacy"),
            "data/test/FakeApi.kt": _retrofit("GET only-in-tests"),
        },
        "2026-01-01T00:00:00",
        tag="(1)",
    )
    _commit(
        app,
        {
            "app/build.gradle.kts": 'versionName = "2.0.0"\n',
            "data/Api.kt": _retrofit("GET items", "POST orders"),
        },
        "2026-02-01T00:00:00",
        tag="(2)",
    )
    _commit(
        app,
        {"data/Api.kt": _retrofit("GET items", "POST orders", "GET items/{itemId}")},
        "2026-03-01T00:00:00",
    )
    return app


def _config(root: Path, app: Path, backend: Path | None = None, **backend_keys: Any) -> str:
    backend_repo = backend or _repo(root / "example-backend")
    lines = [
        "backend:",
        f"  repo: {backend_repo}",
        *[f"  {k}: {v}" for k, v in backend_keys.items()],
        "clients:",
        "  - name: android",
        f"    repo: {app}",
        "    scanner: retrofit",
        "    base_path: /api/v1/",
        "    branch: main",
        "    exclude: ['*/test/*']",
        "    tags: '(*)'",
        '    version: {file: app/build.gradle.kts, pattern: \'versionName = "([^"]+)"\'}',
    ]
    return _write(root / "aua-api.yaml", "\n".join(lines) + "\n")


def _labels(user: dict[str, Any]) -> list[str]:
    return [v["version"] for v in user["versions"]]


def _item_fields(*names: str) -> dict[str, Any]:
    return _op({n: {"type": "string"} for n in names}, required=list(names))


def test_a_change_is_reported_beside_exactly_the_versions_that_call_it(tmp_path):
    app = _client_history(tmp_path)
    config = _config(tmp_path, app)
    base = _spec(
        {
            "/api/v1/items": {"get": _item_fields("name", "price")},
            "/api/v1/items/{item_id}": {"get": _op()},
            "/api/v1/legacy": {"get": _op()},
            "/api/v1/orders": {"post": _op()},
            "/api/v1/unused": {"get": _item_fields("x")},
        }
    )
    head = _spec(
        {
            "/api/v1/items": {"get": _item_fields("price")},
            "/api/v1/items/{item_id}": {"get": _op()},
            "/api/v1/orders": {"post": _op()},
            "/api/v1/unused": {"get": _op()},
        }
    )

    result = check(
        config,
        tmp_path / "cache",
        base_spec=_write(tmp_path / "base.json", base),
        head_spec=_write(tmp_path / "head.json", head),
        fetch=False,
    )

    by_change = {(f["operations"][0]["operation"], f["change"]): f for f in result["findings"]}
    removed = by_change[("GET /api/v1/legacy", "operation_removed")]
    assert [_labels(u) for u in removed["used_by"]] == [["1.0.0"]]
    field = by_change[("GET /api/v1/items", "response_field_removed")]
    assert field["field"] == "name"
    [user] = field["used_by"]
    assert _labels(user) == ["unreleased", "2.0.0", "1.0.0"]
    # Every version carries its own sources and commit: versions differ, none stands for another.
    assert [v["sources"] for v in user["versions"]] == [["data/Api.kt:2"]] * 3
    assert all(len(v["sha"]) == 40 for v in user["versions"])
    # A removed operation outranks a field change, and nothing the apps never call is a finding.
    assert result["findings"][0]["change"] == "operation_removed"
    assert result["unused_changes"] == ["GET /api/v1/unused: response_field_removed `x`"]
    assert "agent_brief" in result
    [android] = result["clients"]
    assert [v["label"] for v in android["versions"]] == ["unreleased", "2.0.0", "1.0.0"]
    assert android["unmatched_calls"]["count"] == 0  # the test-only call was excluded


def test_a_second_run_reads_no_file_it_has_already_scanned(tmp_path):
    app = _client_history(tmp_path)
    config = _config(tmp_path, app)
    spec = _write(tmp_path / "spec.json", _spec({"/api/v1/items": {"get": _op()}}))
    args = {"base_spec": spec, "head_spec": spec, "fetch": False}

    first = check(config, tmp_path / "cache", **args)
    second = check(config, tmp_path / "cache", **args)

    assert first["clients"][0]["files_scanned_now"] > 0
    assert second["clients"][0]["files_scanned_now"] == 0
    assert second["clients"][0]["versions"] == first["clients"][0]["versions"]


def test_history_starts_at_the_version_floor_and_keeps_the_newest_versions(tmp_path):
    app = _client_history(tmp_path)
    _commit(
        app, {"app/build.gradle.kts": 'versionName = "3.0.0"\n'}, "2026-04-01T00:00:00", tag="(3)"
    )
    config = Path(_config(tmp_path, app))
    config.write_text(config.read_text() + "    min_version: '2.0.0'\n    max_versions: 1\n")

    result = usage(config, tmp_path / "cache", fetch=False)

    assert [v["label"] for v in result["clients"][0]["versions"]] == ["unreleased", "3.0.0"]


def test_a_pattern_reads_a_method_and_path_written_on_different_lines(tmp_path):
    app = _repo(tmp_path / "example-ios")
    _commit(
        app,
        {
            "Net/ThingEndpoint.swift": (
                "case .delete(let id):\n"
                '    Builder()\n        .path("things/\\(id)")\n        .method(.DELETE)\n'
            )
        },
        "2026-01-01T00:00:00",
    )
    config = _write(
        tmp_path / "aua-api.yaml",
        f"backend: {{repo: {app}}}\n"
        "clients:\n"
        f"  - {{name: ios, repo: {app}, branch: main, files: ['*.swift'], base_path: /api/v1/,\n"
        '      pattern: \'\\.path\\("(?P<path>[^"]+)"\\)\\s*\\.method\\(\\.(?P<method>[A-Z]+)\\)\'}\n',
    )

    result = usage(config, tmp_path / "cache", fetch=False)

    assert result["calls"] == [
        {
            "call": "DELETE /api/v1/things/{}",
            "clients": {
                "ios": {"versions": ["unreleased"], "sources": ["Net/ThingEndpoint.swift:3"]}
            },
        }
    ]


def test_the_base_spec_is_exported_from_its_ref_without_touching_the_backend_checkout(tmp_path):
    app = _client_history(tmp_path)
    backend = _repo(tmp_path / "example-backend")
    exporter = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "fields = Path('fields.txt').read_text().split()\n"
        "props = {f: {'type': 'string'} for f in fields}\n"
        "op = {'responses': {'200': {'content': {'application/json': {'schema':\n"
        "    {'type': 'object', 'properties': props, 'required': fields}}}}}}\n"
        "Path(sys.argv[1]).write_text(json.dumps({'paths': {'/api/v1/items': {'get': op}}}))\n"
    )
    _commit(backend, {"export.py": exporter, "fields.txt": "name price\n"}, "2026-03-01T00:00:00")
    (backend / "fields.txt").write_text("price\n")  # the uncommitted change under review
    config = _config(
        tmp_path, app, backend, base="main", spec_command=f"{sys.executable} export.py {{out}}"
    )

    result = check(config, tmp_path / "cache", fetch=False)

    [finding] = result["findings"]
    assert (finding["change"], finding["field"]) == ("response_field_removed", "name")
    assert result["backend"]["head"]["dirty"] is True
    assert (backend / "fields.txt").read_text() == "price\n"
    assert _git(backend, "worktree", "list").count("\n") == 1
    sha = _git(backend, "rev-parse", "main").strip()
    assert list((tmp_path / "cache" / "api-usage" / "specs").glob(f"{sha}-*.json"))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("backend: {repo: .}\nclients: []\n", "at least one client"),
        ("backend: {repo: .}\nclients: [{name: a, repo: ., scanner: nope}]\n", "unknown scanner"),
        (
            "backend: {repo: .}\nclients: [{name: a, repo: ., files: ['*'], pattern: 'x'}]\n",
            "`path`",
        ),
    ],
)
def test_a_config_mistake_is_a_usage_error_that_says_what_to_fix(tmp_path, text, message):
    with pytest.raises(UsageError, match=message):
        load_config(_write(tmp_path / "aua-api.yaml", text))


def test_a_missing_config_shows_an_example(tmp_path):
    with pytest.raises(UsageError) as raised:
        load_config(tmp_path / "absent.yaml")
    assert "spec_command" in (raised.value.hint or "")


def test_retrofit_reads_the_http_annotation_whatever_order_its_arguments_are_in(tmp_path):
    # A shipped client deleted threads through `@HTTP(method = "DELETE", …)`, unseen until now.
    app = _repo(tmp_path / "example-android")
    api = (
        "interface ExampleApi {\n"
        '    @HTTP(method = "DELETE", path = "threads", hasBody = true)\n'
        "    suspend fun a(@Body ids: Ids): Any\n"
        '    @HTTP(path = "items/{itemId}", method = "PATCH")\n'
        "    suspend fun b(): Any\n"
        "}\n"
    )
    _commit(app, {"data/Api.kt": api}, "2026-01-01T00:00:00")
    config = _write(
        tmp_path / "aua-api.yaml",
        f"backend: {{repo: {app}}}\nclients:\n"
        f"  - {{name: android, repo: {app}, branch: main, scanner: retrofit, base_path: /api/v1/}}\n",
    )

    result = usage(config, tmp_path / "cache", fetch=False)

    assert [c["call"] for c in result["calls"]] == [
        "DELETE /api/v1/threads",
        "PATCH /api/v1/items/{}",
    ]


def test_a_cached_base_spec_is_exported_again_when_its_exporter_changes_or_on_refresh(tmp_path):
    app = _client_history(tmp_path)
    backend = _repo(tmp_path / "example-backend")
    environment = tmp_path / "exporter-environment.txt"  # stands in for installed dependencies
    environment.write_text("name\n")
    exporter = (
        "import json, sys\n"
        "from pathlib import Path\n"
        f"fields = Path({str(environment)!r}).read_text().split() + sys.argv[2:]\n"
        "props = {f: {'type': 'string'} for f in fields}\n"
        "op = {'responses': {'200': {'content': {'application/json': {'schema':\n"
        "    {'type': 'object', 'properties': props}}}}}}\n"
        "Path(sys.argv[1]).write_text(json.dumps({'paths': {'/api/v1/items': {'get': op}}}))\n"
    )
    _commit(backend, {"export.py": exporter}, "2026-03-01T00:00:00")
    head = _write(tmp_path / "head.json", _spec({"/api/v1/items": {"get": _op()}}))

    def removed(extra: str = "", refresh: bool = False) -> list[str]:
        command = f"{sys.executable} export.py {{out}}{extra}"
        config = _config(tmp_path, app, backend, base="main", spec_command=command)
        result = check(config, tmp_path / "cache", head_spec=head, fetch=False, refresh=refresh)
        return sorted(f["field"] for f in result["findings"])

    assert removed() == ["name"]
    environment.write_text("name price\n")
    assert removed() == ["name"]  # same commit, same exporter: the cached export stands
    assert removed(refresh=True) == ["name", "price"]
    assert removed(extra=" extra") == ["extra", "name", "price"]  # a new exporter is a new export


def test_one_change_to_a_shared_model_is_one_finding_with_every_operation_under_it(tmp_path):
    # Found on a real backend: one renamed field of a shared model read as 35 separate findings.
    app = _repo(tmp_path / "example-android")
    _commit(app, {"data/Api.kt": _retrofit("GET score", "POST chat")}, "2026-01-01T00:00:00")
    config = _write(
        tmp_path / "aua-api.yaml",
        f"backend: {{repo: {app}}}\nclients:\n"
        f"  - {{name: android, repo: {app}, branch: main, scanner: retrofit, base_path: /api/v1/}}\n",
    )

    def spec(score: dict[str, Any]) -> dict[str, Any]:
        def body(name: str) -> dict[str, Any]:
            schema = {"$ref": f"#/components/schemas/{name}"}
            return {"responses": {"200": {"content": {"application/json": {"schema": schema}}}}}

        reply = {"type": "object", "properties": {"score": {"$ref": "#/components/schemas/Score"}}}
        return _spec(
            {"/api/v1/score": {"get": body("Score")}, "/api/v1/chat": {"post": body("Reply")}},
            {"Score": {"type": "object", "properties": score}, "Reply": reply},
        )

    result = check(
        config,
        tmp_path / "cache",
        base_spec=_write(tmp_path / "base.json", spec({"points": {"type": "integer"}})),
        head_spec=_write(tmp_path / "head.json", spec({})),
        fetch=False,
    )

    [finding] = result["findings"]
    assert (finding["change"], finding["schema"], finding["field"]) == (
        "response_field_removed",
        "Score",
        "points",
    )
    assert finding["operations"] == [
        {"operation": "GET /api/v1/score", "field": "points"},
        {"operation": "POST /api/v1/chat", "field": "score.points"},
    ]
    [user] = finding["used_by"]
    assert user["versions"][0]["sources"] == [
        "data/Api.kt:2 (GET /api/v1/score)",
        "data/Api.kt:4 (POST /api/v1/chat)",
    ]


def test_the_result_says_what_the_check_could_not_see(tmp_path):
    app = _client_history(tmp_path)
    config = Path(_config(tmp_path, app))
    config.write_text(config.read_text() + "    max_versions: 1\n")
    untyped = {"responses": {"200": {"content": {"application/json": {"schema": {}}}}}}
    spec = _write(
        tmp_path / "spec.json",
        _spec({"/api/v1/items": {"get": untyped}, "/api/v1/orders": {"post": _item_fields("id")}}),
    )

    result = check(config, tmp_path / "cache", base_spec=spec, head_spec=spec, fetch=False)

    coverage = result["coverage"]
    # Only removing `GET items` can show there: its body is not described.
    assert coverage["untyped_called_operations"] == {
        "count": 1,
        "operations": ["GET /api/v1/items"],
    }
    assert coverage["operations"] == {"base": 2, "called": 2}
    assert coverage["not_checked"]
    [android] = result["clients"]
    assert [v["label"] for v in android["versions"]] == ["unreleased", "2.0.0"]
    assert android["history"] == {
        "max_versions": 1,
        "min_version": None,
        "tags_not_examined": 1,
        "tags_below_min_version": 0,
    }
