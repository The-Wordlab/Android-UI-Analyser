"""The mitmproxy addon, executed for real against a stub ``mitmproxy`` module.

Asserting on the addon's *source text* (what this replaced) passes for any rewrite that
keeps the same words, and fails for any that does not — it tracks the prose, not the
behaviour. The addon is the component that decides whether a device has a working network,
so it is worth running.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from android_ui_analyser import proxy_mock as pm

# --------------------------------------------------------------------------- stub mitm


class _Headers(dict):
    def get(self, key, default=None):  # mitmproxy headers are case-insensitive
        for k, v in self.items():
            if k.lower() == str(key).lower():
                return v
        return default


class _Message:
    def __init__(self, *, headers=None, text=""):
        self.headers = _Headers(headers or {})
        self._text = text

    def get_text(self, strict=True):
        return self._text

    @property
    def raw_content(self):
        return self._text.encode("utf-8", "replace")

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        self._text = value


class _Request(_Message):
    def __init__(self, method="GET", path="/", host="api.example.com", **kw):
        super().__init__(**kw)
        self.method = method
        self.path = path
        self.host = host


class _Response(_Message):
    def __init__(self, status_code=200, **kw):
        super().__init__(**kw)
        self.status_code = status_code

    @classmethod
    def make(cls, status=200, content=b"", headers=None):
        text = content.decode("utf-8") if isinstance(content, bytes) else str(content)
        return cls(status_code=status, headers=headers or {}, text=text)


class _Flow:
    def __init__(self, request, response=None):
        self.request = request
        self.response = response
        self.metadata: dict = {}


def _load(monkeypatch: pytest.MonkeyPatch, cache: Path):
    """Exec ADDON_SCRIPT against the stub and return the addon instance."""
    http_mod = types.ModuleType("mitmproxy.http")
    http_mod.Response = _Response  # type: ignore[attr-defined]
    http_mod.HTTPFlow = _Flow  # type: ignore[attr-defined]
    pkg = types.ModuleType("mitmproxy")
    pkg.http = http_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mitmproxy", pkg)
    monkeypatch.setitem(sys.modules, "mitmproxy.http", http_mod)

    cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AUA_MOCK_RULES", str(pm.rules_path(cache)))
    monkeypatch.setenv("AUA_MOCK_RECORD", str(pm.record_path(cache)))
    monkeypatch.setenv("AUA_FLOW_LOG", str(pm.flow_log_path(cache)))
    monkeypatch.setenv("AUA_FLOW_BODIES", str(pm.flow_bodies_path(cache)))
    monkeypatch.setenv("AUA_MOCK_MODE", "map")

    namespace: dict = {"__name__": "aua_mitm_addon"}
    exec(compile(pm.ADDON_SCRIPT, "aua_mitm_addon.py", "exec"), namespace)  # noqa: S102
    return namespace["addons"][0]


def _exchange(addon, *, method="GET", path="/", host="api.example.com", body="", upstream=None):
    """Drive one request/response pair through the addon; return the flow."""
    flow = _Flow(_Request(method=method, path=path, host=host, text=body))
    addon.request(flow)
    if flow.response is None:  # not stubbed — the server answered
        flow.response = upstream if upstream is not None else _Response(200, text="{}")
    addon.response(flow)
    return flow


def _rules(cache: Path, entries: list[dict], *, mode: str = "map") -> None:
    pm.save_doc(
        pm.rules_path(cache),
        {"mode": mode, "capture_bodies": True, "rules": entries},
    )


# --------------------------------------------------------------------------- passthrough


def test_no_rules_relays_untouched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    addon = _load(monkeypatch, tmp_path)
    flow = _exchange(addon, path="/v1/chat", upstream=_Response(201, text='{"real":true}'))
    assert flow.response.status_code == 201
    assert flow.response.text == '{"real":true}'
    assert flow.metadata.get("aua_action") is None


def test_unmatched_request_survives_an_armed_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One endpoint under test must not take the rest of the app's traffic with it."""
    addon = _load(monkeypatch, tmp_path)
    _rules(tmp_path, [pm.map_rule("GET", "/v1/chat", status=500, body={"stub": True})])
    flow = _exchange(addon, path="/v1/profile", upstream=_Response(200, text='{"me":1}'))
    assert flow.response.status_code == 200
    assert flow.response.text == '{"me":1}'


# --------------------------------------------------------------------------- matching


def test_path_match_is_anchored_not_substring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A rule for `/hub` must not swallow `/api/v4.0/hub` — or every other path.

    Substring matching is what let one short rule stub the whole internet, which on the
    device is indistinguishable from losing the network.
    """
    addon = _load(monkeypatch, tmp_path)
    _rules(tmp_path, [pm.map_rule("GET", "/hub", status=204)])
    assert _exchange(addon, path="/api/v4.0/hub").response.status_code == 200
    assert _exchange(addon, path="/hubbub").response.status_code == 200
    assert _exchange(addon, path="/hub").response.status_code == 204
    # A path-segment prefix still counts: /hub covers /hub/items.
    assert _exchange(addon, path="/hub/items").response.status_code == 204


def test_root_path_rule_does_not_match_everything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    addon = _load(monkeypatch, tmp_path)
    _rules(tmp_path, [pm.map_rule("GET", "/", status=418, host="only.example.com")])
    assert _exchange(addon, path="/v1/anything").response.status_code == 200


def test_host_scopes_a_rule(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    addon = _load(monkeypatch, tmp_path)
    _rules(tmp_path, [pm.map_rule("GET", "/v1/x", status=204, host="api.example.com")])
    assert _exchange(addon, path="/v1/x", host="api.example.com").response.status_code == 204
    assert _exchange(addon, path="/v1/x", host="cdn.other.com").response.status_code == 200


def test_glob_paths_and_hosts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    addon = _load(monkeypatch, tmp_path)
    _rules(tmp_path, [pm.map_rule("*", "/v1/*/items", status=204, host="*.example.com")])
    assert _exchange(addon, path="/v1/abc/items", host="api.example.com").response.status_code == 204
    assert _exchange(addon, path="/v1/abc/items", host="api.other.com").response.status_code == 200


def test_method_is_honoured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    addon = _load(monkeypatch, tmp_path)
    _rules(tmp_path, [pm.map_rule("POST", "/v1/chat", status=429)])
    assert _exchange(addon, method="GET", path="/v1/chat").response.status_code == 200
    assert _exchange(addon, method="POST", path="/v1/chat").response.status_code == 429


# --------------------------------------------------------------------------- rewrite


def test_rewrite_patches_the_real_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Charles move: the server answers, then one field of its answer changes."""
    addon = _load(monkeypatch, tmp_path)
    _rules(
        tmp_path,
        [
            pm.rewrite_rule(
                method="GET",
                path="/api/v4.0/hub",
                host="api.example.com",
                set_json={"data.items.0.title": "Renamed"},
            )
        ],
    )
    upstream = _Response(
        200, text=json.dumps({"data": {"items": [{"title": "Original", "id": 7}]}})
    )
    flow = _exchange(addon, path="/api/v4.0/hub", upstream=upstream)
    body = json.loads(flow.response.text)
    assert body["data"]["items"][0]["title"] == "Renamed"
    # Everything the rule did not name survives — that is the point over a stub.
    assert body["data"]["items"][0]["id"] == 7
    assert flow.metadata["aua_action"] == "rewrite"


def test_rewrite_can_force_a_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    addon = _load(monkeypatch, tmp_path)
    _rules(tmp_path, [pm.rewrite_rule(method="POST", path="/v1/chat", status=429)])
    flow = _exchange(addon, method="POST", path="/v1/chat", upstream=_Response(200, text="{}"))
    assert flow.response.status_code == 429


def test_rewrite_deletes_and_replaces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    addon = _load(monkeypatch, tmp_path)
    _rules(
        tmp_path,
        [
            pm.rewrite_rule(
                method="GET",
                path="/v1/me",
                delete_json=["promo"],
                replace=[("Hello", "Hola")],
            )
        ],
    )
    upstream = _Response(200, text=json.dumps({"promo": {"x": 1}, "greeting": "Hello"}))
    flow = _exchange(addon, path="/v1/me", upstream=upstream)
    body = json.loads(flow.response.text)
    assert "promo" not in body
    assert body["greeting"] == "Hola"


def test_rewrite_survives_a_non_json_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A JSON patch against HTML must leave the response alone, not break the request."""
    addon = _load(monkeypatch, tmp_path)
    _rules(tmp_path, [pm.rewrite_rule(method="GET", path="/page", set_json={"a": 1})])
    flow = _exchange(addon, path="/page", upstream=_Response(200, text="<html></html>"))
    assert flow.response.text == "<html></html>"
    assert flow.response.status_code == 200


# --------------------------------------------------------------------------- budgets


def test_times_budget_applies_then_gets_out_of_the_way(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--once` is the whole "manipulate one request" workflow."""
    addon = _load(monkeypatch, tmp_path)
    _rules(tmp_path, [pm.rewrite_rule(method="GET", path="/v1/me", status=429, times=1)])
    first = _exchange(addon, path="/v1/me", upstream=_Response(200, text="{}"))
    second = _exchange(addon, path="/v1/me", upstream=_Response(200, text="{}"))
    assert first.response.status_code == 429
    assert second.response.status_code == 200


def test_stub_times_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    addon = _load(monkeypatch, tmp_path)
    _rules(tmp_path, [pm.map_rule("GET", "/v1/me", status=204, times=2)])
    codes = [_exchange(addon, path="/v1/me").response.status_code for _ in range(3)]
    assert codes == [204, 204, 200]


# --------------------------------------------------------------------------- observability


def test_flow_log_is_written_for_every_exchange(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`await net:` must work without record mode and without any rule armed."""
    addon = _load(monkeypatch, tmp_path)
    _exchange(addon, method="POST", path="/v1/chat", upstream=_Response(200, text="{}"))
    entries = pm.read_flows_since(tmp_path, 0)
    assert [(e["method"], e["path"], e["status"]) for e in entries] == [
        ("POST", "/v1/chat", 200)
    ]
    assert entries[0]["ts"] > 0


def test_flow_log_stays_free_of_bodies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A streamed chat turn would otherwise write megabytes into the polling path."""
    addon = _load(monkeypatch, tmp_path)
    _exchange(addon, path="/v1/chat", upstream=_Response(200, text="x" * 5000))
    raw = pm.flow_log_path(tmp_path).read_text(encoding="utf-8")
    assert "xxxx" not in raw
    assert len(raw) < 500


def test_bodies_land_in_the_separate_capture_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    addon = _load(monkeypatch, tmp_path)
    _exchange(
        addon,
        method="POST",
        path="/v1/chat",
        body='{"ask":"hi"}',
        upstream=_Response(200, text='{"answer":"yo"}'),
    )
    captured = pm.read_flow_bodies(tmp_path)
    assert len(captured) == 1
    assert captured[0]["request_body"] == '{"ask":"hi"}'
    assert captured[0]["response_body"] == '{"answer":"yo"}'
    assert captured[0]["n"] == 1


def test_binary_bodies_are_not_captured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    addon = _load(monkeypatch, tmp_path)
    upstream = _Response(200, headers={"Content-Type": "image/png"}, text="\x89PNG…")
    _exchange(addon, path="/logo.png", upstream=upstream)
    assert "omitted" in pm.read_flow_bodies(tmp_path)[0]["response_body"]


def test_record_mode_captures_while_rules_still_apply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Recording a flow you are also rewriting is a normal thing to want."""
    addon = _load(monkeypatch, tmp_path)
    _rules(
        tmp_path,
        [pm.rewrite_rule(method="GET", path="/v1/me", status=429)],
        mode="record",
    )
    flow = _exchange(addon, path="/v1/me", upstream=_Response(200, text="{}"))
    assert flow.response.status_code == 429
    recorded = pm.load_record(tmp_path)
    assert len(recorded) == 1
    assert recorded[0]["request"]["path"] == "/v1/me"


def test_mode_flips_without_restarting_the_proxy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A restart drops the device's only route to the network while it rebinds."""
    addon = _load(monkeypatch, tmp_path)
    _rules(tmp_path, [], mode="map")
    _exchange(addon, path="/a")
    assert pm.load_record(tmp_path) == []

    pm.set_mode(pm.rules_path(tmp_path), "record")
    _exchange(addon, path="/b")
    assert [e["request"]["path"] for e in pm.load_record(tmp_path)] == ["/b"]
