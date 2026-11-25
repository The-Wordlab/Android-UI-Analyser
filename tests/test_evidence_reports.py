"""An agent writes the page; AUA adds the player and publishes it without ever exposing it by accident."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from android_ui_analyser.engine import Engine
from android_ui_analyser.errors import UsageError
from android_ui_analyser.mcp_server import _LEASE_FREE_TOOLS, _dispatch, _tool_definitions
from android_ui_analyser.report import add_player, guide, publish
from conftest import FakeDevice, make_config

API = "https://drophere.cc/api/v1"
SITE = "https://abc.drophere.cc/"
PAGE = "<html><body><video src='media/flow.mp4' controls></video>\n</body></html>"
REPORT_TOOLS = ("report_guide", "report_player", "report_publish")


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    (root / "media").mkdir(parents=True)
    (root / "index.html").write_text(PAGE)
    (root / "media" / "flow.mp4").write_bytes(b"mp4")
    (root / ".drophere").mkdir()
    (root / ".drophere" / "state.json").write_text("{}")
    return root


@pytest.fixture
def no_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DROPHERE_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture
def key(no_key: None, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("DROPHERE_API_KEY", "k" * 64)
    return "k" * 64


def _drophere(router: respx.MockRouter, *, visibility: str, claim: bool = False) -> None:
    created = {
        "slug": "abc",
        "versionId": "v1",
        "siteUrl": SITE,
        "uploads": [
            {"path": path, "method": "PUT", "url": f"{API}/upload/abc/v1/{path}", "headers": {}}
            for path in ("index.html", "media/flow.mp4")
        ],
    }
    if claim:
        created["claimToken"] = "ct"
    router.post(f"{API}/artifact").respond(201, json=created)
    router.patch(f"{API}/artifact/abc/access").respond(200, json={})
    router.put(url__startswith=f"{API}/upload/").respond(200)
    router.post(f"{API}/artifact/abc/finalize").respond(200, json={"siteUrl": SITE})
    router.get(f"{API}/artifact/abc").respond(200, json={"artifact": {"expiresAt": "2026-12-02"}})
    router.get(f"{API}/artifact/abc/access").respond(
        200, json={"visibility": visibility, "allowedDomains": ["example.com"]}
    )


def _site(router: respx.MockRouter, *, anonymous: int) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200 if "authorization" in request.headers else anonymous)

    for path in ("", "media/flow.mp4"):
        router.get(SITE + path).mock(side_effect=answer)


def test_the_player_goes_in_before_body_and_replaces_an_earlier_copy(tmp_path: Path) -> None:
    page = tmp_path / "index.html"
    page.write_text(PAGE)

    add_player([str(page)])
    add_player([str(page)])

    html = page.read_text()
    assert html.count("<script data-video-player>") == 1
    assert html.index("<script data-video-player>") < html.index("</body>")


def test_a_page_without_a_video_is_left_alone(tmp_path: Path) -> None:
    page = tmp_path / "notes.html"
    page.write_text("<html><body><p>No clip here.</p></body></html>")

    result = add_player([str(page)])

    assert result["pages"][0]["player"] == "skipped"
    assert "video-player" not in page.read_text()


def test_a_page_without_a_body_end_is_refused(tmp_path: Path) -> None:
    page = tmp_path / "broken.html"
    page.write_text("<video controls></video>")

    with pytest.raises(UsageError, match="</body>"):
        add_player([str(page)])


def test_access_is_restricted_before_any_file_is_uploaded(bundle: Path, key: str) -> None:
    with respx.mock(assert_all_called=False) as router:
        _drophere(router, visibility="restricted")
        _site(router, anonymous=403)

        result = publish(str(bundle), ttl="7d", allow_domains=["example.com"])
        requests = [call.request for call in router.calls]

    steps = [(request.method, request.url.path) for request in requests]
    restrict = steps.index(("PATCH", "/api/v1/artifact/abc/access"))
    first_upload = next(i for i, (method, _) in enumerate(steps) if method == "PUT")
    assert restrict < first_upload
    manifest = json.loads(requests[0].content)
    assert manifest["ttlSeconds"] == 7 * 24 * 3600
    assert [entry["path"] for entry in manifest["files"]] == ["index.html", "media/flow.mp4"]
    assert result["ok"] is True
    assert result["anonymous_status"] == 403
    assert result["files_ok"] == "2/2"
    assert result["expires_at"] == "2026-12-02"


def test_a_restricted_page_that_strangers_can_still_open_is_not_ok(
    bundle: Path, key: str
) -> None:
    with respx.mock(assert_all_called=False) as router:
        _drophere(router, visibility="restricted")
        _site(router, anonymous=200)

        result = publish(str(bundle), allow_domains=["example.com"])

    assert result["ok"] is False


def test_without_a_key_the_page_is_public_for_a_day(bundle: Path, no_key: None) -> None:
    with respx.mock(assert_all_called=False) as router:
        _drophere(router, visibility="public", claim=True)
        _site(router, anonymous=200)

        result = publish(str(bundle))
        requests = [call.request for call in router.calls]

    create = requests[0]
    finalize = next(request for request in requests if request.url.path.endswith("/finalize"))
    assert "authorization" not in create.headers
    assert "ttlSeconds" not in json.loads(create.content)
    assert json.loads(finalize.content)["claimToken"] == "ct"
    assert result["ok"] is True
    assert result["visibility"] == "public"


def test_an_allow_list_without_a_key_is_refused_before_anything_is_sent(
    bundle: Path, no_key: None
) -> None:
    with respx.mock() as router:
        with pytest.raises(UsageError, match="API key"):
            publish(str(bundle), allow_domains=["example.com"])
        assert not router.calls


def test_every_report_tool_is_published_over_mcp_and_needs_no_device() -> None:
    tools = {tool.name for tool in _tool_definitions()}
    for name in REPORT_TOOLS:
        assert name in tools
        assert name in _LEASE_FREE_TOOLS
    engine = Engine(make_config(), device=FakeDevice())
    assert _dispatch(engine, "report_guide", {})["guide"] == guide()

