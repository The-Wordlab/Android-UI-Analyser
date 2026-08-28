"""The courtesy update check must never turn somebody else's CI job into an outage.

It runs without a device on automation that may be offline, rate-limited, or behind a shared
cache. Every expected transport failure therefore becomes data and an exit code, never a traceback.
"""

from __future__ import annotations

import json
import urllib.error
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import android_ui_analyser.release_check as release_check
from android_ui_analyser import __version__
from android_ui_analyser.cli import app

runner = CliRunner()


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


@pytest.fixture(autouse=True)
def isolated_release_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUA_CACHE__DIR", str(tmp_path / "release-cache"))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)


def _github_release(tag: str) -> dict[str, str]:
    return {
        "tag_name": tag,
        "published_at": "2026-08-28T10:00:00Z",
        "html_url": f"https://example.test/releases/{tag}",
        "body": "A useful release note.",
    }


@pytest.mark.parametrize(
    ("tag", "available"),
    [("v999.0.0", True), (f"v{__version__}", False), ("v0.0.1", False)],
)
def test_new_same_and_older_releases_compare_correctly(
    tag: str, available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(release_check, "_fetch_latest", lambda _timeout: _github_release(tag))

    status = release_check.check_for_update()

    assert status.latest == tag.removeprefix("v")
    assert status.update_available is available
    assert status.error is None


@pytest.mark.parametrize(
    "failure",
    [urllib.error.URLError("offline"), TimeoutError("timed out")],
)
def test_transport_failures_are_returned_not_raised(
    failure: OSError, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_timeout: float) -> dict[str, Any]:
        raise failure

    monkeypatch.setattr(release_check, "_fetch_latest", fail)

    status = release_check.check_for_update()

    assert status.latest is None
    assert status.error


@pytest.mark.parametrize(
    ("code", "message"),
    [(403, "rate limit"), (404, "no releases")],
)
def test_github_http_failures_are_returned_not_raised(
    code: int, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = urllib.error.HTTPError(release_check.LATEST_RELEASE_URL, code, "failed", None, None)

    def fail(_timeout: float) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(release_check, "_fetch_latest", fail)

    status = release_check.check_for_update()

    assert status.latest is None
    assert status.error and message in status.error


def test_malformed_json_is_returned_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        release_check.urllib.request,
        "urlopen",
        lambda _request, timeout: _Response(b"not json"),
    )

    status = release_check.check_for_update()

    assert status.latest is None
    assert status.error and "could not reach GitHub" in status.error


def test_the_release_cache_avoids_repeat_calls_and_force_bypasses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fetch(_timeout: float) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return _github_release("v999.0.0")

    monkeypatch.setattr(release_check, "_fetch_latest", fetch)

    first = release_check.check_for_update()
    second = release_check.check_for_update()
    forced = release_check.check_for_update(force=True)

    assert calls == 2
    assert first.from_cache is False
    assert second.from_cache is True
    assert forced.from_cache is False


@pytest.mark.parametrize(
    ("left", "right", "is_less"),
    [
        ("0.9.0", "0.13.0", True),
        ("0.14.0-rc1", "0.14.0", True),
        ("0.14.0-rc1", "0.14.0-rc2", True),
        ("1.0.0", "0.99.99", False),
    ],
)
def test_versions_use_semver_precedence(left: str, right: str, is_less: bool) -> None:
    left_key = release_check._version_key(left)
    right_key = release_check._version_key(right)
    assert left_key is not None and right_key is not None
    assert (left_key < right_key) is is_less


def test_non_semver_release_tags_are_reported_instead_of_guessed() -> None:
    status = release_check._status(
        _github_release("v1.2.3.4"), checked_at="2026-08-28T10:00:00+00:00", from_cache=False
    )
    assert status.error == f"cannot compare version {__version__} with release 1.2.3.4"
    assert status.update_available is False


def _status(*, update_available: bool = False, error: str | None = None) -> release_check.UpdateStatus:
    return release_check.UpdateStatus(
        installed=__version__,
        latest=None if error else ("999.0.0" if update_available else __version__),
        update_available=update_available,
        tag=None if error else ("v999.0.0" if update_available else f"v{__version__}"),
        published_at=None,
        release_url=None,
        notes=None,
        checked_at="2026-08-28T10:00:00+00:00",
        from_cache=False,
        error=error,
    )


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [(_status(), 0), (_status(update_available=True), 10), (_status(error="offline"), 1)],
)
def test_the_cli_exit_code_is_an_automation_contract(
    status: release_check.UpdateStatus, exit_code: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(release_check, "check_for_update", lambda **_kwargs: status)

    result = runner.invoke(app, ["update", "--check", "--json"])

    assert result.exit_code == exit_code
    assert json.loads(result.stdout) == asdict(status)
