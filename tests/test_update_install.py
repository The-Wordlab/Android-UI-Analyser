"""`aua update --install`: upgrade a uv tool install in place, and say what to run otherwise."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import android_ui_analyser.release_install as release_install
from android_ui_analyser import __version__
from android_ui_analyser.cli import app
from android_ui_analyser.errors import UsageError
from android_ui_analyser.release_check import UpdateStatus
from android_ui_analyser.release_install import Install, detect, install

RECEIPT = """\
[tool]
requirements = [
    { name = "android-ui-analyser", extras = ["apple", "web"], git = "https://example.invalid/aua.git?tag=v0.1.0" },
    { name = "mitmproxy" },
    { name = "onnxruntime", specifier = ">=1.17" },
]
entrypoints = [
    { name = "aua", install-path = "/home/me/.local/bin/aua", from = "android-ui-analyser" },
]
"""


def _package_dir(tmp_path: Path) -> Path:
    """A package directory that is not inside a git clone."""
    package = tmp_path / "site-packages" / "android_ui_analyser"
    package.mkdir(parents=True)
    return package


def test_a_uv_tool_install_keeps_its_extras_and_with_packages(tmp_path: Path) -> None:
    (tmp_path / "uv-receipt.toml").write_text(RECEIPT, encoding="utf-8")

    found = detect(prefix=str(tmp_path), package_dir=_package_dir(tmp_path))

    assert found == Install(
        "uv-tool",
        str(tmp_path),
        extras=("apple", "web"),
        with_packages=("mitmproxy", "onnxruntime>=1.17"),
        entrypoint="/home/me/.local/bin/aua",
    )


def test_an_editable_install_from_a_clone_is_a_clone(tmp_path: Path) -> None:
    repo = tmp_path / "aua"
    (repo / ".git").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    package = repo / "src" / "android_ui_analyser"
    package.mkdir(parents=True)

    assert detect(prefix=str(tmp_path / "venv"), package_dir=package) == Install("clone", str(repo))


@pytest.mark.parametrize(
    ("prefix", "kind"),
    [
        ("/home/me/.local/pipx/venvs/android-ui-analyser", "pipx"),
        ("/home/me/.cache/uv/archive-v0/abc123", "uvx"),
        ("/opt/venvs/custom", "other"),
    ],
)
def test_other_installs_are_recognised_by_their_environment(
    tmp_path: Path, prefix: str, kind: str
) -> None:
    assert detect(prefix=prefix, package_dir=_package_dir(tmp_path)).kind == kind


def _latest(monkeypatch: pytest.MonkeyPatch, tag: str = "v99.0.0") -> None:
    status = UpdateStatus(
        installed=__version__,
        latest=tag[1:],
        update_available=True,
        tag=tag,
        published_at=None,
        release_url=None,
        notes=None,
        checked_at="2026-01-01T00:00:00+00:00",
        from_cache=False,
        error=None,
    )
    monkeypatch.setattr(release_install.release_check, "check_for_update", lambda **_: status)


def _runs(monkeypatch: pytest.MonkeyPatch, returncode: int = 0) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode, "", "resolver said no")

    monkeypatch.setattr(release_install.subprocess, "run", fake_run)
    return calls


def test_a_uv_tool_install_is_reinstalled_and_its_existing_skills_refreshed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _latest(monkeypatch)
    tool = Install("uv-tool", "/env", ("apple",), ("mitmproxy",), "/bin/aua")
    monkeypatch.setattr(release_install, "detect", lambda: tool)
    monkeypatch.setattr(release_install, "_uv", lambda: "/bin/uv")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    claude_skill = tmp_path / ".claude/skills/android-ui-analyser/SKILL.md"
    claude_skill.parent.mkdir(parents=True)
    claude_skill.write_text("old", encoding="utf-8")
    calls = _runs(monkeypatch)

    result = install()

    assert result["action"] == "installed" and result["ok"] is True
    assert calls[0] == [
        "/bin/uv",
        "tool",
        "install",
        "--force",
        "--with",
        "mitmproxy",
        "android-ui-analyser[apple] @ git+https://github.com/The-Wordlab/Android-UI-Analyser.git@v99.0.0",
    ]
    # Only the skill that already existed is rewritten, by the newly installed aua.
    assert calls[1:] == [["/bin/aua", "guide", "--emit-skill", str(claude_skill)]]
    assert result["skills_refreshed"] == [str(claude_skill)]
    assert not (tmp_path / ".agents").exists()


def test_a_clone_is_never_switched_only_told_what_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _latest(monkeypatch, "v99.1.0")
    monkeypatch.setattr(release_install, "detect", lambda: Install("clone", "/src/aua"))
    calls = _runs(monkeypatch)

    result = install()

    assert calls == []
    assert result["action"] == "manual"
    assert result["commands"] == [
        "git -C /src/aua fetch --tags && git -C /src/aua checkout v99.1.0 && /src/aua/install.sh"
    ]


def test_a_failed_uv_install_reports_why_and_what_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _latest(monkeypatch)
    monkeypatch.setattr(release_install, "detect", lambda: Install("uv-tool", "/env", ()))
    monkeypatch.setattr(release_install, "_uv", lambda: "/bin/uv")
    _runs(monkeypatch, returncode=2)

    result = install()

    assert result["ok"] is False
    assert result["error"] == "resolver said no"
    assert result["commands"] == [
        "uv tool install --force 'android-ui-analyser @ git+https://github.com/The-Wordlab/"
        "Android-UI-Analyser.git@v99.0.0'"
    ]


def test_nothing_happens_when_this_is_already_the_latest(monkeypatch: pytest.MonkeyPatch) -> None:
    status = UpdateStatus(
        __version__, __version__, False, f"v{__version__}", None, None, None, "t", False, None
    )
    monkeypatch.setattr(release_install.release_check, "check_for_update", lambda **_: status)
    calls = _runs(monkeypatch)

    assert install()["action"] == "none"
    assert calls == []


def test_a_chosen_version_must_be_a_published_release(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def exists(tag: str, **_: Any) -> bool:
        asked.append(tag)
        return tag == "v0.30.0"

    monkeypatch.setattr(release_install.release_check, "release_exists", exists)
    monkeypatch.setattr(release_install, "detect", lambda: Install("clone", "/src/aua"))

    assert install("0.30.0")["commands"][0].endswith("checkout v0.30.0 && /src/aua/install.sh")
    with pytest.raises(UsageError, match="no published release v0.29.9"):
        install("v0.29.9")
    with pytest.raises(UsageError, match="not a version"):
        install("latest-ish")
    assert asked == ["v0.30.0", "v0.29.9"]


def test_the_cli_exit_codes_tell_installed_from_run_this(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    outcomes = iter(
        [
            {"ok": True, "action": "installed", "message": "done"},
            {"ok": True, "action": "manual", "message": "m", "commands": ["x"]},
            {"ok": False, "installed": "0.1.0", "error": "offline"},
        ]
    )
    monkeypatch.setattr(release_install, "install", lambda version=None: next(outcomes))

    assert runner.invoke(app, ["update", "--install"]).exit_code == 0
    manual = runner.invoke(app, ["update", "--install", "--json"])
    assert manual.exit_code == 10 and json.loads(manual.output)["commands"] == ["x"]
    assert runner.invoke(app, ["update", "--install"]).exit_code == 1
    assert runner.invoke(app, ["update", "--version", "0.1.0"]).exit_code == 2
