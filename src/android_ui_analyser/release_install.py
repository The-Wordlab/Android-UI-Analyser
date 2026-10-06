"""Install an AUA release over the one running: `aua update --install [--version X.Y.Z]`.

AUA reaches people in several ways, and only one can safely upgrade itself: a ``uv tool``
install of a release. That one is reinstalled at the target tag with the extras and ``--with``
packages its receipt recorded, so nothing the user added is lost. A git clone is a working tree
someone may be editing, a ``uvx`` run and the plugins take their version from the command that
starts them, and any other environment belongs to whoever built it. For those the result is the
exact command to run instead.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__, release_check
from .errors import UsageError

PACKAGE = "android-ui-analyser"
REPO_URL = "https://github.com/The-Wordlab/Android-UI-Analyser.git"
# The extras the README's install commands use, for a command printed without a receipt to copy.
DEFAULT_EXTRAS = ("apple", "rapidocr", "audio")
PLUGIN_UPDATE = "/plugin update android-ui-analyser@the-wordlab"
_UV_DIRS = ("~/.local/bin", "/opt/homebrew/bin", "/usr/local/bin", "~/.cargo/bin")


@dataclass(frozen=True)
class Install:
    """How the running ``aua`` was installed."""

    kind: str  # uv-tool | clone | uvx | pipx | other
    location: str
    extras: tuple[str, ...] = DEFAULT_EXTRAS
    with_packages: tuple[str, ...] = ()
    entrypoint: str | None = None


def detect(*, prefix: str | None = None, package_dir: Path | None = None) -> Install:
    package_dir = package_dir or Path(__file__).resolve().parent
    repo = package_dir.parents[1]
    if (repo / ".git").exists() and (repo / "pyproject.toml").is_file():
        return Install("clone", str(repo))
    env = Path(prefix or sys.prefix)
    try:
        tool = tomllib.loads((env / "uv-receipt.toml").read_text(encoding="utf-8"))["tool"]
    except (OSError, ValueError, KeyError):
        tool = None
    if isinstance(tool, dict):
        requirements = [r for r in tool.get("requirements") or [] if isinstance(r, dict)]
        main = next((r for r in requirements if r.get("name") == PACKAGE), None)
        if main is not None and not {"editable", "directory", "path"} & set(main):
            return Install(
                "uv-tool",
                str(env),
                extras=tuple(main.get("extras") or ()),
                with_packages=tuple(
                    f"{r['name']}{r.get('specifier') or ''}"
                    for r in requirements
                    if r.get("name") and r is not main
                ),
                entrypoint=next(
                    (
                        e.get("install-path")
                        for e in tool.get("entrypoints") or []
                        if isinstance(e, dict) and e.get("name") == "aua"
                    ),
                    None,
                ),
            )
    text = env.as_posix()
    if "/pipx/venvs/" in text:
        return Install("pipx", text)
    if "/uv/" in text and "/archive-v" in text:
        return Install("uvx", text)
    return Install("other", text)


def spec(tag: str, extras: tuple[str, ...]) -> str:
    named = f"{PACKAGE}[{','.join(extras)}]" if extras else PACKAGE
    return f"{named} @ git+{REPO_URL}@{tag}"


def _commands(install: Install, tag: str) -> list[str]:
    """What the user runs when AUA must not, or cannot, change this install itself."""
    target = spec(tag, install.extras)
    if install.kind == "clone":
        root = install.location
        return [f"git -C {root} fetch --tags && git -C {root} checkout {tag} && {root}/install.sh"]
    if install.kind == "uvx":
        return [f"uvx --from '{target}' aua …", PLUGIN_UPDATE]
    if install.kind == "pipx":
        return [f"pipx install --force '{target}'"]
    if install.kind == "uv-tool":
        withs = "".join(f" --with '{w}'" for w in install.with_packages)
        return [f"uv tool install --force{withs} '{target}'"]
    return [f"python -m pip install --upgrade '{target}'"]


def _uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for folder in _UV_DIRS:
        candidate = Path(folder).expanduser() / "uv"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _skill_files() -> list[tuple[Path, str]]:
    """User-level skill files `install.sh` writes, as ``(path, guide flag)``."""
    home = Path.home()
    codex_home = Path(os.environ.get("CODEX_HOME") or home / ".codex")
    out = [(home / ".claude/skills/android-ui-analyser/SKILL.md", "--emit-skill")]
    for root in (
        home / ".agents/skills/android-ui-analyser",
        codex_home / "skills/android-ui-analyser",
    ):
        out += [
            (root / "SKILL.md", "--emit-skill"),
            (root / "agents/openai.yaml", "--emit-codex-metadata"),
        ]
    return out


def _refresh_skills(aua: str) -> tuple[list[str], list[str]]:
    """Rewrite the skill files that already exist with the new version's guide; create none."""
    refreshed: list[str] = []
    failed: list[str] = []
    for path, flag in _skill_files():
        if not path.is_file():
            continue
        done = subprocess.run(
            [aua, "guide", flag, str(path)], capture_output=True, text=True, timeout=120
        )
        (refreshed if done.returncode == 0 else failed).append(str(path))
    return refreshed, failed


def _tail(text: str, lines: int = 15) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def install(version: str | None = None) -> dict[str, Any]:
    """Install the latest release, or ``version``, over this ``aua`` when it can do so safely.

    ``action`` is ``none`` (already there), ``installed``, or ``manual`` (run ``commands``).
    """
    if version:
        target = version.strip().removeprefix("v").removeprefix("V")
        if release_check.version_key(target) is None:
            raise UsageError(f"{version!r} is not a version", hint="Use one like 0.34.0.")
        tag = f"v{target}"
        found = release_check.release_exists(tag)
        if found is False:
            raise UsageError(
                f"AUA has no published release {tag}",
                hint="Releases: https://github.com/The-Wordlab/Android-UI-Analyser/releases",
            )
        if found is not True:
            return {"ok": False, "installed": __version__, "target": target, "error": found}
    else:
        status = release_check.check_for_update(force=True)
        if status.error is not None or status.tag is None:
            return {"ok": False, "installed": __version__, "error": status.error}
        target, tag = status.latest or "", status.tag
        if not status.update_available:
            return {
                "ok": True,
                "action": "none",
                "installed": __version__,
                "target": target,
                "message": f"aua {__version__} is current; the latest release is {target}",
            }
    result: dict[str, Any] = {"ok": True, "installed": __version__, "target": target, "tag": tag}
    if target == __version__:
        return {**result, "action": "none", "message": f"aua {target} is already installed"}

    found_install = detect()
    result["method"] = found_install.kind
    uv = _uv() if found_install.kind == "uv-tool" else None
    if uv is None:
        notes = {
            "clone": "This aua runs from a git clone; AUA does not switch a working tree for you.",
            "uvx": "uvx runs the version its --from names; the plugins update through their client.",
            "uv-tool": "uv is not on PATH, so AUA cannot reinstall itself.",
        }
        return {
            **result,
            "action": "manual",
            "commands": _commands(found_install, tag),
            "message": notes.get(found_install.kind, "AUA cannot tell how this aua was installed."),
        }

    argv = [uv, "tool", "install", "--force"]
    for package in found_install.with_packages:
        argv += ["--with", package]
    argv.append(spec(tag, found_install.extras))
    done = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
    if done.returncode != 0:
        return {
            **result,
            "ok": False,
            "action": "manual",
            "commands": _commands(found_install, tag),
            "error": _tail(done.stderr or done.stdout) or f"uv exited {done.returncode}",
        }
    aua = found_install.entrypoint or shutil.which("aua") or sys.argv[0]
    refreshed, failed = _refresh_skills(aua)
    result.update(action="installed", skills_refreshed=refreshed)
    if failed:
        result["skills_failed"] = failed
    result["message"] = (
        f"aua {target} installed; a warm daemon retires itself and the next command starts {target}"
    )
    return result


def format_result(result: dict[str, Any]) -> str:
    if not result.get("ok") and result.get("action") != "manual":
        return f"aua {result.get('installed')} — update not installed: {result.get('error')}"
    lines = [str(result.get("message", ""))]
    if result.get("action") == "manual":
        if result.get("error"):
            lines.append(f"uv failed: {result['error']}")
        lines += ["Run:", *(f"  {command}" for command in result.get("commands", []))]
    for path in result.get("skills_refreshed", []):
        lines.append(f"Skill refreshed: {path}")
    for path in result.get("skills_failed", []):
        lines.append(f"Skill NOT refreshed (run `aua guide --emit-skill {path}`): {path}")
    return "\n".join(line for line in lines if line)
