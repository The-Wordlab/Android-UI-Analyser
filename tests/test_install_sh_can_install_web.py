"""The bootstrap exposes a persistent, explicit install path for the web transport."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO / "install.sh"


def _plan(*args: str, env: dict[str, str] | None = None) -> dict[str, str]:
    environ = dict(os.environ)
    environ.pop("AUA_INSTALL_WEB", None)
    environ.update(env or {})
    proc = subprocess.run(
        ["bash", str(INSTALL_SH), "--print-plan", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env=environ,
    )
    assert proc.returncode == 0, proc.stderr
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            out[key.strip()] = value.strip()
    return out


def test_web_runtime_is_opt_in_to_keep_the_default_install_light() -> None:
    plan = _plan()
    assert plan["web-extra"] in {"", "(none)"}
    assert "web" not in plan["extras"].split(",")


def test_with_web_records_the_extra_in_every_install_path() -> None:
    plan = _plan("--with-web")
    assert plan["web-extra"] == "web"
    assert "web" in plan["extras"].split(",")
    assert plan["target"].endswith("[web]")


def test_web_and_policy_extras_compose() -> None:
    plan = _plan("--with-web", "--with-policy")
    assert plan["target"].endswith("[web,functiongemma]")
    assert {"web", "functiongemma"} <= set(plan["extras"].split(","))


def test_web_env_var_opts_in() -> None:
    assert _plan(env={"AUA_INSTALL_WEB": "true"})["web-extra"] == "web"
