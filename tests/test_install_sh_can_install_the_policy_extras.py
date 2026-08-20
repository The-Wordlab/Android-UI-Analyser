"""`install.sh` must be *able* to install the local-policy runtime — and must not by default.

The global install is the one agents reach through PATH. Its `FEATURE_PKGS` covered OCR, audio and
proxy but nothing the local policy needs, so `aua session autopilot` on a globally installed `aua`
could never load a model: every step reported "optional dependency missing" and handed off. There
was no installer path that produced a working policy at all.

The runtime is deliberately **not** in the default install (it is ~250 MB of MLX/transformers for
the small selector alone, Apple-silicon-only, and useless without a separately downloaded base
model), so this asserts both halves of the contract:

* the default install stays light — no policy extra sneaks in;
* an explicit opt-in (`--with-policy`, or `AUA_INSTALL_POLICY=1`) installs it, and installs it as
  part of the *install target* so `uv` records the extras in the tool receipt. Extras injected
  into a `uv tool` environment any other way vanish at the next `uv tool upgrade`, silently, which
  is the exact trap the autopilot error message warns about.

`--print-plan` resolves the plan and exits before touching anything, so this test never installs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO / "install.sh"

POLICY_EXTRAS = ("functiongemma", "hybrid-policy")


def _plan(*args: str, env: dict[str, str] | None = None) -> dict[str, str]:
    environ = dict(os.environ)
    environ.pop("AUA_INSTALL_POLICY", None)
    environ.update(env or {})
    proc = subprocess.run(
        ["bash", str(INSTALL_SH), "--print-plan", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env=environ,
    )
    assert proc.returncode == 0, f"--print-plan failed: {proc.returncode}\n{proc.stdout}{proc.stderr}"
    plan: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            plan[key.strip()] = value.strip()
    assert {"target", "extras", "with", "policy-extras"} <= set(plan), plan
    return plan


def _pyproject_extras() -> set[str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return set(data["project"]["optional-dependencies"])


def test_print_plan_installs_nothing() -> None:
    """The seam this test drives must be a dry run, not a very fast installer."""
    proc = subprocess.run(
        ["bash", str(INSTALL_SH), "--print-plan"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0
    assert "Setup complete" not in proc.stdout
    assert "aua doctor" not in proc.stdout


def test_default_install_stays_light() -> None:
    plan = _plan()
    assert plan["policy-extras"] in {"", "(none)"}, plan
    for extra in POLICY_EXTRAS:
        assert extra not in plan["extras"], f"default install pulls the heavy {extra} extra"
        assert extra not in plan["target"], f"default install pulls the heavy {extra} extra"


def test_opt_in_installs_the_policy_runtime() -> None:
    plan = _plan("--with-policy")
    assert "functiongemma" in plan["policy-extras"]
    assert "functiongemma" in plan["extras"], "the venv fallback must get the extra too"


def test_opt_in_rides_in_the_install_target_so_uv_records_it() -> None:
    """A `uv tool upgrade` rebuilds from the receipt; only the target spec's extras survive."""
    plan = _plan("--with-policy")
    assert plan["target"].endswith("[functiongemma]"), plan["target"]
    assert "functiongemma" not in plan["with"], (
        "policy extras must not be side-injected with --with: the receipt would not name them "
        "and the next `uv tool upgrade` would drop them again"
    )


def test_hybrid_opt_in_adds_the_reviewer_runtime() -> None:
    plan = _plan("--with-policy=hybrid")
    assert "hybrid-policy" in plan["policy-extras"]
    assert "hybrid-policy" in plan["target"] and "functiongemma" in plan["target"]


def test_env_var_opts_in_too() -> None:
    plan = _plan(env={"AUA_INSTALL_POLICY": "1"})
    assert "functiongemma" in plan["policy-extras"]


def test_every_extra_the_installer_names_exists() -> None:
    known = _pyproject_extras()
    for args in ((), ("--with-policy",), ("--with-policy=hybrid",)):
        plan = _plan(*args)
        named = {e.strip() for e in plan["extras"].split(",") if e.strip()}
        assert named <= known, f"install.sh names unknown extras {sorted(named - known)}"


def test_unknown_option_is_refused() -> None:
    """A typo'd opt-in must fail loudly, not install a policy-less CLI and claim success."""
    proc = subprocess.run(
        ["bash", str(INSTALL_SH), "--with-poilcy"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode != 0, proc.stdout


def test_the_runtime_is_apple_silicon_only_and_the_plan_says_so() -> None:
    plan = _plan("--with-policy")
    expected = sys.platform == "darwin" and os.uname().machine in {"arm64", "aarch64"}
    assert ("policy-runtime" in plan) is True
    if expected:
        assert "unsupported" not in plan["policy-runtime"]
    else:
        assert "unsupported" in plan["policy-runtime"]
