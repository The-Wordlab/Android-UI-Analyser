"""`install.sh` must install every extra a shipped CLI command hard-requires.

The global install is the one agents reach through PATH. `aua proxy` and `aua mock` shell out to
`mitmdump` and import `mitmproxy`, so an install that omits the `proxy` extra hands the agent a
CLI whose `--help` advertises the commands and whose `start` fails only at run time.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

INSTALL_SH = Path(__file__).resolve().parent.parent / "install.sh"

# Extras whose absence breaks a command `aua --help` still lists.
REQUIRED_EXTRAS = {"proxy"}
REQUIRED_PKGS = {"mitmproxy"}


def _feature_vars() -> tuple[str, set[str]]:
    """Evaluate install.sh's package-selection block and return (EXTRA, FEATURE_PKGS)."""
    text = INSTALL_SH.read_text()
    match = re.search(
        r'^case "\$\(uname -s\)" in.*?^FEATURE_PKGS=\([^\n]*\)$',
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match, "install.sh no longer has the case/FEATURE_PKGS selection block"
    script = match.group(0) + '\nprintf "%s\\n" "$EXTRA"\nprintf "%s\\n" "${FEATURE_PKGS[@]}"\n'
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return out[0], {p.lower() for p in out[1:] if p}


def test_venv_fallback_installs_required_extras() -> None:
    extra, _ = _feature_vars()
    names = {e.strip() for e in extra.split(",")}
    missing = REQUIRED_EXTRAS - names
    assert not missing, f"install.sh EXTRA={extra!r} omits {sorted(missing)}"


def test_global_install_injects_required_packages() -> None:
    _, pkgs = _feature_vars()
    missing = REQUIRED_PKGS - pkgs
    assert not missing, f"install.sh FEATURE_PKGS omits {sorted(missing)} (uv/pipx global install)"
