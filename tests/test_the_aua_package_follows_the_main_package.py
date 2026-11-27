"""`aua` on PyPI is a second name for android-ui-analyser, not a second product.

It is written from the main `pyproject.toml` at release time. If it could name a different
version, `uvx aua` and `uvx android-ui-analyser` would start different tools while both claim to
be the current release.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[1]


def _builder() -> ModuleType:
    path = REPO / "scripts/build-aua-package.py"
    spec = importlib.util.spec_from_file_location("build_aua_package", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_aua_installs_the_same_version_of_the_main_package() -> None:
    main = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    version = main["project"]["version"]

    alias = tomllib.loads(_builder().alias_pyproject(main))["project"]

    assert alias["name"] == "aua"
    assert alias["version"] == version
    assert alias["dependencies"] == [f"android-ui-analyser=={version}"]
    assert alias["scripts"] == {"aua": main["project"]["scripts"]["aua"]}


def test_every_extra_is_passed_through_to_the_main_package() -> None:
    main = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    version = main["project"]["version"]

    alias = tomllib.loads(_builder().alias_pyproject(main))["project"]

    expected = {
        extra: [f"android-ui-analyser[{extra}]=={version}"]
        for extra in main["project"]["optional-dependencies"]
    }
    assert alias["optional-dependencies"] == expected
