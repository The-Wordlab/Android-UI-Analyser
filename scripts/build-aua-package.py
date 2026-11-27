#!/usr/bin/env python3
"""Build `aua`, the short PyPI name for android-ui-analyser.

`aua` holds no code. It requires the same version of `android-ui-analyser` and provides the `aua`
command, so `uvx aua …` and `pip install aua` work. Everything it says about itself is read from
the main `pyproject.toml` when it is built, so the two packages cannot disagree on the version,
the extras or the links.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _people(people: list[dict[str, str]]) -> str:
    tables = []
    for person in people:
        fields = ", ".join(f"{key} = {json.dumps(value)}" for key, value in person.items())
        tables.append("{ " + fields + " }")
    return "[" + ", ".join(tables) + "]"


def alias_pyproject(main: dict[str, Any]) -> str:
    """The `aua` package's pyproject.toml, written from the main package's."""
    project = main["project"]
    name = project["name"]
    version = project["version"]

    lines = [
        "[build-system]",
        'requires = ["hatchling"]',
        'build-backend = "hatchling.build"',
        "",
        "[project]",
        'name = "aua"',
        f"version = {json.dumps(version)}",
        f"description = {json.dumps(project['description'])}",
        'readme = "README.md"',
        f"requires-python = {json.dumps(project['requires-python'])}",
        'license = { file = "LICENSE" }',
        f"authors = {_people(project['authors'])}",
        f"keywords = {json.dumps(project['keywords'])}",
        f"classifiers = {json.dumps(project['classifiers'])}",
        f"dependencies = {json.dumps([f'{name}=={version}'])}",
        "",
        "[project.optional-dependencies]",
    ]
    for extra in project["optional-dependencies"]:
        lines.append(f"{extra} = {json.dumps([f'{name}[{extra}]=={version}'])}")
    lines += ["", "[project.urls]"]
    for label, url in project["urls"].items():
        lines.append(f"{label} = {json.dumps(url)}")
    lines += [
        "",
        "[project.scripts]",
        f"aua = {json.dumps(project['scripts']['aua'])}",
        "",
        "# There is nothing to select: the wheel is metadata and the command only.",
        "[tool.hatch.build.targets.wheel]",
        "bypass-selection = true",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    out_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "dist"
    main_pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        project_dir = Path(tmp) / "aua"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").write_text(
            alias_pyproject(main_pyproject), encoding="utf-8"
        )
        for shared in ("README.md", "LICENSE"):
            shutil.copy(ROOT / shared, project_dir / shared)
        subprocess.run(["uv", "build", "--out-dir", str(out_dir), str(project_dir)], check=True)


if __name__ == "__main__":
    main()
