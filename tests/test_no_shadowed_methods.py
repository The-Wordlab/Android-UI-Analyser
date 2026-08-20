"""No class may define the same method name twice.

Python does not consider this an error. The second definition simply replaces the first, so
the losing method still reads perfectly in the file, still has its docstring, still gets
imported — and never runs. In a class the size of ``Engine`` the two definitions are
thousands of lines apart, and nothing brings them into the same eyeline.

This was written after ``demo_record_start`` was very nearly shipped as ``record_start``,
which ``Engine`` already used for ``adb screenrecord``. The new method was defined first and
therefore silently lost; every call reached the screen-recorder instead. The only reason it
surfaced was a test that asserted the device must not be connected, and the video recorder
connects. Without that assertion the collision would have been invisible.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "android_ui_analyser"

MODULES = sorted(SRC.rglob("*.py"))


@pytest.mark.parametrize("module", MODULES, ids=lambda p: p.name)
def test_no_method_is_defined_twice_in_one_class(module: Path) -> None:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))

    duplicates: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        seen: dict[str, int] = {}
        for item in node.body:
            if not isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            # An overload or a conditional redefinition is deliberate; a decorator such as
            # @property/@x.setter legitimately reuses a name too.
            decorated = {
                ast.unparse(d).split("(")[0].split(".")[-1] for d in item.decorator_list
            }
            if decorated & {"overload", "setter", "getter", "deleter", "register"}:
                continue
            if item.name in seen:
                duplicates.append(
                    f"{node.name}.{item.name} redefined at line {item.lineno} "
                    f"(first at line {seen[item.name]}) — the first one never runs"
                )
            seen[item.name] = item.lineno

    assert not duplicates, "\n".join(duplicates)
