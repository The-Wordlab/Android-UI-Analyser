"""`analyze --annotate PATH` writes the marks even when the screen has not changed.

`analyze` reuses the previous payload when the hierarchy hashes the same, which is every time a
caller asks for marks on the screen `session start` or an action just returned. The reuse path
returned before the annotation step, so the command exited 0 with no file and no
`meta.annotated_image`. It also carried the previous call's `annotated_image` forward, so a
later call reported a file it had not written.

The annotation is a side effect the caller asked for, like the id cache: reusing the tree must
not skip it.
"""

from __future__ import annotations

from pathlib import Path

from android_ui_analyser.engine import Engine
from conftest import FakeDevice
from test_memory import APPS, P, _engine


def _engine_on_a_still_screen(tmp_path: Path, serial: str) -> Engine:
    return _engine(tmp_path, FakeDevice(hierarchy_xml=APPS, package=P, serial=serial))


def test_annotate_on_an_unchanged_frame_writes_the_file(tmp_path: Path) -> None:
    engine = _engine_on_a_still_screen(tmp_path, "emu-annotate-still")
    engine.analyze(source="hierarchy")
    marks = tmp_path / "marks.png"

    again = engine.analyze(source="hierarchy", annotate=str(marks))

    assert again.meta.via == "hierarchy-unchanged", "an identical tree must take the reuse path"
    assert marks.is_file(), "the caller named a file, so it has to exist afterwards"
    assert again.meta.annotated_image == str(marks)


def test_reuse_does_not_report_an_annotation_this_call_did_not_write(tmp_path: Path) -> None:
    engine = _engine_on_a_still_screen(tmp_path, "emu-annotate-stale")
    earlier = tmp_path / "earlier.png"
    annotated = engine.analyze(source="hierarchy", annotate=str(earlier))
    assert annotated.meta.annotated_image == str(earlier), "a first analyze writes its marks"

    plain = engine.analyze(source="hierarchy")

    assert plain.meta.via == "hierarchy-unchanged"
    assert plain.meta.annotated_image is None, "this call was not asked for marks and wrote none"
