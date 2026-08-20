"""The committed helper APK must be the one the committed Java produces.

The APK is checked in as a binary (``src/android_ui_analyser/data/aua-helper.apk``) so that
installing AUA needs no Android SDK. That convenience has a failure mode with no natural
symptom: edit the Java, forget ``helper/tools/build.py``, and every test still passes while
devices run the previous build. It has happened twice. The only prior check was that the file
starts with ``PK`` — that it is a zip at all.

So the APK carries a stamp of the sources it was built from, and this recomputes it. Reading
the stamp needs nothing but ``zipfile``, so the guard works for contributors who cannot build
the APK themselves; it simply tells them a rebuild is owed.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "helper"
APK = REPO / "src" / "android_ui_analyser" / "data" / "aua-helper.apk"

sys.path.insert(0, str(HELPER / "tools"))

pytestmark = pytest.mark.skipif(
    not HELPER.is_dir(),
    reason="no helper/ sources in this checkout — nothing to compare the APK against",
)

REBUILD = "helper/tools/build.py"


def test_the_committed_apk_was_built_from_the_committed_sources() -> None:
    from source_digest import STAMP_ENTRY, helper_source_digest

    assert APK.is_file(), f"the bundled helper APK is missing; rebuild it with {REBUILD}"
    assert APK.read_bytes()[:2] == b"PK", "bundled helper APK is not a zip archive"

    with zipfile.ZipFile(APK) as archive:
        names = set(archive.namelist())
        assert STAMP_ENTRY in names, (
            f"the bundled APK carries no source stamp, so nothing can tell whether it matches "
            f"helper/. Rebuild it with `{REBUILD}`."
        )
        stamped = archive.read(STAMP_ENTRY).decode("ascii").strip()

    actual = helper_source_digest(HELPER)
    assert stamped == actual, (
        "the bundled helper APK is older than the Java beside it: it was built from sources "
        f"hashing {stamped[:12]}…, but helper/ now hashes {actual[:12]}…. Devices would run "
        f"the previous build. Rebuild and re-copy it with `{REBUILD}`."
    )


def test_the_digest_notices_a_changed_source_file(tmp_path: Path) -> None:
    """A guard that cannot go red is not a guard, so prove the digest actually moves."""

    from source_digest import helper_source_digest

    fake = tmp_path / "helper"
    (fake / "app" / "src" / "main" / "java" / "dev" / "aua").mkdir(parents=True)
    java = fake / "app" / "src" / "main" / "java" / "dev" / "aua" / "Thing.java"
    java.write_text("class Thing {}\n")
    (fake / "app" / "src" / "main" / "AndroidManifest.xml").write_text("<manifest/>\n")

    before = helper_source_digest(fake)
    java.write_text("class Thing { void added() {} }\n")
    after = helper_source_digest(fake)

    assert before != after, "editing a source file left the digest unchanged"


def test_the_digest_ignores_where_the_checkout_lives(tmp_path: Path) -> None:
    """Two clones of the same commit must agree, or the guard fails for the wrong reason."""

    from source_digest import helper_source_digest

    digests = []
    for name in ("clone-a", "some/deeper/clone-b"):
        root = tmp_path / name / "helper"
        (root / "app" / "src" / "main" / "java").mkdir(parents=True)
        (root / "app" / "src" / "main" / "java" / "A.java").write_text("class A {}\n")
        (root / "app" / "src" / "main" / "AndroidManifest.xml").write_text("<manifest/>\n")
        digests.append(helper_source_digest(root))

    assert digests[0] == digests[1], "the digest depends on the checkout path"
