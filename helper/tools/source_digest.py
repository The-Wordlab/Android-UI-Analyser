"""One definition of "what the helper APK was built from", shared by the build and the test.

The committed APK (``src/android_ui_analyser/data/aua-helper.apk``) is a binary that no
reviewer reads and no test executes, so nothing used to notice when it fell behind the Java
beside it — the rebuild-and-copy step was forgotten twice, and both times the symptom was a
device running months-old behaviour while the source said otherwise.

The fix is to make the APK carry its own provenance: :func:`helper_source_digest` is stamped
into the archive at build time and recomputed at test time, so a forgotten rebuild fails
loudly instead of shipping. Both sides must hash the same bytes the same way, which is the
only reason this lives in a module of its own rather than inside either caller.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Where the stamp is written inside the APK. An asset is used rather than a BuildConfig field
# or a manifest entry because it must be readable with nothing but Python's ``zipfile`` — a
# contributor who cannot build the APK still has to be able to check one.
STAMP_ENTRY = "assets/aua-helper-source.sha256"

# Everything that can change what the APK does. Kept explicit rather than "the whole tree":
# build outputs, IDE files and the stamp itself all live under helper/ too, and hashing those
# would make the digest depend on whether someone had run gradle.
_SOURCE_GLOBS = (
    "app/src/main/java/**/*.java",
    "app/src/main/AndroidManifest.xml",
    "app/src/main/res/**/*",
    "app/build.gradle.kts",
    "build.gradle.kts",
    "settings.gradle.kts",
    "gradle.properties",
)


def helper_source_files(helper_dir: Path) -> list[Path]:
    """Every file the digest covers, sorted, so both callers walk them in one order."""

    found: set[Path] = set()
    for pattern in _SOURCE_GLOBS:
        for path in helper_dir.glob(pattern):
            if path.is_file():
                found.add(path)
    return sorted(found)


def helper_source_digest(helper_dir: Path) -> str:
    """A stable sha256 over the helper's sources, independent of checkout path and mtimes.

    Paths go into the hash as POSIX-relative strings so the digest is identical on every
    machine, and file contents are hashed as raw bytes so a line-ending change is a real
    change rather than an invisible one.
    """

    digest = hashlib.sha256()
    for path in helper_source_files(helper_dir):
        rel = path.relative_to(helper_dir).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()
