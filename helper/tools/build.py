#!/usr/bin/env python3
"""Rebuild the on-device helper APK and put it where AUA ships it from.

This exists so the rebuild is one command rather than three remembered ones. The step that
kept being forgotten was the copy, and the step that had no way of being noticed was the
stamp — so both are done here, together, and never separately.

    helper/tools/build.py            # rebuild, stamp, copy
    helper/tools/build.py --check    # just say whether the committed APK is current

Needs a JDK and an Android SDK (``ANDROID_HOME``). Contributors without either can still run
``--check``, and ``tests/test_helper_apk_matches_its_source.py`` tells them the same thing.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from source_digest import STAMP_ENTRY, helper_source_digest  # noqa: E402

HELPER = Path(__file__).resolve().parents[1]
REPO = HELPER.parent
BUILT = HELPER / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"
SHIPPED = REPO / "src" / "android_ui_analyser" / "data" / "aua-helper.apk"
# The stamp is an asset so gradle packages it verbatim and ``zipfile`` can read it back with
# no Android tooling. It is deliberately outside the globs the digest covers, so writing it
# cannot change the value it records.
STAMP_SOURCE = HELPER / "app" / "src" / "main" / "assets" / "aua-helper-source.sha256"


def stamped_digest(apk: Path) -> str | None:
    """The digest an existing APK claims it was built from, or None if it makes no claim."""

    if not apk.is_file():
        return None
    try:
        with zipfile.ZipFile(apk) as archive:
            return archive.read(STAMP_ENTRY).decode("ascii").strip()
    except (KeyError, zipfile.BadZipFile):
        return None


def check() -> int:
    want = helper_source_digest(HELPER)
    got = stamped_digest(SHIPPED)
    if got == want:
        print(f"up to date: {SHIPPED.relative_to(REPO)} matches helper/ ({want[:12]}…)")
        return 0
    where = "carries no source stamp" if got is None else f"was built from {got[:12]}…"
    print(
        f"STALE: {SHIPPED.relative_to(REPO)} {where}, but helper/ now hashes {want[:12]}….\n"
        f"Rebuild with: {Path(__file__).relative_to(REPO)}",
        file=sys.stderr,
    )
    return 1


def build() -> int:
    digest = helper_source_digest(HELPER)
    STAMP_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    STAMP_SOURCE.write_text(digest + "\n")
    print(f"stamping {digest[:12]}… into {STAMP_SOURCE.relative_to(REPO)}")

    gradlew = HELPER / "gradlew"
    result = subprocess.run(  # noqa: S603
        [str(gradlew), ":app:assembleDebug"], cwd=HELPER, check=False
    )
    if result.returncode != 0:
        print("gradle build failed; the committed APK was left alone", file=sys.stderr)
        return result.returncode
    if not BUILT.is_file():
        print(f"gradle reported success but {BUILT} is missing", file=sys.stderr)
        return 1

    # Verify before copying. A build that silently dropped the asset would otherwise ship an
    # APK that fails the guard, and the failure would point at the sources rather than here.
    if stamped_digest(BUILT) != digest:
        print(
            f"the freshly built APK does not carry the stamp — is {STAMP_ENTRY} being "
            "packaged? The committed APK was left alone.",
            file=sys.stderr,
        )
        return 1

    SHIPPED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BUILT, SHIPPED)
    size_kb = SHIPPED.stat().st_size // 1024
    print(f"copied {BUILT.relative_to(REPO)} -> {SHIPPED.relative_to(REPO)} ({size_kb} KB)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the committed APK matches helper/ without building anything",
    )
    args = parser.parse_args()
    return check() if args.check else build()


if __name__ == "__main__":
    raise SystemExit(main())
