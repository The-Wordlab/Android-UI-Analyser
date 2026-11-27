#!/usr/bin/env python3
"""Build the Chrome Web Store zip of the bundled extension.

The store sets the extension's identity itself, so the upload carries no manifest `key`; the
version follows AUA's, as it does for `aua browser extension install`.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "src/android_ui_analyser/chrome_extension"


def aua_version() -> str:
    text = (ROOT / "src/android_ui_analyser/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if match is None:
        raise SystemExit("cannot read __version__")
    return ".".join(re.findall(r"\d+", match.group(1))[:4])


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist"
    version = aua_version()
    manifest = json.loads((EXTENSION / "manifest.json").read_text(encoding="utf-8"))
    manifest.pop("key", None)
    manifest["version"] = version
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"aua-chrome-extension-{version}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        for path in sorted(EXTENSION.rglob("*")):
            if path.is_file() and path.name != "manifest.json" and path.suffix != ".svg":
                archive.write(path, path.relative_to(EXTENSION).as_posix())
    print(target)


if __name__ == "__main__":
    main()
