"""The installed extension follows AUA's version, so a loaded copy can tell it is out of date."""

from __future__ import annotations

import json

from android_ui_analyser import chrome_extension_setup as setup


def test_install_stamps_the_aua_version_and_a_newer_aua_refreshes_the_copy(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    installed = setup.copy_extension_files() / "manifest.json"
    assert json.loads(installed.read_text())["version"] == setup._extension_version()
    assert setup.refresh_installed_extension() is False

    monkeypatch.setattr(setup, "__version__", "99.1.2")
    assert setup.refresh_installed_extension() is True
    assert json.loads(installed.read_text())["version"] == "99.1.2"
