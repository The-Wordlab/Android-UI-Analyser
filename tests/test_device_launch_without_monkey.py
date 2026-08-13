from __future__ import annotations

from android_ui_analyser.device import Uiautomator2Device


class _ShellDevice:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def shell(self, command: str) -> str:
        self.commands.append(command)
        if command.startswith("cmd package query-activities"):
            return (
                "2 activities found:\n"
                "  com.example.catalog/.MainActivity\n"
                "  com.example.catalog/.DeveloperToolsActivity\n"
            )
        return "Starting: Intent"


def test_package_launch_resolves_component_without_monkey() -> None:
    wrapper = object.__new__(Uiautomator2Device)
    raw = _ShellDevice()
    wrapper._d = raw
    wrapper.serial = "emulator-fictional"

    wrapper.launch_app("com.example.catalog")

    assert raw.commands == [
        "cmd package query-activities --brief -a android.intent.action.MAIN "
        "-c android.intent.category.LAUNCHER -p com.example.catalog",
        "am start -n com.example.catalog/.MainActivity",
    ]
