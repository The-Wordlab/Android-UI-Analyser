#!/usr/bin/env python3
"""Claude PreToolUse guard: route device work through AUA instead of raw adb/ffmpeg."""

from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ShellCall:
    executable: str
    args: tuple[str, ...]


_SEPARATORS = frozenset({";", "&", "&&", "|", "||", "(", ")"})
_SHELLS = frozenset({"bash", "dash", "sh", "zsh"})


def _tokens(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(
            command.replace("\n", " ; "), posix=True, punctuation_chars=";&|()"
        )
        lexer.whitespace_split = True
        lexer.commenters = "#"
        return list(lexer)
    except ValueError:
        return []


def _basename(word: str) -> str:
    return os.path.basename(word.rstrip("/"))


def _unwrap(segment: list[str]) -> ShellCall | None:
    words = list(segment)
    while words and "=" in words[0] and not words[0].startswith(("-", "/")):
        name, _equals, _value = words[0].partition("=")
        if not name.replace("_", "a").isalnum():
            break
        words.pop(0)
    while words:
        executable = _basename(words[0])
        if executable in {"command", "builtin", "nohup"}:
            words.pop(0)
            continue
        if executable == "env":
            words.pop(0)
            while words and (words[0].startswith("-") or "=" in words[0]):
                words.pop(0)
            continue
        if executable == "sudo":
            words.pop(0)
            while words and words[0].startswith("-"):
                words.pop(0)
            continue
        if executable in {"timeout", "gtimeout"}:
            words.pop(0)
            while words and words[0].startswith("-"):
                words.pop(0)
            if words:
                words.pop(0)
            continue
        return ShellCall(executable=executable, args=tuple(words[1:]))
    return None


def shell_calls(command: str) -> list[ShellCall]:
    """Return executable positions, excluding harmless mentions such as ``rg adb README``."""

    calls: list[ShellCall] = []
    segment: list[str] = []
    for token in [*_tokens(command), ";"]:
        if token in _SEPARATORS or token.strip(";&|()") == "":
            call = _unwrap(segment)
            if call is not None:
                calls.append(call)
                if call.executable in _SHELLS:
                    command_index = next(
                        (
                            index
                            for index, arg in enumerate(call.args)
                            if arg.startswith("-") and "c" in arg[1:]
                        ),
                        None,
                    )
                    if command_index is not None and command_index + 1 < len(call.args):
                        calls.extend(shell_calls(call.args[command_index + 1]))
            segment = []
        else:
            segment.append(token)
    return calls


def _decision(decision: str, reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": decision,
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def _adb_replacement(args: tuple[str, ...]) -> str | None:
    words = list(args)
    # Global targeting flags do not alter which operation follows.
    while words and words[0] in {"-s", "-t", "-H", "-P", "-L"}:
        words = words[2:]
    while words and words[0] in {"-d", "-e"}:
        words = words[1:]
    lower = " ".join(words).casefold()
    if " shell input " in f" {lower} " or " uiautomator " in f" {lower} ":
        return (
            "Raw coordinate/UI input is blocked. Start with `aua session start --goal \"<goal>\"`, "
            "then act on a fresh returned id/rid with `aua tap-and-analyze`, "
            "`input-and-analyze`, `swipe-and-analyze`, or `key-and-analyze`."
        )
    if "screenrecord" in lower or (
        lower.startswith("pull ") and any(token in lower for token in (".mp4", "recording"))
    ):
        return (
            "Raw screenrecord is blocked. Use `aua record start`, perform selector-safe AUA "
            "actions, then `aua record stop <path>`; stop validates MP4 finalization."
        )
    if "screencap" in lower:
        return "Raw screencap is blocked. Use `aua screenshot <path>` or the returned AUA evidence."
    if lower.startswith("logcat") or "shell logcat" in lower:
        return "Raw logcat is blocked. Use `aua logcat mark`, perform the action, then `aua logcat`."
    if lower.startswith(("install ", "install-multiple ", "uninstall ")):
        return "Raw package installation is blocked. Use `aua install <apk> [--launch]`."
    if any(
        token in lower
        for token in (
            "shell am start",
            "shell am force-stop",
            "shell monkey",
            "shell cmd package",
        )
    ):
        return (
            "Raw app lifecycle commands are blocked. Use `aua app launch|stop|kill|clear`, "
            "`aua open-and-analyze`, or `aua install`."
        )
    if any(
        token in lower
        for token in (
            "shell settings put",
            "shell settings delete",
            "shell svc ",
            " shell emu ",
        )
    ) or lower.startswith("emu "):
        return (
            "Raw persistent device settings are blocked. Use AUA's session-owned `network`, "
            "`dev`, `orientation`, `location`, or emulator controls so cleanup is ledgered."
        )
    if any(token in lower for token in ("shell run-as", "sqlite3", "/databases/")):
        return "Raw app-data access is blocked. Use guarded `aua db list|schema|query|execute`."
    if lower.startswith("devices"):
        return (
            "Raw device discovery is blocked for goal work. Use "
            "`aua session start --goal \"<goal>\"`; it selects, leases, and provisions safely."
        )
    if any(
        token in lower
        for token in (
            "shell getprop",
            "shell dumpsys",
            "shell pidof",
            "shell pm path",
            "shell settings get",
            "get-state",
            "wait-for-device",
        )
    ):
        return "Raw device diagnostics are blocked. Use the bounded read-only `aua shell ...` surface."
    return None


def main() -> int:
    try:
        payload: dict[str, Any] = json.load(sys.stdin)
    except (ValueError, TypeError):
        return 0
    command = str((payload.get("tool_input") or {}).get("command") or "")
    for call in shell_calls(command):
        if call.executable == "adb":
            replacement = _adb_replacement(call.args)
            if replacement is not None:
                _decision("deny", replacement)
            else:
                _decision(
                    "ask",
                    "This raw adb operation is outside AUA's selector, lease, and cleanup guardrails. "
                    "Run `aua capabilities --goal \"<operation>\"` first. Approve only if AUA "
                    "genuinely has no equivalent.",
                )
            return 0
        if call.executable == "ffmpeg":
            joined = " ".join(call.args).casefold()
            if any(
                marker in joined
                for marker in ("/android-ui-analyser/", "/captures/", "/aua_", "/aua-")
            ):
                _decision(
                    "deny",
                    "Use `aua capture sheet <output.png> --since last-action --max-frames 6 "
                    "--timestamps` for AUA transition evidence; it is bounded and needs no ffmpeg.",
                )
                return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
