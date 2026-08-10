"""Typed errors, exit codes, and the structured stderr emitter (PRD §5, §14).

Exit codes:
    0  success
    1  ``has`` miss, OR an unexpected internal error (see :attr:`ExitCode.INTERNAL`)
    2  usage error
    3  no device / device error
    4  provider error after exhausting fallbacks
    5  config error
    6  selector matched nothing (target/assertion target absent)
    7  selector matched several (caller must disambiguate)
    8  expectation failed (`aua expect-and-analyze` predicate is false)

Codes 6-8 exist so an agent can tell *addressing* failures apart from *device* failures:
a selector that matches nothing is a script bug, exit 3 is a broken phone. A silent
success on either is the worst possible outcome for an autonomous caller.

Exit ``1`` is overloaded on purpose: ``aua has`` uses it as a cheap boolean miss (no
structured error), while an unexpected exception also exits ``1`` with
``{"error":{"code":"internal_error",…}}`` on stderr — agents branch on the JSON shape.

Errors print a structured object to **stderr** (JSON results go to stdout):
    {"error": {"code": ..., "message": ..., "hint": ...}}
"""

from __future__ import annotations

import json
import sys
from enum import IntEnum
from typing import IO


class ExitCode(IntEnum):
    OK = 0
    INTERNAL = 1  # also: `aua has` miss (no AuaError); see module docstring
    USAGE = 2
    DEVICE = 3
    PROVIDER = 4
    CONFIG = 5
    NOT_FOUND = 6
    AMBIGUOUS = 7
    ASSERTION = 8
    # Distinct from DEVICE on purpose: "another agent is driving this one" is a *routable*
    # condition — the caller should pick a different emulator and carry on — whereas a plain
    # device error means nothing is reachable. A runner that cannot tell them apart either
    # aborts a run that could have proceeded, or retries forever against a busy device.
    LEASED = 9


class AuaError(Exception):
    """Base class for all tool errors.

    Carries a machine-readable ``code``, a human ``message``, an actionable ``hint``,
    and the process ``exit_code`` to use.
    """

    exit_code: ExitCode = ExitCode.USAGE
    code: str = "error"

    def __init__(self, message: str, *, hint: str | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        if code is not None:
            self.code = code

    def to_dict(self) -> dict[str, object]:
        err: dict[str, object] = {"code": self.code, "message": self.message}
        if self.hint:
            err["hint"] = self.hint
        return {"error": err}


class UsageError(AuaError):
    exit_code = ExitCode.USAGE
    code = "usage"


class DeviceError(AuaError):
    exit_code = ExitCode.DEVICE
    code = "device"


class DeviceLeasedError(AuaError):
    """The requested device is held by another agent, or nothing free matches ``--needs``.

    Separate from :class:`DeviceError` so a caller can branch: this one says *try a different
    emulator*, and the hint names which are free.
    """

    exit_code = ExitCode.LEASED
    code = "device_leased"


class ConfigError(AuaError):
    exit_code = ExitCode.CONFIG
    code = "config"


class ProviderError(AuaError):
    """Raised when an entire provider fallback chain is exhausted (PRD §7)."""

    exit_code = ExitCode.PROVIDER
    code = "provider_exhausted"

    def __init__(
        self,
        kind: str,
        attempts: list[tuple[str, str]] | None = None,
        *,
        hint: str | None = None,
    ) -> None:
        self.kind = kind
        self.attempts = attempts or []
        detail = "; ".join(f"{name}: {reason}" for name, reason in self.attempts)
        message = f"all {kind} providers failed"
        if detail:
            message += f" ({detail})"
        if hint is None:
            hint = (
                f"Check `aua doctor` for {kind} provider availability, or adjust the "
                f"`{kind}.chain` in your config."
            )
        super().__init__(message, hint=hint)


class ElementNotFoundError(AuaError):
    """A referenced element id is not in the cached analyze result."""

    exit_code = ExitCode.USAGE
    code = "element_not_found"


class StabilityTimeout(AuaError):
    """``wait --for-stable`` never settled within the timeout (PRD §5, AC14)."""

    exit_code = ExitCode.DEVICE
    code = "wait_timeout"


class SelectorNotFoundError(AuaError):
    """A ``--rid``/``--text``/``--by`` selector matched no element on screen."""

    exit_code = ExitCode.NOT_FOUND
    code = "selector_not_found"


class SelectorAmbiguousError(AuaError):
    """A selector matched several elements; picking one silently would be a coin flip."""

    exit_code = ExitCode.AMBIGUOUS
    code = "selector_ambiguous"


class ExpectationFailed(AuaError):
    """An ``aua expect-and-analyze`` predicate is false — a test failure, not a tool failure."""

    exit_code = ExitCode.ASSERTION
    code = "expectation_failed"


def emit_error(err: AuaError, *, stream: IO[str] | None = None) -> int:
    """Write the structured error object to stderr and return its exit code."""
    stream = stream if stream is not None else sys.stderr
    json.dump(err.to_dict(), stream, ensure_ascii=False)
    stream.write("\n")
    stream.flush()
    return int(err.exit_code)
