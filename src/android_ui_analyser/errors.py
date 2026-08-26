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
from typing import IO, Any


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


class UnsupportedPlatformCapabilityError(UsageError):
    """The selected platform does not implement one optional AUA capability.

    This is a usage-level, machine-readable refusal rather than a missing attribute or a
    backend import error.  Agents can therefore choose a different action/platform, and a
    third-party platform can grow capability-by-capability without pretending to support the
    entire Android surface on day one.
    """

    code = "platform_capability_unsupported"

    def __init__(self, platform: str, capability: str) -> None:
        super().__init__(
            f"platform {platform!r} does not support capability {capability!r}",
            hint=(
                "Choose a platform adapter that provides this capability, or install/update "
                f"a {platform!r} platform plugin that implements {capability!r}."
            ),
        )


class DeviceError(AuaError):
    exit_code = ExitCode.DEVICE
    code = "device"


class DaemonBusyError(AuaError):
    """A live daemon owns the device but cannot accept another call yet."""

    exit_code = ExitCode.DEVICE
    code = "daemon_busy"


class DaemonOutcomeUnknownError(AuaError):
    """A daemon accepted a request but its response did not reach this caller.

    Retrying a state-changing request in-process can execute it twice.  This error is therefore
    deliberately structured and terminal at the routing boundary: the caller may inspect the
    current screen after the daemon becomes responsive, but must not replay the action blindly.
    """

    exit_code = ExitCode.DEVICE
    code = "daemon_outcome_unknown"


class DeviceLeasedError(AuaError):
    """The requested device is held by another agent, or nothing free matches ``--needs``.

    Separate from :class:`DeviceError` so a caller can branch: this one says *try a different
    emulator*, and the hint names which are free.
    """

    exit_code = ExitCode.LEASED
    code = "device_leased"


class LeasedTargetUnavailableError(DeviceError):
    """The caller's sticky target temporarily vanished from platform discovery.

    This is deliberately not a lease-conflict error: another free target is not a safe
    substitute for a screen whose ids, app state, and session history belong to the retained
    lease.
    """

    code = "leased_target_unavailable"


class LeaseSwitchRequiredError(UsageError):
    """A caller already owns another device and must acknowledge replacing it."""

    code = "lease_switch_required"


class LeaseHandoffPendingError(UsageError):
    """The current owner froze its lease while a one-time transfer is pending."""

    code = "lease_handoff_pending"


class ConfigError(AuaError):
    exit_code = ExitCode.CONFIG
    code = "config"


class InvalidPlatformCapabilityError(ConfigError):
    """A plugin advertised a capability but did not implement its common contract."""

    code = "platform_capability_invalid"

    def __init__(self, platform: str, capability: str, missing: list[str]) -> None:
        joined = ", ".join(missing)
        super().__init__(
            f"platform {platform!r} has an invalid {capability!r} capability; missing: {joined}",
            hint="Update the platform plugin to implement the complete capability contract.",
        )


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


class _CarriesObservation(AuaError):
    """An error that hands back the screen it already read.

    "Your target is not here" is only actionable with the screen that proves it. Every raise
    site that reaches this has *already* analyzed — that read is how it knows the target is
    absent — so attaching it costs nothing and removes a guaranteed round trip on the one path
    where the caller is most confused about what is on screen.

    Trimmed on the way out (``compact``): a failure has no business costing more bytes than a
    success.
    """

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        observation: Any | None = None,
    ) -> None:
        super().__init__(message, hint=hint)
        self.observation = observation

    def to_dict(self) -> dict[str, object]:
        payload = super().to_dict()
        if self.observation is None:
            return payload
        if hasattr(self.observation, "as_dict"):
            observation = self.observation.as_dict("compact")
        else:
            observation = self.observation
        error = payload["error"]
        if isinstance(error, dict):
            error["observation_present"] = True
            error["observation"] = observation
        return payload


class ElementNotFoundError(_CarriesObservation):
    """A referenced element id is not in the cached analyze result."""

    exit_code = ExitCode.USAGE
    code = "element_not_found"


class StaleElementIdError(ElementNotFoundError):
    """A frame-local id no longer names the element observed under that id.

    This is distinct from a missing id: the id existed in the cached observation, but a fresh
    read shows that the frame changed before the requested action could be dispatched.  Reporting
    that explicitly is what lets an agent recover with a selector instead of retrying the same
    integer against a newly-numbered screen.
    """

    code = "stale_element_id"


class StabilityTimeout(AuaError):
    """``wait --for-stable`` never settled within the timeout (PRD §5, AC14)."""

    exit_code = ExitCode.DEVICE
    code = "wait_timeout"


class JobCancelledError(AuaError):
    """A background read-only job stopped at the caller's explicit request."""

    exit_code = ExitCode.DEVICE
    code = "job_cancelled"


class SelectorNotFoundError(_CarriesObservation):
    """A ``--rid``/``--text``/``--by`` selector matched no element on screen."""

    exit_code = ExitCode.NOT_FOUND
    code = "selector_not_found"

    # Behaviour comes from `_CarriesObservation`; only the code and exit status differ.


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
