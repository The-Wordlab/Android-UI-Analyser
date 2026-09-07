"""Platform-neutral wall-clock write verification, including teardown replay."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .errors import DeviceError

if TYPE_CHECKING:
    from .platforms.runtime import TargetRuntime


def verify_clock_readback(runtime: TargetRuntime, timestamp_ms: int, started: float) -> int:
    """Require a readback compatible with the requested time and command duration.

    Native clocks may accept only whole seconds. Two seconds accommodates that rounding and
    scheduling, while the monotonic duration allows a slow native call without masking a no-op.
    Failure deliberately leaves restoration bookkeeping to the caller.
    """
    actual = runtime.get_clock_ms()
    elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
    if actual is None or not timestamp_ms - 2_000 <= actual <= timestamp_ms + elapsed_ms + 2_000:
        raise DeviceError(
            "device clock readback did not confirm the requested time "
            f"(requested={timestamp_ms}, observed={actual})",
            code="clock_write_unverified",
            hint=(
                "The clock may be unchanged or controlled by automatic time synchronization. "
                "The restore point remains pending; do not restart the app to verify this write."
            ),
        )
    return actual


def set_clock_verified(runtime: TargetRuntime, timestamp_ms: int) -> int:
    started = time.monotonic()
    runtime.set_clock(timestamp_ms)
    return verify_clock_readback(runtime, timestamp_ms, started)


def restore_clock_verified(runtime: TargetRuntime, timestamp_ms: int) -> int:
    """Prove the desired original clock before attempting another native write.

    A denied time-travel write commonly leaves the clock unchanged. Requiring a second write
    to that already-correct clock makes cleanup fail forever on an unprivileged target.
    Only readback establishes this no-op; an earlier shell denial alone proves nothing.
    """
    started = time.monotonic()
    try:
        return verify_clock_readback(runtime, timestamp_ms, started)
    except DeviceError as exc:
        if exc.code != "clock_write_unverified":
            raise
    # Account for time spent on the failed pre-read before restoring, including slow adapters.
    elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
    return set_clock_verified(runtime, timestamp_ms + elapsed_ms)
