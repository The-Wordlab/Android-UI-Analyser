"""Platform-neutral lifecycle status for AUA-managed automation targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TargetSupervisionStatus:
    """What the selected platform knows about one target's AUA-owned lifecycle.

    A physical device or a simulator started outside AUA normally has no status. Platforms that
    supervise targets can expose their owner and idle-retirement policy without leaking native
    metadata paths, process vocabulary, or emulator-specific fields into shared callers.
    """

    target_id: str
    managed: bool = False
    owner: str | None = None
    instance_id: str | None = None
    started_at: float | None = None
    last_activity: float | None = None
    idle_timeout_s: float | None = None
    monitor_running: bool = False
    idle_stop_explicit: bool = False

    def __post_init__(self) -> None:
        if not str(self.target_id).strip():
            raise ValueError("target_id must not be empty")


__all__ = ["TargetSupervisionStatus"]
