"""Pluggable platform strategies used by the AUA engine."""

from .base import AppBundle, InstalledApp, NormalizedTree, PlatformAdapter
from .registry import ENTRY_POINT_GROUP, PlatformFactory, register_platform, registered_platforms
from .services import CAPABILITY_METHODS

__all__ = [
    "ENTRY_POINT_GROUP",
    "AppBundle",
    "InstalledApp",
    "CAPABILITY_METHODS",
    "NormalizedTree",
    "PlatformAdapter",
    "PlatformFactory",
    "register_platform",
    "registered_platforms",
]
