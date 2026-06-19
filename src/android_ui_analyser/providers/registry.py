"""Provider registry, factory, and ordered fallback-chain runner (PRD §7).

- **Registry:** a decorator (`register_ocr` / `register_detection` / `register_grounding`)
  maps a provider *name* → strategy *class*, keyed by kind.
- **Discovery:** ``load_providers`` auto-imports every module in each provider package so
  their decorators fire — drop a new file in ``providers/ocr/`` and it is found, no
  ``__init__`` edit required.
- **Factory:** ``ProviderFactory`` instantiates configured strategies (reading their
  ``models.<name>`` settings) and assembles the ordered chain from the ``chain:`` list.
- **Chain runner:** ``run_chain`` tries providers in order; on unavailable/error/timeout
  it logs to stderr and advances; a clean-but-empty result lets the next provider try but
  is returned if the whole chain is empty; if every provider failed it raises
  ``ProviderError`` (exit 4).

The engine and CLI depend only on this module + ``base.py`` — never on a concrete provider.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import TYPE_CHECKING, Any, TypeVar

from ..errors import ConfigError, ProviderError
from .base import ChainSpec, Provider

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import Config

logger = logging.getLogger("android_ui_analyser.providers")

_KINDS = ("ocr", "detection", "grounding")
_REGISTRY: dict[str, dict[str, type[Provider]]] = {k: {} for k in _KINDS}
_PROVIDER_PACKAGES: dict[str, str] = {
    "ocr": "android_ui_analyser.providers.ocr",
    "detection": "android_ui_analyser.providers.detection",
    "grounding": "android_ui_analyser.providers.grounding",
}
_loaded: set[str] = set()


# --------------------------------------------------------------------------- register


def register(kind: str, name: str) -> Callable[[type[Provider]], type[Provider]]:
    """Class decorator: register ``cls`` as provider ``name`` of ``kind``."""
    if kind not in _REGISTRY:
        _REGISTRY[kind] = {}

    def deco(cls: type[Provider]) -> type[Provider]:
        cls.kind = kind
        cls.name = name
        _REGISTRY[kind][name] = cls
        return cls

    return deco


def register_ocr(name: str) -> Callable[[type[Provider]], type[Provider]]:
    return register("ocr", name)


def register_detection(name: str) -> Callable[[type[Provider]], type[Provider]]:
    return register("detection", name)


def register_grounding(name: str) -> Callable[[type[Provider]], type[Provider]]:
    return register("grounding", name)


# --------------------------------------------------------------------------- discovery


def load_providers(kind: str | None = None) -> None:
    """Import provider modules so their ``@register_*`` decorators run.

    Robust to a single broken module: an import failure is logged and skipped rather
    than killing the whole registry (consistent with the lazy-import philosophy).
    """
    kinds = [kind] if kind else list(_PROVIDER_PACKAGES)
    for k in kinds:
        if k in _loaded:
            continue
        pkg_name = _PROVIDER_PACKAGES.get(k)
        if pkg_name is None:
            continue
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("could not import provider package %s: %s", pkg_name, exc)
            _loaded.add(k)
            continue
        for modinfo in pkgutil.iter_modules(pkg.__path__):
            if modinfo.name.startswith("_"):
                continue
            full = f"{pkg_name}.{modinfo.name}"
            try:
                importlib.import_module(full)
            except Exception as exc:
                logger.debug("provider module %s failed to import: %s", full, exc)
        _loaded.add(k)


def registered_names(kind: str) -> list[str]:
    load_providers(kind)
    return sorted(_REGISTRY.get(kind, {}))


def get_provider_class(kind: str, name: str) -> type[Provider]:
    load_providers(kind)
    try:
        return _REGISTRY[kind][name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY.get(kind, {}))) or "(none)"
        raise ConfigError(
            f"unknown {kind} provider '{name}'",
            hint=f"Known {kind} providers: {known}. Check your `{kind}.chain` config.",
        ) from exc


def all_registered() -> dict[str, list[str]]:
    """Every registered provider, keyed by kind (loads all packages)."""
    load_providers()
    return {k: sorted(v) for k, v in _REGISTRY.items()}


# --------------------------------------------------------------------------- factory


class ProviderFactory:
    """Builds configured strategy instances and ordered chains from config."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def _settings_for(self, name: str) -> dict[str, Any]:
        models = getattr(self.config, "models", {}) or {}
        value = models.get(name, {})
        return dict(value) if value else {}

    def create(self, kind: str, name: str) -> Provider:
        cls = get_provider_class(kind, name)
        return cls(self._settings_for(name))

    def is_enabled(self, kind: str) -> bool:
        section = getattr(self.config, kind, None)
        return bool(getattr(section, "enabled", False)) if section is not None else False

    def chain_names(self, kind: str) -> list[str]:
        section = getattr(self.config, kind, None)
        return list(getattr(section, "chain", []) or [])

    def build_chain(self, kind: str) -> ChainSpec:
        load_providers(kind)
        providers: list[Provider] = []
        for name in self.chain_names(kind):
            if name not in _REGISTRY.get(kind, {}):
                logger.warning("unknown %s provider '%s' in chain; skipping", kind, name)
                continue
            providers.append(self.create(kind, name))
        return ChainSpec(kind=kind, providers=providers)


# --------------------------------------------------------------------------- runner

T = TypeVar("T")


def _default_is_empty(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, (list, tuple, dict, str)):
        return len(result) == 0
    return False


def run_chain(
    chain: ChainSpec,
    op: Callable[[Provider], T],
    *,
    is_empty: Callable[[Any], bool] | None = None,
    timeout_s: float | None = None,
) -> tuple[T, str]:
    """Run ``op`` against each provider in order; return ``(result, provider_name)``.

    Skips unavailable providers, advances past errors/timeouts/empty results, returns
    the first non-empty result. If the chain produced only clean-but-empty results,
    returns the first such (genuinely-empty screen). If every provider was unavailable
    or errored, raises :class:`ProviderError`.
    """
    empty_check = is_empty or _default_is_empty
    attempts: list[tuple[str, str]] = []
    clean_empty: tuple[T, str] | None = None

    for provider in chain.providers:
        avail = provider.is_available()
        if not avail.ok:
            attempts.append((provider.name, avail.reason))
            logger.info("skip %s provider %s: %s", chain.kind, provider.name, avail.reason)
            continue
        try:
            if timeout_s is not None:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(op, provider).result(timeout=timeout_s)
            else:
                result = op(provider)
        except FuturesTimeout:
            attempts.append((provider.name, f"timeout after {timeout_s}s"))
            logger.warning("%s provider %s timed out", chain.kind, provider.name)
            continue
        except Exception as exc:
            attempts.append((provider.name, f"error: {exc}"))
            logger.warning("%s provider %s errored: %s", chain.kind, provider.name, exc)
            continue
        if empty_check(result):
            attempts.append((provider.name, "empty result"))
            logger.info("%s provider %s returned empty; advancing", chain.kind, provider.name)
            if clean_empty is None:
                clean_empty = (result, provider.name)
            continue
        logger.debug("%s provider %s succeeded", chain.kind, provider.name)
        return result, provider.name

    if clean_empty is not None:
        return clean_empty
    raise ProviderError(chain.kind, attempts)
