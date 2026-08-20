"""What ``aua doctor`` says about the optional local policy.

Doctor used to say nothing at all. Observed on a real host: `policy.mode: advisory` with a
two-provider chain, neither provider's MLX runtime present in the environment `aua` runs from, and
the whole doctor report contained zero mentions of the policy. Autopilot then handed off every
step, and the missing import was found by hand. A configured feature whose runtime is absent is
precisely a doctor finding — and "no providers" is the one thing it must not look like.

This module is host-only and read-only. It builds on :func:`policy.policy_status` (the surface
`aua policy status` and MCP already share, which never loads a model or touches a device) and
turns it into a doctor check that separates three faults with three different fixes:

* the **runtime** is not installed in this environment → install the extra, here;
* the runtime is fine but the **base model** was never pointed at → configure ``model_path``;
* the host **cannot** run it at all (MLX is Apple-silicon-only) → nothing to install.

It deliberately does not care whether ``policy.enabled`` is on. That switch buys passive advice on
every ordinary ``analyze`` and was measured at roughly twenty seconds per call, which is why it is
off; ``mode`` is what makes the lane usable by ``aua session autopilot``. Doctor reports the cost
and never asks for it.
"""

from __future__ import annotations

from typing import Any

__all__ = ["POLICY_EXTRAS", "install_command", "policy_check"]

#: Extras that carry a policy runtime, most capable last — used to pick the installer invocation.
POLICY_EXTRAS = ("functiongemma", "hybrid-policy")

#: What the passive switch costs. Measured on the host this check was written for: an ordinary
#: `analyze` went from 0.31s to 22.7s once `policy.enabled` attached the chain to every call.
PASSIVE_COST_NOTE = (
    "policy.enabled=false, so ordinary analyze calls run no model "
    "(measured ~20s per call while it was on); `mode` alone is what "
    "`aua session autopilot` reads"
)
PASSIVE_COST_NOTE_ON = (
    "policy.enabled=true, so every ordinary analyze and session-progress call runs the chain "
    "(measured ~20s per call); `aua session autopilot` does not need it"
)


def install_command(extras: tuple[str, ...] | list[str]) -> str:
    """The exact command that puts *extras* into the environment ``aua`` runs from.

    ``install.sh`` is named first because the failure this fixes is specifically a *global*
    install: extras added to a `uv tool` environment by hand disappear at the next
    ``uv tool upgrade``, since the receipt never named them. The installer's opt-in puts them in
    the install target, so the receipt does.
    """

    wanted = [e for e in POLICY_EXTRAS if e in set(extras)] or ["functiongemma"]
    if "hybrid-policy" in wanted:
        # The installer's hybrid opt-in is cumulative — it installs the selector runtime too, so
        # say what the command actually does rather than only the extra that was asked about.
        wanted = list(POLICY_EXTRAS)
    flag = "--with-policy=hybrid" if "hybrid-policy" in wanted else "--with-policy"
    joined = ",".join(wanted)
    return (
        f"./install.sh {flag}   # installs the {joined} extra(s) into the environment `aua` runs "
        f"from; without a clone: pip install 'android-ui-analyser[{joined}]'"
    )


def _entry(status: dict[str, Any]) -> dict[str, Any]:
    """Turn one provider's host-only status into a doctor row with a cause and a fix."""

    name = str(status.get("provider") or "?")
    mode_supported = bool(status.get("configured_mode_supported", True))
    runnable = bool(status.get("available")) and mode_supported
    runtime = status.get("runtime")
    runtime_ready = bool(runtime.get("ready")) if isinstance(runtime, dict) else None
    runtime_reason = str(runtime.get("reason") or "") if isinstance(runtime, dict) else ""
    extra = status.get("install_extra")
    reason = str(status.get("reason") or "")
    health = status.get("selection_health")

    entry: dict[str, Any] = {
        "name": name,
        "runnable": runnable,
        "reason": reason,
        "runtime_ready": runtime_ready,
        "install_extra": extra,
        "mode_supported": mode_supported,
        "cause": "ok",
    }

    if runnable:
        return entry

    if isinstance(health, dict) and health.get("usable") is False:
        entry["cause"] = "unusable_output"
        entry["remedy"] = (
            "this provider is broken, not slow: replace it in `policy.chain` "
            "(`aua policy status` shows the per-provider rate)"
        )
        return entry
    if not mode_supported:
        entry["cause"] = "mode_unsupported"
        entry["remedy"] = (
            f"{name} does not authorize the configured `policy.mode`; see `aua policy status` "
            "for the rollout its manifest allows"
        )
        return entry
    if runtime_ready is False:
        # The provider owns its platform contract. Do not infer that every unavailable local
        # runtime is unsupported merely because doctor itself happens to be running on Linux:
        # fake adapters and future non-MLX providers may be perfectly valid there. The real MLX
        # providers report this explicit reason before they check imports.
        if "requires apple silicon" in runtime_reason.casefold():
            entry["cause"] = "unsupported_host"
            entry["remedy"] = None  # nothing to install: the MLX runtime is Apple-silicon-only
            return entry
        entry["cause"] = "runtime_missing"
        entry["remedy"] = install_command([extra] if extra else [])
        return entry
    if runtime_ready is True:
        entry["cause"] = "artifacts"
        entry["remedy"] = (
            f"set `models.{name}.model_path` to the local base model directory "
            "(docs/LOCAL_POLICY_SETUP.md step 2)"
        )
        return entry
    # No runtime verdict at all: the provider could not even be constructed (an unknown name in
    # `policy.chain` is the usual cause), so guessing at `model_path` would send the reader to the
    # wrong file. Report the fault it gave us and point at the full readiness surface.
    entry["cause"] = "unattributed"
    entry["remedy"] = "`aua policy status` has the full per-provider readiness report"
    return entry


def policy_check(config: Any, *, factory: Any | None = None) -> dict[str, Any]:
    """Report the configured policy chain, what can run here, and the exact fix if not.

    ``ok`` is False only when the policy is *in use* — ``mode`` is shadow/advisory, or the passive
    switch is on — and nothing in its chain can run. An untouched optional feature is not
    breakage, so the shipped default reports ``ok: True`` and stays quiet.
    """

    from .policy import policy_status

    status = policy_status(config, factory=factory)
    mode = str(status.get("mode") or "off")
    enabled = bool(status.get("enabled"))
    chain = [str(name) for name in status.get("chain") or []]
    configured = enabled or mode in {"shadow", "advisory"}
    providers = [_entry(dict(value)) for value in status.get("providers") or []]
    runnable = [p for p in providers if p["runnable"]]
    blocked = [p for p in providers if not p["runnable"]]

    check: dict[str, Any] = {
        "ok": True,
        "configured": configured,
        "enabled": enabled,
        "mode": mode,
        "chain": chain,
        "providers": providers,
        "cost": PASSIVE_COST_NOTE_ON if enabled else PASSIVE_COST_NOTE,
    }

    if not configured:
        # Say what is configured even while it sleeps: "off" plus the chain is the difference
        # between a feature nobody asked for and one somebody switched off on purpose.
        chain_text = ", ".join(chain) or "(no chain)"
        check["detail"] = (
            f"off (mode={mode}, enabled={str(enabled).lower()}) — "
            f"chain {chain_text} is not consulted"
        )
        return check

    if not chain:
        check["ok"] = False
        check["detail"] = f"mode={mode} but `policy.chain` is empty, so nothing can run"
        check["hint"] = "Name a provider in `policy.chain`, or set `policy.mode: off`."
        return check

    if runnable:
        check["detail"] = (
            f"mode={mode} · runnable: {', '.join(p['name'] for p in runnable)}"
            + (
                f" · unavailable: {', '.join(p['name'] for p in blocked)}"
                if blocked
                else ""
            )
        )
        return check

    check["ok"] = False
    faults = ", ".join(f"{p['name']} ({_short_cause(p)})" for p in blocked)
    check["detail"] = f"mode={mode} but the chain cannot run here: {faults}"
    remedies = [p["remedy"] for p in blocked if p.get("remedy")]
    extras = [p["install_extra"] for p in blocked if p["cause"] == "runtime_missing"]
    if extras:
        check["hint"] = install_command([e for e in extras if e])
    elif remedies:
        check["hint"] = remedies[0]
    elif any(p["cause"] == "unsupported_host" for p in blocked):
        check["hint"] = (
            "The local MLX policy runtime is Apple-silicon-only; this host cannot run it. "
            "Set `policy.mode: off` to stop configuring a lane that cannot start."
        )
    return check


def _short_cause(entry: dict[str, Any]) -> str:
    return {
        "runtime_missing": "runtime not installed here",
        "unsupported_host": "unsupported host",
        "artifacts": "base model not configured",
        "mode_unsupported": "mode not authorized",
        "unusable_output": "unusable output",
        "unattributed": "unavailable",
    }.get(str(entry.get("cause")), str(entry.get("reason") or "unavailable"))
