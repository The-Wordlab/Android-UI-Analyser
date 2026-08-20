"""`aua doctor` must tell the truth about the optional local policy.

Observed on a real host: the user's config had `policy.mode: advisory` with a two-provider chain,
neither provider's MLX runtime was installed in the environment `aua` actually runs from, and
`aua doctor` printed *nothing at all* about the policy — zero mentions in the whole report. The
missing import was tracked down by hand after autopilot handed off every step. Doctor is exactly
the surface that should have said "the policy is configured but its runtime is not installed here".

So this pins four things:

* a configured provider that cannot run appears, as itself, with the reason and the exact fix — it
  must never present as "no providers";
* a runtime that is missing is distinguished from a base model that was never pointed at, because
  the two have different fixes;
* the policy being *off* is not a failure and doctor must not read as an instruction to switch it
  on — enabling it attaches inference to every ordinary `analyze` and was measured at ~20s a call;
* the cost is discoverable from doctor either way.

Everything here runs against fake providers: no MLX, no model, no device.
"""

from __future__ import annotations

from typing import Any

import android_ui_analyser.cli as cli
from android_ui_analyser import policy_doctor
from android_ui_analyser.config import Config


class _FakeProvider:
    def __init__(self, name: str, status: dict[str, Any], *, supported_mode: bool = True) -> None:
        self.name = name
        self._status = status
        self._supported_mode = supported_mode

    def status(self) -> dict[str, Any]:
        return dict(self._status)

    def supports_mode(self, mode: str) -> bool:
        return self._supported_mode


class _FakeFactory:
    def __init__(self, providers: dict[str, _FakeProvider]) -> None:
        self._providers = providers

    def create(self, kind: str, name: str) -> _FakeProvider:
        assert kind == "policy", kind
        return self._providers[name]


def _config(*, mode: str = "advisory", enabled: bool = False, chain: list[str]) -> Config:
    cfg = Config()
    cfg.policy.mode = mode  # type: ignore[assignment]
    cfg.policy.enabled = enabled
    cfg.policy.chain = chain
    return cfg


def _runtime_missing(name: str, extra: str) -> _FakeProvider:
    reason = f"optional dependency missing; install android-ui-analyser[{extra}]"
    return _FakeProvider(
        name,
        {
            "provider": name,
            "available": False,
            "reason": reason,
            "loaded": False,
            "install_extra": extra,
            "runtime": {"ready": False, "reason": reason},
        },
    )


def _artifacts_missing(name: str, extra: str) -> _FakeProvider:
    return _FakeProvider(
        name,
        {
            "provider": name,
            "available": False,
            "reason": "model_path is not configured",
            "loaded": False,
            "install_extra": extra,
            "runtime": {"ready": True, "reason": "local MLX runtime available"},
            "artifacts": {"ready": False, "reason": "model_path is not configured"},
        },
    )


def _ready(name: str, extra: str) -> _FakeProvider:
    return _FakeProvider(
        name,
        {
            "provider": name,
            "available": True,
            "reason": "local model and adapter are ready",
            "loaded": False,
            "install_extra": extra,
            "runtime": {"ready": True, "reason": "local MLX runtime available"},
        },
    )


def test_a_configured_provider_that_cannot_run_is_named_not_omitted() -> None:
    cfg = _config(chain=["small_selector", "reviewer"])
    factory = _FakeFactory(
        {
            "small_selector": _runtime_missing("small_selector", "functiongemma"),
            "reviewer": _runtime_missing("reviewer", "hybrid-policy"),
        }
    )

    check = policy_doctor.policy_check(cfg, factory=factory)

    assert check["ok"] is False, "a configured policy whose runtime is absent is a doctor failure"
    assert check["configured"] is True
    names = [p["name"] for p in check["providers"]]
    assert names == ["small_selector", "reviewer"], check
    assert all(p["runnable"] is False for p in check["providers"])
    assert all(p["cause"] == "runtime_missing" for p in check["providers"])
    assert "not installed" in check["detail"] or "cannot run" in check["detail"], check["detail"]


def test_the_failure_carries_the_exact_command_that_fixes_it() -> None:
    cfg = _config(chain=["small_selector"])
    factory = _FakeFactory({"small_selector": _runtime_missing("small_selector", "functiongemma")})

    check = policy_doctor.policy_check(cfg, factory=factory)

    remedy = check["providers"][0]["remedy"]
    assert "install.sh --with-policy" in remedy, remedy
    assert "functiongemma" in remedy, remedy
    assert "install.sh --with-policy" in check["hint"], check["hint"]


def test_an_explicitly_unsupported_host_is_not_told_to_install_the_runtime() -> None:
    provider = _FakeProvider(
        "small_selector",
        {
            "provider": "small_selector",
            "available": False,
            "reason": "FunctionGemma policy requires Apple silicon",
            "install_extra": "functiongemma",
            "runtime": {"ready": False, "reason": "FunctionGemma policy requires Apple silicon"},
        },
    )

    check = policy_doctor.policy_check(
        _config(chain=["small_selector"]), factory=_FakeFactory({"small_selector": provider})
    )

    entry = check["providers"][0]
    assert entry["cause"] == "unsupported_host"
    assert entry["remedy"] is None
    assert "Apple-silicon-only" in check["hint"]


def test_a_missing_base_model_is_not_reported_as_a_missing_runtime() -> None:
    """Different fault, different fix: installing the extra again would fix nothing here."""
    cfg = _config(chain=["small_selector"])
    factory = _FakeFactory({"small_selector": _artifacts_missing("small_selector", "functiongemma")})

    check = policy_doctor.policy_check(cfg, factory=factory)

    entry = check["providers"][0]
    assert entry["cause"] == "artifacts"
    assert entry["runtime_ready"] is True
    assert "model_path" in entry["remedy"], entry["remedy"]
    assert "--with-policy" not in entry["remedy"], entry["remedy"]


def test_a_runnable_provider_passes() -> None:
    cfg = _config(chain=["small_selector"])
    factory = _FakeFactory({"small_selector": _ready("small_selector", "functiongemma")})

    check = policy_doctor.policy_check(cfg, factory=factory)

    assert check["ok"] is True
    assert check["providers"][0]["runnable"] is True


def test_policy_switched_off_is_not_a_failure() -> None:
    cfg = _config(mode="off", chain=["small_selector"])
    factory = _FakeFactory({"small_selector": _runtime_missing("small_selector", "functiongemma")})

    check = policy_doctor.policy_check(cfg, factory=factory)

    assert check["ok"] is True, "an unused optional feature is not breakage"
    assert check["configured"] is False


def test_doctor_never_reads_as_an_instruction_to_enable_the_passive_policy() -> None:
    """`policy.enabled` was measured at ~20s per analyze; doctor must not push it back on."""
    cfg = _config(chain=["small_selector"])
    factory = _FakeFactory({"small_selector": _ready("small_selector", "functiongemma")})

    rendered = cli._render_doctor_pretty(
        {"checks": {"policy": policy_doctor.policy_check(cfg, factory=factory)}}
    )

    lowered = rendered.lower()
    for nudge in (
        "enabled: true",
        "enabled=true",
        "set policy.enabled",
        "enable the policy",
        "turn on the policy",
    ):
        assert nudge not in lowered, f"doctor tells the operator to pay the ~20s tax: {nudge!r}"
    assert "20s" in lowered, "the cost of the passive switch must be discoverable from doctor"


def test_the_check_is_wired_into_the_pretty_report() -> None:
    cfg = _config(chain=["small_selector"])
    factory = _FakeFactory({"small_selector": _runtime_missing("small_selector", "functiongemma")})

    rendered = cli._render_doctor_pretty(
        {"checks": {"policy": policy_doctor.policy_check(cfg, factory=factory)}}
    )

    assert "policy" in rendered
    assert "small_selector" in rendered, "a configured provider must be named, not summarised away"
    assert "FAIL" in rendered
    assert "install.sh --with-policy" in rendered, "the fix must be visible, not just the fault"
    assert "no providers" not in rendered.lower()


def test_doctor_reports_the_policy_on_the_default_config() -> None:
    """The real report, real providers, no fakes — it must contain a policy section."""
    from android_ui_analyser.providers.registry import ProviderFactory

    cfg = Config()
    check = policy_doctor.policy_check(cfg, factory=ProviderFactory(cfg))

    assert check["ok"] is True and check["configured"] is False
    rendered = cli._render_doctor_pretty({"checks": {"policy": check}})
    assert "policy" in rendered


def test_a_chain_name_that_does_not_exist_is_not_blamed_on_the_base_model() -> None:
    """A provider that could not even be constructed must not be reported as a model_path fault."""
    from android_ui_analyser.providers.registry import ProviderFactory

    cfg = _config(chain=["no_such_provider"])
    check = policy_doctor.policy_check(cfg, factory=ProviderFactory(cfg))

    entry = check["providers"][0]
    assert check["ok"] is False
    assert entry["runnable"] is False
    assert entry["cause"] == "unattributed", entry
    assert "model_path" not in str(entry["remedy"])
