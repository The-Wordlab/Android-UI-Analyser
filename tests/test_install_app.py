"""``Engine.install_app`` — the idempotence rules, the destructive gate, and the readback.

Everything here runs against a fake adapter, so it also demonstrates that the engine reaches an
app install purely through the platform contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from android_ui_analyser.config import Config
from android_ui_analyser.device import Device
from android_ui_analyser.engine import Engine, _install_versions_differ
from android_ui_analyser.errors import DeviceError, UsageError
from android_ui_analyser.platforms import AppBundle, InstalledApp
from android_ui_analyser.platforms.base import NormalizedTree, PlatformAdapter
from android_ui_analyser.schema import DeviceInfo
from conftest import FakeDevice

BUNDLE = AppBundle(app_id="com.example.app", version_name="2.1.0", version_code="7")


class FakeInstallPlatform(PlatformAdapter):
    name = "fake-install"
    capabilities = frozenset({"ui.tree", "app.install"})

    def __init__(
        self,
        config: Config,
        *,
        present: InstalledApp | None = None,
        bundle: AppBundle = BUNDLE,
        lands: bool = True,
        warning: str | None = None,
    ) -> None:
        super().__init__(config)
        self.bundle = bundle
        self.warning = warning
        self.lands = lands
        self.state = present or InstalledApp(app_id=bundle.app_id, installed=False)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def connect(self, target_id: str | None = None) -> Device:
        raise AssertionError("the engine is given its runtime directly")

    def list_targets(self) -> list[DeviceInfo]:
        return []

    def normalize_tree(
        self,
        raw_tree: str,
        screen_size: tuple[int, int],
        *,
        ignored_app_ids: Sequence[str] = (),
    ) -> NormalizedTree:
        return NormalizedTree(elements=[])

    def inspect_app_bundle(self, bundle: Path) -> AppBundle:
        self.calls.append(("inspect", {"bundle": str(bundle)}))
        return self.bundle

    def installed_app(self, runtime: Device, app_id: str) -> InstalledApp:
        self.calls.append(("installed", {"app_id": app_id}))
        return self.state

    def install_app_bundle(
        self,
        runtime: Device,
        bundle: Path,
        *,
        replace: bool = True,
        grant_permissions: bool = False,
        timeout_s: float = 300.0,
    ) -> None:
        self.calls.append(
            ("install", {"replace": replace, "grant": grant_permissions, "timeout_s": timeout_s})
        )
        if self.lands:
            self.state = InstalledApp(
                app_id=self.bundle.app_id,
                installed=True,
                version_name=self.bundle.version_name,
                version_code=self.bundle.version_code,
            )

    def uninstall_app(self, runtime: Device, app_id: str) -> None:
        self.calls.append(("uninstall", {"app_id": app_id}))
        self.state = InstalledApp(app_id=app_id, installed=False)

    def install_persistence_warning(self, runtime: Device) -> str | None:
        return self.warning


def make_engine(tmp_path, **kwargs) -> tuple[Engine, FakeInstallPlatform, FakeDevice]:
    cfg = Config.model_validate(
        {
            "memory": {"enabled": False},
            "cache": {"dir": str(tmp_path / "cache")},
            "perf": {"prefetch": False},
            "lease": {"enabled": False},
        }
    )
    platform = FakeInstallPlatform(cfg, **kwargs)
    runtime = FakeDevice(package="com.example.app")
    return Engine(cfg, device=runtime, platform=platform), platform, runtime


def apk(tmp_path) -> str:
    path = tmp_path / "example-debug.apk"
    path.write_bytes(b"stub - the fake adapter reads identity, not bytes")
    return str(path)


def install_args(platform: FakeInstallPlatform) -> dict[str, object]:
    """The one ``install`` entry's kwargs. Not ``calls[-1]`` — a readback follows the push."""

    return next(args for name, args in platform.calls if name == "install")


# --------------------------------------------------------------- idempotence


def test_a_missing_app_is_installed(tmp_path) -> None:
    engine, platform, _ = make_engine(tmp_path)

    result = engine.install_app(apk(tmp_path))

    assert result.ok is True
    assert result.action == "app-install"
    assert result.app_install is not None
    assert result.app_install["pushed"] is True
    assert result.app_install["reason"] == "missing"
    assert result.app_install["package"] == "com.example.app"
    assert result.app_install["version_name"] == "2.1.0"
    # An install changes nothing on screen, so no observation is folded in.
    assert result.observation is None
    assert [name for name, _ in platform.calls].count("install") == 1


def test_the_same_version_already_present_is_left_alone(tmp_path) -> None:
    engine, platform, _ = make_engine(
        tmp_path,
        present=InstalledApp("com.example.app", True, version_name="2.1.0", version_code="7"),
    )

    result = engine.install_app(apk(tmp_path))

    assert result.app_install is not None
    assert result.app_install["pushed"] is False
    assert result.app_install["reason"] == "already-present"
    assert "install" not in [name for name, _ in platform.calls]


def test_a_different_version_still_installs_under_if_needed(tmp_path) -> None:
    engine, platform, _ = make_engine(
        tmp_path,
        present=InstalledApp("com.example.app", True, version_name="2.0.0", version_code="6"),
    )

    result = engine.install_app(apk(tmp_path))

    assert result.app_install is not None
    assert result.app_install["reason"] == "version-differs"
    assert result.app_install["pushed"] is True


def test_reinstall_pushes_over_a_matching_version_without_removing_data(tmp_path) -> None:
    engine, platform, _ = make_engine(
        tmp_path,
        present=InstalledApp("com.example.app", True, version_name="2.1.0", version_code="7"),
    )

    result = engine.install_app(apk(tmp_path), mode="reinstall")

    assert result.app_install is not None
    assert result.app_install["reason"] == "reinstall-requested"
    assert result.app_install["uninstalled_first"] is False
    assert "uninstall" not in [name for name, _ in platform.calls]
    assert install_args(platform)["replace"] is True


def test_fresh_uninstalls_first_and_installs_without_replace(tmp_path) -> None:
    engine, platform, _ = make_engine(
        tmp_path,
        present=InstalledApp("com.example.app", True, version_name="2.1.0", version_code="7"),
    )

    result = engine.install_app(apk(tmp_path), mode="fresh", confirmed=True)

    assert result.app_install is not None
    assert result.app_install["uninstalled_first"] is True
    names = [name for name, _ in platform.calls]
    assert names.index("uninstall") < names.index("install")
    # Nothing is there to replace after an uninstall, so `-r` must not be requested.
    assert install_args(platform)["replace"] is False


def test_an_unknown_mode_is_rejected_before_the_device_is_touched(tmp_path) -> None:
    engine, platform, _ = make_engine(tmp_path)

    with pytest.raises(UsageError, match="unknown install mode"):
        engine.install_app(apk(tmp_path), mode="clean")

    assert platform.calls == []


# --------------------------------------------------------------- gates and failures


def test_fresh_refuses_to_wipe_app_data_without_confirmation(tmp_path) -> None:
    engine, platform, _ = make_engine(
        tmp_path,
        present=InstalledApp("com.example.app", True, version_name="2.1.0", version_code="7"),
    )

    with pytest.raises(UsageError, match="ALL its data"):
        engine.install_app(apk(tmp_path), mode="fresh")

    assert "uninstall" not in [name for name, _ in platform.calls]
    assert "install" not in [name for name, _ in platform.calls]


def test_fresh_needs_no_confirmation_when_there_is_nothing_to_lose(tmp_path) -> None:
    engine, platform, _ = make_engine(tmp_path)

    result = engine.install_app(apk(tmp_path), mode="fresh")

    assert result.app_install is not None
    assert result.app_install["uninstalled_first"] is False
    assert result.app_install["reason"] == "missing"


def test_a_package_that_the_bundle_does_not_declare_is_refused(tmp_path) -> None:
    engine, platform, _ = make_engine(tmp_path)

    with pytest.raises(UsageError, match="declares package"):
        engine.install_app(apk(tmp_path), package="com.example.other")

    assert "install" not in [name for name, _ in platform.calls]


def test_an_install_the_package_manager_never_registered_is_a_failure(tmp_path) -> None:
    engine, _, _ = make_engine(tmp_path, lands=False)

    with pytest.raises(DeviceError) as raised:
        engine.install_app(apk(tmp_path))

    assert raised.value.code == "install_unverified"


def test_a_platform_without_the_capability_refuses_explicitly(tmp_path) -> None:
    engine, platform, _ = make_engine(tmp_path)
    platform.capabilities = frozenset({"ui.tree"})

    with pytest.raises(DeviceError) as raised:
        engine.install_app(apk(tmp_path))

    assert raised.value.code == "unsupported_capability"
    assert platform.calls == []


# --------------------------------------------------------------- reporting


def test_a_disposable_target_says_so_on_the_result(tmp_path) -> None:
    engine, _, _ = make_engine(tmp_path, warning="emulator-5566 was booted -read-only")

    result = engine.install_app(apk(tmp_path))

    assert result.app_install is not None
    assert result.app_install["persists"] is False
    assert result.note is not None
    assert "read-only" in result.note


def test_a_skipped_install_claims_nothing_about_persistence(tmp_path) -> None:
    engine, _, _ = make_engine(
        tmp_path,
        present=InstalledApp("com.example.app", True, version_name="2.1.0", version_code="7"),
        warning="emulator-5566 was booted -read-only",
    )

    result = engine.install_app(apk(tmp_path))

    assert result.note is None
    assert result.app_install is not None
    assert "persists" not in result.app_install


def test_grant_still_applies_when_the_push_was_skipped(tmp_path) -> None:
    # `-g` only affects the install itself, so an idempotent skip would otherwise silently drop
    # the caller's permission request.
    engine, _, runtime = make_engine(
        tmp_path,
        present=InstalledApp("com.example.app", True, version_name="2.1.0", version_code="7"),
    )

    engine.install_app(apk(tmp_path), grant_permissions=True)

    assert any(call[0] == "grant_permissions" for call in runtime.calls)


def test_a_new_build_invalidates_the_previous_build_s_cached_reads(tmp_path) -> None:
    engine, _, _ = make_engine(tmp_path)
    engine._version_cache["com.example.app"] = "2.0.0"
    engine._last_hierarchy_hash = "stale"

    engine.install_app(apk(tmp_path))

    assert "com.example.app" not in engine._version_cache
    assert engine._last_hierarchy_hash is None
    assert engine._last_analyze_result is None


def test_the_engine_passes_its_millisecond_budget_down_as_seconds(tmp_path) -> None:
    engine, platform, _ = make_engine(tmp_path)

    engine.install_app(apk(tmp_path), timeout_ms=90_000)

    assert install_args(platform)["timeout_s"] == pytest.approx(90.0)


# --------------------------------------------------------------- version comparison


@pytest.mark.parametrize(
    ("installed", "bundle", "differs"),
    [
        (InstalledApp("a", True, "2.1.0", "7"), AppBundle("a", "2.1.0", "7"), False),
        (InstalledApp("a", True, "2.1.0", "6"), AppBundle("a", "2.1.0", "7"), True),
        (InstalledApp("a", True, "2.0.0", "7"), AppBundle("a", "2.1.0", "7"), True),
        # A suffixed name is normal and unorderable; only difference is knowable.
        (InstalledApp("a", True, "1.0-rc2+abc", "7"), AppBundle("a", "1.0-rc2+abc", "7"), False),
        # Fails open: an unanswerable "is this the same build?" must not resolve to "yes, skip".
        (InstalledApp("a", True, None, None), AppBundle("a", "2.1.0", "7"), True),
        # Nothing to compare against: the bundle declares no version either.
        (InstalledApp("a", True, None, None), AppBundle("a", None, None), False),
    ],
)
def test_version_difference_is_compared_not_ordered(installed, bundle, differs: bool) -> None:
    assert _install_versions_differ(installed, bundle) is differs
