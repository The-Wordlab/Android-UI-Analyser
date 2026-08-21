"""Locale awareness: ``meta.device_locale`` + the cross-locale text-lookup bridge.

The failure mode this pins down: a caller's query written in one language drives a
device whose UI renders labels in another ('Edit basket' vs 'Editar cesta'), and
asserting the query's literal misses every time. The tool keeps the caller oriented by
reporting the device locale on every analyze and explaining text misses on
``has``/``wait``/``scroll-to`` — and, once ``explore mine`` has harvested the app's
string resources, bridges the languages by itself. Language-neutral throughout: any
query language crossed with any device locale.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.device import Device, Uiautomator2Device, parse_locale
from android_ui_analyser.engine import Engine
from android_ui_analyser.explore import _values_dir_locale, mine_strings
from android_ui_analyser.guide import render_json, render_markdown
from android_ui_analyser.memory import AppStrings
from conftest import FakeDevice, make_engine

ON_SCREEN_ES = {"Editar cesta": (10, 10, 200, 60)}

APP_STRINGS = AppStrings(
    package="com.test.app",
    locales=["default", "es-ES", "pt-BR", "it", "fr"],
    entries={
        "basket_edit": {
            "default": "Edit basket",
            "es-ES": "Editar cesta",
            "pt-BR": "Editar cesto",
            "it": "Modifica cestino",
            "fr": "Modifier le panier",
        },
        "loading_indicator": {"default": "Loading", "es-ES": "Cargando"},
    },
)


def engine_with_strings(
    locale: str | None, text_index: dict[str, tuple[int, int, int, int]] | None = None
) -> Engine:
    engine = make_engine(device=FakeDevice(locale=locale, text_index=text_index or {}))
    store = engine._memory
    assert store is not None
    store.save_strings(APP_STRINGS)
    return engine


# ------------------------------------------------------------------------ parse_locale


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("null", None),
        ("NULL", None),
        ("undefined", None),
        ("es-ES", "es-ES"),
        ("en-US,es-ES", "en-US"),
        (" pt-BR \n", "pt-BR"),
    ],
)
def test_parse_locale(raw: str | None, expected: str | None) -> None:
    assert parse_locale(raw) == expected


def test_platform_neutral_locale_contract_does_not_probe_a_native_shell() -> None:
    device = FakeDevice()
    assert Device.device_locale(device) is None
    assert not any(name == "shell" for name, _ in device.calls)


def test_android_locale_probe_uses_fallback_order_and_is_memoized() -> None:
    device = object.__new__(Uiautomator2Device)
    device._device_locale_memo = None
    device._device_locale_read = False
    calls: list[str] = []

    def shell(command: str) -> str:
        calls.append(command)
        return "es-ES" if command == "settings get system system_locales" else ""

    device.shell = shell  # type: ignore[method-assign]
    assert device.device_locale() == "es-ES"
    assert device.device_locale() == "es-ES"
    assert calls == ["getprop persist.sys.locale", "settings get system system_locales"]


# ------------------------------------------------------------------------ analyze meta


def test_analyze_reports_device_locale() -> None:
    engine = make_engine(device=FakeDevice(locale="es-ES"))
    assert engine.analyze().meta.device_locale == "es-ES"


def test_analyze_compact_drops_unknown_locale() -> None:
    engine = make_engine(device=FakeDevice())
    result = engine.analyze()
    assert result.meta.device_locale is None
    assert "device_locale" not in result.as_dict("compact")["meta"]


def test_analyze_query_reports_device_locale() -> None:
    engine = make_engine(device=FakeDevice(locale="es-ES"))
    assert engine.analyze(query="the edit button").meta.device_locale == "es-ES"


# ------------------------------------------------------------------------ has (no bridge)


def test_has_text_miss_carries_locale_and_hint() -> None:
    engine = make_engine(device=FakeDevice(locale="es-ES", text_index=ON_SCREEN_ES))
    res = engine.has("Edit basket")
    assert res.found is False
    assert res.device_locale == "es-ES"
    assert res.hint is not None and "es-ES" in res.hint and "--by id" in res.hint


def test_has_id_miss_carries_locale_but_no_hint() -> None:
    engine = make_engine(device=FakeDevice(locale="es-ES"))
    res = engine.has("missingContainer", by="id")
    assert res.found is False
    assert res.device_locale == "es-ES"
    assert res.hint is None


@pytest.mark.parametrize("by", ["id", "rid", "RID"])
def test_has_resource_id_miss_never_suggests_a_translation(by: str) -> None:
    engine = make_engine(device=FakeDevice(locale="es-ES"))
    res = engine.has("missingContainer", by=by)
    assert res.found is False
    assert res.device_locale == "es-ES"
    assert res.hint is None


def test_has_miss_hints_on_english_devices_too() -> None:
    # Language-neutral by design: the query may be written in any language, so an
    # English device is not "safe" — a Spanish query misses there just the same.
    engine = make_engine(device=FakeDevice(locale="en-US"))
    res = engine.has("Editar cesta")
    assert res.found is False
    assert res.device_locale == "en-US"
    assert res.hint is not None and "en-US" in res.hint


def test_has_miss_with_unknown_locale_has_no_hint() -> None:
    engine = make_engine(device=FakeDevice())
    res = engine.has("Edit basket")
    assert res.found is False
    assert res.device_locale is None
    assert res.hint is None


def test_has_hit_carries_no_locale_fields() -> None:
    engine = make_engine(device=FakeDevice(locale="es-ES", text_index=ON_SCREEN_ES))
    res = engine.has("Editar cesta")
    assert res.found is True
    assert res.device_locale is None
    assert res.hint is None


# ------------------------------------------------------------------------ string mining


@pytest.mark.parametrize(
    ("dirname", "expected"),
    [
        ("values", "default"),
        ("values-es", "es"),
        ("values-es-rES", "es-ES"),
        ("values-pt-rBR", "pt-BR"),
        ("values-fil", "fil"),
        ("values-b+sr+Latn", "sr-Latn"),
        ("values-night", None),
        ("values-v21", None),
        ("values-sw600dp", None),
        ("values-land", None),
        ("values-car", None),
    ],
)
def test_values_dir_locale(dirname: str, expected: str | None) -> None:
    assert _values_dir_locale(dirname) == expected


def _write_strings(root: Path, values_dir: str, body: str) -> None:
    d = root / "app" / "src" / "main" / "res" / values_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "strings.xml").write_text(f"<resources>{body}</resources>", encoding="utf-8")


def test_mine_strings_harvests_per_locale(tmp_path: Path) -> None:
    _write_strings(
        tmp_path,
        "values",
        '<string name="basket_edit">Edit basket</string>'
        '<string name="quoted">"  Edit  basket  "</string>'
        "<string name=\"apos\">Don\\'t</string>",
    )
    _write_strings(tmp_path, "values-es-rES", '<string name="basket_edit">Editar cesta</string>')
    _write_strings(tmp_path, "values-night", '<string name="basket_edit">ignored</string>')
    test_dir = tmp_path / "app" / "src" / "test" / "res" / "values"
    test_dir.mkdir(parents=True)
    (test_dir / "strings.xml").write_text(
        '<resources><string name="fixture_only">nope</string></resources>', encoding="utf-8"
    )

    mined = mine_strings(tmp_path)
    assert mined.locales == ["default", "es-ES"]
    assert mined.entries["basket_edit"] == {
        "default": "Edit basket",
        "es-ES": "Editar cesta",
    }
    assert mined.entries["quoted"]["default"] == "Edit  basket"
    assert mined.entries["apos"]["default"] == "Don't"
    assert "fixture_only" not in mined.entries


def test_explore_mine_saves_strings(tmp_path: Path) -> None:
    _write_strings(tmp_path, "values", '<string name="basket_edit">Edit basket</string>')
    _write_strings(tmp_path, "values-es-rES", '<string name="basket_edit">Editar cesta</string>')
    engine = make_engine(device=FakeDevice())

    out = engine.explore_mine(str(tmp_path), package="com.test.app")
    assert out["strings_found"] == 1
    assert out["strings_saved"] == 1
    assert out["string_locales"] == ["default", "es-ES"]
    store = engine._memory
    assert store is not None
    saved = store.load_strings("com.test.app")
    assert saved is not None
    assert saved.entries["basket_edit"]["es-ES"] == "Editar cesta"


# ------------------------------------------------------------------------ locale bridge


@pytest.mark.parametrize(
    ("device_locale", "on_screen"),
    [
        ("es-ES", "Editar cesta"),
        ("es", "Editar cesta"),
        ("pt-BR", "Editar cesto"),
        ("it-IT", "Modifica cestino"),
        ("fr-FR", "Modifier le panier"),
    ],
)
def test_has_bridges_any_locale(device_locale: str, on_screen: str) -> None:
    engine = engine_with_strings(device_locale, {on_screen: (10, 10, 200, 60)})
    res = engine.has("Edit basket")
    assert res.found is True
    assert res.text == on_screen
    assert res.hint is not None and "basket_edit" in res.hint and on_screen in res.hint


def test_has_bridges_reverse_direction() -> None:
    engine = engine_with_strings("en-US", {"Edit basket": (10, 10, 200, 60)})
    res = engine.has("Editar cesta")
    assert res.found is True
    assert res.text == "Edit basket"
    assert res.hint is not None and "basket_edit" in res.hint


def test_has_timeout_polls_the_translated_rendering_without_waiting_on_the_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = engine_with_strings("es-ES", ON_SCREEN_ES)

    def untranslated_only_wait(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the source-language literal must not consume the whole wait budget")

    monkeypatch.setattr(engine.device, "wait_for", untranslated_only_wait)
    res = engine.has("Edit basket", timeout_ms=300)
    assert res.found is True
    assert res.text == "Editar cesta"


def test_exact_device_locale_ranks_before_other_regions() -> None:
    engine = make_engine(device=FakeDevice(locale="es-ES"))
    store = engine._memory
    assert store is not None
    strings = APP_STRINGS.model_copy(deep=True)
    strings.entries["basket_edit"] = {
        "default": "Edit basket",
        "es-AR": "Editar changuito",
        "es-MX": "Editar carrito",
        "es-US": "Editar canasta",
        "es-ES": "Editar cesta",
    }
    store.save_strings(strings)

    assert engine._locale_candidates(engine.device, "Edit basket")[0] == (
        "Editar cesta",
        "es-ES",
        "basket_edit",
    )


def test_has_miss_reports_expected_rendering() -> None:
    engine = engine_with_strings("es-ES")
    res = engine.has("Edit basket")
    assert res.found is False
    assert res.hint is not None
    assert "Editar cesta" in res.hint and "basket_edit" in res.hint
    assert "not on screen" in res.hint


def test_wait_bridges_locale() -> None:
    engine = engine_with_strings("es-ES", {"Editar cesta": (10, 10, 200, 60)})
    res = engine.wait(for_="Edit basket", timeout_ms=300)
    assert res.ok is True
    assert res.detail == "Edit basket"
    assert res.hint is not None and "Editar cesta" in res.hint


def test_wait_miss_names_expected_rendering() -> None:
    engine = engine_with_strings("es-ES")
    res = engine.wait(for_="Edit basket", timeout_ms=1)
    assert res.ok is False
    assert res.detail is not None and "Editar cesta" in res.detail


def test_wait_absent_counts_translated_rendering_as_present() -> None:
    engine = engine_with_strings("es-ES", {"Cargando": (10, 10, 200, 60)})
    res = engine.wait(for_="Loading", timeout_ms=300, absent=True)
    assert res.ok is False


def test_wait_absent_true_when_all_renderings_gone() -> None:
    engine = engine_with_strings("es-ES")
    res = engine.wait(for_="Loading", timeout_ms=300, absent=True)
    assert res.ok is True


def test_scroll_to_bridges_locale() -> None:
    engine = engine_with_strings("es-ES", {"Editar cesta": (10, 10, 200, 60)})
    res = engine.scroll_to("Edit basket", observe=False)
    assert res.ok is True
    assert res.hint is not None and "Editar cesta" in res.hint and "basket_edit" in res.hint


def test_scroll_to_miss_names_expected_rendering() -> None:
    engine = engine_with_strings("es-ES")
    res = engine.scroll_to("Edit basket", observe=False, max_swipes=1)
    assert res.ok is False
    assert res.hint is not None and "Editar cesta" in res.hint


# ------------------------------------------------------------------------ guide


def test_guide_teaches_locale_protocol() -> None:
    md = render_markdown()
    assert "device_locale" in md
    assert "locale-proof" in md
    assert "explore mine" in md
    assert "device_locale" in render_json()["schema_fields"]["meta"]  # type: ignore[index]
