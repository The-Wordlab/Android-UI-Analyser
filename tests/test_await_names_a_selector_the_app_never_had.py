"""An unmet ``await`` said *which* term was missing but never that it could not exist.

A live run waited on ``rid:myAppsLibraryDraftList`` — an id that appears in none of the 216
screens AUA had already mapped for that app. The response said ``unmet:
rid:myAppsLibraryDraftList`` and nothing else, which reads exactly like "not there yet". The
agent invented a second id, then a third, and burned twelve minutes on a screen that could
never satisfy any of them.

The sibling ``wait`` command already answers this well: on timeout it names the closest
candidates on screen. ``await`` — the multi-term wait an agent is told to reach for most — did
not. So an unmet positive ``rid:`` term is now checked against the app map's own id vocabulary,
and a term no mapped screen has ever carried is reported as such, with the nearest ids that do
exist.

Only resource ids are judged. The map keeps text and content-description anchors only when they
are short and non-dynamic, so a missing ``text:`` anchor proves nothing about the app; a missing
id does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from android_ui_analyser.engine import Engine
from android_ui_analyser.memory import AppMap, ScreenRecord
from android_ui_analyser.providers.registry import ProviderFactory
from android_ui_analyser.selectors import unknown_map_rids
from conftest import FakeDevice, make_config

PKG = "com.example.app"

# Every id the mapped app below has ever shown. Thick enough that an id missing from it is a
# statement about the app, not about a barely-explored map.
_VOCABULARY = frozenset(
    {
        "librarytabdrafts",
        "homefeed",
        "buttonsettings",
        "searchfield",
        "detailheader",
        "profileavatar",
    }
)


_WHEN = "2026-01-01T00:00:00+00:00"


def _screen(name: str, anchors: list[str]) -> ScreenRecord:
    return ScreenRecord(
        name=name,
        signature=name,
        first_seen=_WHEN,
        last_seen=_WHEN,
        last_verified=_WHEN,
        anchors=anchors,
    )


def _mapped_app() -> AppMap:
    """A map thick enough to make an absence claim honest."""
    app = AppMap(package=PKG)
    app.screens["library"] = _screen("library", ["id:librarytabdrafts", "tx:drafts"])
    app.screens["home"] = _screen("home", ["id:homefeed", "tx:home"])
    app.screens["settings"] = _screen("settings", ["id:buttonsettings"])
    app.screens["search"] = _screen("search", ["id:searchfield"])
    app.screens["detail"] = _screen("detail", ["id:detailheader"])
    app.screens["profile"] = _screen("profile", ["id:profileavatar"])
    return app


# ------------------------------------------------------------------ the pure judgement


def test_an_id_no_mapped_screen_carries_is_named_as_impossible() -> None:
    vocabulary = {
        "librarytabdrafts",
        "homefeed",
        "buttonsettings",
        "searchfield",
        "detailheader",
        "profileavatar",
    }
    found = unknown_map_rids(["rid:libraryDraftList"], vocabulary)
    assert [row["term"] for row in found] == ["rid:libraryDraftList"]
    # The nearest real id is the actionable half — a bare "does not exist" still leaves the
    # caller guessing, which is the loop this exists to break.
    assert "librarytabdrafts" in found[0]["nearest"]


def test_an_id_the_map_knows_is_left_alone() -> None:
    assert unknown_map_rids(["rid:librarytabdrafts"], _VOCABULARY) == []


def test_a_package_qualified_id_is_compared_on_its_tail() -> None:
    assert unknown_map_rids([f"rid:{PKG}:id/librarytabdrafts"], _VOCABULARY) == []


def test_only_resource_ids_are_judged() -> None:
    # Text and desc anchors are stored selectively, so their absence is not evidence.
    assert unknown_map_rids(["text:Some long body copy", "desc:Close"], _VOCABULARY) == []


def test_a_negated_term_is_never_called_impossible() -> None:
    # `!rid:x` unmet means x IS present. Calling it non-existent would be backwards.
    assert unknown_map_rids(["!rid:neverHeardOfIt"], _VOCABULARY) == []


def test_a_thin_map_makes_no_absence_claim() -> None:
    # Two screens is not a survey of the app; claiming an id cannot exist would be a guess.
    assert unknown_map_rids(["rid:whatever"], {"homefeed"}, min_vocabulary=3) == []


# ------------------------------------------------------------------ wired into await


def _engine(tmp_path: Path, device: FakeDevice) -> Engine:
    cfg = make_config(memory={"dir": str(tmp_path / "home")}, daemon={"enabled": False})
    return Engine(cfg, device=device, factory=ProviderFactory(cfg))


@pytest.fixture()
def mapped_engine(tmp_path: Path) -> Engine:
    engine = _engine(tmp_path, FakeDevice(package=PKG))
    assert engine._memory is not None
    engine._memory.save(_mapped_app())
    return engine


def test_an_unmet_await_reports_the_id_the_app_never_had(mapped_engine: Engine) -> None:
    result = mapped_engine._await_result(
        "timeout",
        [
            {"term": "rid:libraryDraftList", "present": False, "satisfied": False},
            {"term": "tx:drafts", "present": True, "satisfied": True},
        ],
        started_at=0.0,
        checks=8,
        origin=(PKG, ".Main"),
        now=(PKG, ".Main"),
        observe=False,
    )
    assert result.unknown_selectors, "an id in no mapped screen must be reported"
    assert result.unknown_selectors[0]["term"] == "rid:libraryDraftList"
    # It must be readable without parsing structured fields, too — the CLI prints detail.
    assert "no mapped screen" in (result.detail or "")


def test_a_satisfied_await_says_nothing_about_selectors(mapped_engine: Engine) -> None:
    result = mapped_engine._await_result(
        "satisfied",
        [{"term": "id:librarytabdrafts", "present": True, "satisfied": True}],
        started_at=0.0,
        checks=1,
        origin=(PKG, ".Main"),
        now=(PKG, ".Main"),
        observe=False,
    )
    assert result.unknown_selectors is None
