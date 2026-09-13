"""Recorded knowledge reaches the agent when the goal needs it, not only through ``aua about``.

A live run on a real application had three accurate facts in its knowledge store: where a
setting really lives, that a fresh install force-overwrites it, and a recipe to set it. The
goal "switch the app theme" started with ``candidates: []`` and "no verified route, matching
flow, or relevant deeplink", and the agent re-derived everything by hand. The store was
pull-only; nothing asked it a question at the moment the goal was known.

``session start`` now returns ``relevant_knowledge`` ranked against the goal. Aliases bind a
fact to goal phrasings the way flows bind on theirs; body text only adds a capped hint so a
long claim cannot match everything. Stale and out-of-scope facts stay out.
"""

from __future__ import annotations

from android_ui_analyser.memory import (
    AppMap,
    AppMemoryStore,
    KnowledgeItem,
    KnowledgeScope,
    clean_knowledge_aliases,
)
from android_ui_analyser.schema import AnalyzeResult, Meta, Screen
from android_ui_analyser.session import (
    KNOWLEDGE_MATCH_THRESHOLD,
    knowledge_match_score,
    plan_goal_session,
    relevant_knowledge,
)
from conftest import make_config

_PKG = "com.example.app"
_NOW = "2026-09-13T10:00:00+00:00"


def _item(
    text: str,
    *,
    kind: str = "claim",
    name: str | None = None,
    aliases: list[str] | None = None,
    status: str = "accepted",
    context_id: str | None = None,
    app_version: str | None = None,
    ident: str | None = None,
) -> KnowledgeItem:
    return KnowledgeItem(
        id=ident or f"knowledge_{abs(hash((text, name))) % 10**8:08x}",
        kind=kind,  # type: ignore[arg-type]
        text=text,
        name=name,
        aliases=aliases or [],
        scope=KnowledgeScope(package=_PKG, context_id=context_id, app_version=app_version),
        status=status,  # type: ignore[arg-type]
        created_at=_NOW,
        last_verified=_NOW if status == "accepted" else None,
    )


THEME_CLAIM = _item(
    "The appearance setting is not a feature flag and not shared preferences; it lives in a "
    "DataStore file under key appearance_mode (0=system, 1=light, 2=dark).",
    aliases=["change theme", "switch theme", "set theme", "dark mode", "light mode"],
    ident="knowledge_theme",
)
FRESH_INSTALL_CLAIM = _item(
    "A fresh install forces dark appearance during first launch, so set the theme only after "
    "onboarding has finished.",
    aliases=["change theme", "fresh install theme"],
    ident="knowledge_fresh",
)
RECIPE = _item(
    "Open Settings, tap the Theme row, choose the option; the chosen row appends 'Selected'.",
    kind="recipe",
    name="set-theme-through-settings",
    ident="knowledge_recipe",
)
UNRELATED = _item("Bookings need a warm session before the deeplink resolves.", aliases=["open bookings"], ident="knowledge_bookings")
STALE = _item("Old theme claim that was superseded.", aliases=["change theme"], status="stale", ident="knowledge_stale")
OTHER_CONTEXT = _item("Theme row is hidden in the experiment arm.", aliases=["change theme"], context_id="flags-other", ident="knowledge_ctx")
OLD_VERSION = _item("Theme lived in shared prefs in 1.0.", aliases=["change theme"], app_version="1.0", ident="knowledge_old")


def _app(*items: KnowledgeItem, app_version: str | None = "2.0") -> AppMap:
    return AppMap(package=_PKG, app_version=app_version, knowledge=list(items))


def _observation() -> AnalyzeResult:
    return AnalyzeResult(
        screen=Screen(width=1080, height=2400, package=_PKG, source="hierarchy"),
        elements=[],
        meta=Meta(duration_ms=12, tier_used="hierarchy", path="hierarchy", known_screen="home", device_serial="goal-emulator"),
    )


# ------------------------------------------------------------------------------ the scorer


def test_aliases_bind_a_fact_to_goal_phrasings() -> None:
    assert knowledge_match_score("change theme", THEME_CLAIM) >= KNOWLEDGE_MATCH_THRESHOLD
    assert knowledge_match_score("Open Settings and switch the app theme to Dark mode.", THEME_CLAIM) >= KNOWLEDGE_MATCH_THRESHOLD


def test_body_text_alone_is_a_capped_hint_so_long_claims_do_not_match_everything() -> None:
    wordy = _item(" ".join(f"word{i}" for i in range(200)) + " settings theme dark mode light system")
    # Five goal terms appear in the text; the text contribution stops at three terms (15).
    assert knowledge_match_score("switch settings theme to dark mode", wordy) == 15
    assert knowledge_match_score("switch settings theme to dark mode", wordy) < KNOWLEDGE_MATCH_THRESHOLD


def test_a_recipe_name_counts_like_an_alias() -> None:
    assert knowledge_match_score("set theme through settings", RECIPE) >= KNOWLEDGE_MATCH_THRESHOLD


# -------------------------------------------------------------------------- the ranked view


def test_relevant_knowledge_ranks_matches_and_excludes_stale_and_out_of_scope() -> None:
    app = _app(UNRELATED, STALE, OTHER_CONTEXT, OLD_VERSION, RECIPE, FRESH_INSTALL_CLAIM, THEME_CLAIM)
    hits = relevant_knowledge(app, "change theme", context_id="default")
    assert {hit["id"] for hit in hits} == {"knowledge_theme", "knowledge_fresh"}
    assert hits[0]["score"] >= hits[1]["score"], "best match first"
    assert all(hit["kind"] == "claim" and hit["aliases"][0] == "change theme" for hit in hits)
    assert set(hits[0]) == {"id", "kind", "name", "aliases", "text", "source", "last_verified", "score"}


def test_context_scoped_knowledge_surfaces_in_its_own_context_only() -> None:
    app = _app(THEME_CLAIM, OTHER_CONTEXT)
    default = relevant_knowledge(app, "change theme", context_id="default")
    other = relevant_knowledge(app, "change theme", context_id="flags-other")
    assert [hit["id"] for hit in default] == ["knowledge_theme"]
    assert {hit["id"] for hit in other} == {"knowledge_theme", "knowledge_ctx"}


def test_long_text_is_cut_and_limit_applies() -> None:
    long_item = _item("theme " * 200, aliases=["change theme"], ident="knowledge_long")
    hits = relevant_knowledge(_app(long_item, THEME_CLAIM), "change theme", limit=1)
    assert len(hits) == 1
    assert len(hits[0]["text"]) <= 400 and hits[0]["text"].endswith("…")


def test_unrelated_goal_returns_nothing() -> None:
    assert relevant_knowledge(_app(THEME_CLAIM, FRESH_INSTALL_CLAIM, RECIPE), "open bookings") == []


# ------------------------------------------------------------------------- session start


def test_session_plan_carries_relevant_knowledge_and_says_so() -> None:
    plan = plan_goal_session("switch the app theme to light", _observation(), app=_app(THEME_CLAIM, UNRELATED))
    assert [hit["id"] for hit in plan.relevant_knowledge] == ["knowledge_theme"]
    assert plan.candidates == []
    assert any("relevant_knowledge" in warning for warning in plan.warnings)
    assert plan.model_dump(mode="json")["relevant_knowledge"][0]["kind"] == "claim"


def test_session_plan_without_a_match_is_unchanged() -> None:
    plan = plan_goal_session("open bookings", _observation(), app=_app(THEME_CLAIM))
    assert plan.relevant_knowledge == []
    assert not any("relevant_knowledge" in warning for warning in plan.warnings)


def test_knowledge_from_another_package_is_not_offered() -> None:
    foreign = AppMap(package="com.example.other", knowledge=[THEME_CLAIM])
    plan = plan_goal_session("change theme", _observation(), app=foreign)
    assert plan.relevant_knowledge == []


# ------------------------------------------------------------------------------- the store


def test_remember_knowledge_persists_aliases_and_merges_on_re_add(tmp_path) -> None:
    store = AppMemoryStore(make_config(memory={"dir": str(tmp_path)}).memory)
    first = store.remember_knowledge(_PKG, kind="claim", text="Theme lives in a DataStore file.", aliases=["change theme", " Change Theme ", ""])
    assert first is not None and first.aliases == ["change theme"]
    again = store.remember_knowledge(_PKG, kind="claim", text="Theme lives in a DataStore file.", aliases=["switch theme"])
    assert again is not None and again.id == first.id and again.aliases == ["change theme", "switch theme"]
    reloaded = store.load(_PKG)
    assert reloaded is not None and reloaded.knowledge[0].aliases == ["change theme", "switch theme"]
    assert relevant_knowledge(reloaded, "switch theme")[0]["id"] == first.id


def test_alias_cleaning_trims_dedupes_and_drops_empty() -> None:
    assert clean_knowledge_aliases(["  set  theme ", "set theme", "SET THEME", "", None]) == ["set theme"]  # type: ignore[list-item]
    assert clean_knowledge_aliases(None) == []
