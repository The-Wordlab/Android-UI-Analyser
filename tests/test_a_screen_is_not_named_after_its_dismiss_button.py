"""A screen never takes its name from the control that leaves it.

`title_of` returns the topmost non-dynamic text in the upper fifth of the display, which is the
right rule for a toolbar heading and the wrong one for a bottom sheet: that band holds the sheet's
dismiss chrome. `_GENERIC_TITLES` already refuses bland nouns ("create", "new", "details") so
naming falls through to the next source, but it said nothing about dismiss verbs.

The synthetic regression is a sign-up sheet headed "Create your account", with "Continue with
Example ID" and "Maybe later" under it. It must not be recorded in the map as the screen `cancel`;
that name describes the exit control rather than the destination.

The point of a remembered name is that a later run can trust it without re-deriving it.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.memory import _GENERIC_TITLES, _short


@pytest.mark.parametrize(
    "label",
    ["Cancel", "Close", "Dismiss", "Back", "Done", "Skip", "Next", "OK", "X"],
)
def test_an_english_dismiss_control_cannot_name_a_screen(label: str) -> None:
    assert _short(label) in _GENERIC_TITLES


@pytest.mark.parametrize(
    "label",
    ["Cancelar", "Cerrar", "Listo", "Omitir", "Fechar", "Voltar", "Annuler", "Fermer", "Chiudi"],
)
def test_a_translated_dismiss_control_cannot_name_a_screen(label: str) -> None:
    """The app under test is rarely in English; the existing list already carries es/fr/it."""
    assert _short(label) in _GENERIC_TITLES


def test_the_heading_that_should_have_won_is_still_eligible() -> None:
    assert _short("Create your account") not in _GENERIC_TITLES


@pytest.mark.parametrize(
    "label",
    ["Settings", "Your catalog", "Search", "Checkout", "Backup", "Nextdoor"],
)
def test_a_real_heading_is_not_caught_by_the_new_words(label: str) -> None:
    """`_short` slugs whole words, so "Backup" must not be swallowed by "back"."""
    assert _short(label) not in _GENERIC_TITLES
