"""`--by rid` searched the label instead of the resource-id, and timed out saying so.

Found by a sweep lane: `aua wait --for containerLogin --by rid` timed out on a screen that a
screenshot and a fresh `analyze` both confirmed was present, while a flow's
`wait_for: {id: containerLogin}` matched the same screen without trouble. The lane concluded
the *screen* was wrong, which cost it time and produced a misleading handoff note.

The cause is not two matchers behind one vocabulary, which is what the deferred-fix list
recorded. There is one matcher, and `by` was not validated: `_fields_for` mapped only "id"
and "desc", so "rid" fell through to the `["text", "description"]` default and the lookup
searched the *label* for the string "containerLogin". Meanwhile the flow form sets by="id"
and hits the resourceId branch - hence one surface working and the other not.

`rid` is how the resource-id is spelled everywhere else in the vocabulary - the `--rid` flag,
the selector dict key, `_SELECTOR_FIELDS` - so reaching for `--by rid` is the natural mistake,
and the two spellings must mean the same thing.

The other half — an *unrecognised* `by` degrading to a text search instead of being refused —
was deliberately left out of this change because it alters behaviour for every existing
`has`/`wait` caller. It landed separately; see `test_an_unknown_by_token_is_refused.py`, which
is also where the "no legitimate call regresses" enumeration lives.
"""

from __future__ import annotations

import pytest

from android_ui_analyser.cli import _BY_KINDS
from android_ui_analyser.device import Uiautomator2Device


def _fields(by: str) -> list[str]:
    return Uiautomator2Device._fields_for(object.__new__(Uiautomator2Device), by)


@pytest.mark.parametrize("by", ["id", "rid", "RID", "Id"])
def test_both_spellings_of_resource_id_query_the_resource_id(by):
    assert _fields(by) == ["resourceId"], f"--by {by} must not fall back to a text search"


def test_text_and_desc_are_unchanged():
    assert _fields("desc") == ["description"]
    assert _fields("text") == ["text", "description"]


def test_an_absent_by_still_defaults_to_text():
    assert _fields("") == ["text", "description"]


def test_the_selector_surface_accepts_rid_too():
    """`tap --by rid` used to be refused outright while `wait --by rid` quietly text-searched."""
    assert _BY_KINDS["rid"] == _BY_KINDS["id"] == "rid"
