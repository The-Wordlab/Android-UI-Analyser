"""Nothing the tool says may teach a name the tool refuses.

c6030bf removed the 17 short action aliases so a wrong name fails loudly instead of quietly
returning less. It did not sweep the text: `aua tap-and-analyze --help` still told the caller to
write `aua tap 9`, and the selector-miss hint still said `aua tap --by id homeTabBROWSE` — both of
which now exit 2 with `removed_command`. That is worse than the original problem, because the
caller did the careful thing (read the help, read the hint) and was handed the dead name anyway.

Two later fixes hit the same trap while fixing it: an error added to correct `--text` printed
`aua input` in its example, and the redundant-analyze warning pointed at `aua wait`. Hence a test
rather than another round of grep.

Scope is `src/` — the strings the tool emits or renders. Test docstrings are not user-facing, and
`test_removed_short_aliases.py` has to name the dead commands to assert they are dead.
"""

from __future__ import annotations

import re
from pathlib import Path

from android_ui_analyser.cli import _REMOVED_ACTION_ALIASES

_SRC = Path(__file__).resolve().parent.parent / "src" / "android_ui_analyser"

# `aua`, any global flags, then a bare removed name not already followed by `-and-analyze`
# (or any other suffix, so `scroll-to` is not read as a stale `scroll`).
_SPOKEN = re.compile(
    r"aua (?:--[a-z-]+(?:[= ][^ `]+)? )*("
    + "|".join(sorted(_REMOVED_ACTION_ALIASES, key=len, reverse=True))
    + r")(?![-a-z])"
)


def test_no_source_string_names_a_removed_command() -> None:
    offences: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for match in _SPOKEN.finditer(line):
                offences.append(f"{path.relative_to(_SRC)}:{lineno}: {match.group(0)}")
    assert not offences, "these teach a name that exits 2:\n" + "\n".join(offences)


def test_the_replacement_each_removed_name_names_actually_exists() -> None:
    """The sweep is only safe if every `X` really does have an `X-and-analyze`."""
    from typer.main import get_command

    from android_ui_analyser.cli import app

    real = set(get_command(app).commands)
    missing = {f"{old}-and-analyze" for old in _REMOVED_ACTION_ALIASES} - real
    assert not missing, f"removed names point at commands that do not exist: {missing}"
