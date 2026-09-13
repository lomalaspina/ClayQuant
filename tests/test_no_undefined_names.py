"""Every name the source uses must exist.

This catches one specific, expensive mistake: a function used in a module that
never imports it.  Nothing fails until the line runs, and the lines most likely
to carry it are the ones a unit test does not reach - a button's callback in the
interface, an error path, a branch behind a file dialog.  ``resolve_user_path``
reached a user that way, as ``name 'resolve_user_path' is not defined`` on the
button that loads the phase database, with the same fault waiting on the button
that loads the library and on the one that exports the results.

A whole-program check costs a few milliseconds and finds all three at once.
Only genuinely undefined names fail the test; unused imports and other style
complaints are left to whoever is reading the code.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pyflakes_checker = pytest.importorskip(
    "pyflakes.checker", reason="pyflakes is not installed (pip install -e '.[dev]')"
)
pyflakes_messages = pytest.importorskip("pyflakes.messages")

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(
    path
    for directory in ("src", "tests", "scripts")
    for path in (ROOT / directory).rglob("*.py")
)

UNDEFINED = (
    pyflakes_messages.UndefinedName,
    pyflakes_messages.UndefinedLocal,
    pyflakes_messages.UndefinedExport,
)


def test_there_are_python_files_to_check():
    assert len(SOURCES) > 10, "the source tree was not found"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_undefined_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    checker = pyflakes_checker.Checker(tree, filename=str(path))
    problems = [
        f"{path.relative_to(ROOT)}:{message.lineno}: {message.message % message.message_args}"
        for message in checker.messages
        if isinstance(message, UNDEFINED)
    ]
    assert not problems, "\n".join(problems)
