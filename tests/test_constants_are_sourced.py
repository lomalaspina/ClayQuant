"""Every constant that reaches a reported number must say where it came from.

A number nobody can trace is a number nobody can check, and this program's whole
claim is that its answers rest on measured standards and published structures.
Two faults found by hand make the point: the chlorite composition axis excluded
the structure it was built around while naming an invented point as that
structure, and the heated mount was labelled 500 C where every bracket in the
program was measured at 550.  Neither was a wrong formula; both were a number
that said less about itself than it should have.

So this is a documentation test, enforced rather than encouraged.  It cannot
check that a docstring is *true* - only a measurement does that - but it can stop
a bare literal appearing with nothing attached to it.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "src" / "clayquant"


def module_constants():
    """Every public module-level constant, with the docstring under it."""
    for path in sorted(ROOT.rglob("*.py")):
        body = ast.parse(path.read_text()).body
        for index, node in enumerate(body):
            name = None
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                name = node.targets[0].id
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
            if not name or not name.isupper() or name.startswith("_"):
                continue
            doc = ""
            following = body[index + 1] if index + 1 < len(body) else None
            if (isinstance(following, ast.Expr)
                    and isinstance(following.value, ast.Constant)
                    and isinstance(following.value.value, str)):
                doc = following.value.value
            yield path.relative_to(ROOT).as_posix(), name, node.lineno, doc


ALL = list(module_constants())


def test_the_scan_finds_the_constants_at_all():
    """A guard on the guard: a scan that silently matches nothing passes everything."""
    assert len(ALL) > 100
    names = {name for _, name, _, _ in ALL}
    assert {"CHLORITE_IRON", "HOST_THICKNESSES", "DISCRETE_THICKNESSES"} <= names


@pytest.mark.parametrize("where,name,line,doc",
                         ALL, ids=[f"{w}:{n}" for w, n, _, _ in ALL])
def test_every_constant_is_documented(where, name, line, doc):
    assert doc.strip(), (
        f"{where}:{line} defines {name} with no docstring. A constant that reaches "
        f"a reported number has to say what it is and where its value came from - "
        f"a measurement, a publication, a derivation, or an explicit statement "
        f"that it is a judgement and what changes if it moves."
    )
