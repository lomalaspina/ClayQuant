"""The library must be able to calculate the structure it is built from.

The chlorite entries come from ICSD 164234, a single-crystal refinement at 298 K
of one particular clinochlore - someone else's specimen, measured on someone
else's instrument, which is reason enough to span a composition axis around it.
What had gone wrong is subtler than that: the axis did not contain the
refinement at all.

chlorite_crystal *sets* the octahedral occupancies rather than adding to them, so
chlorite_crystal(0, 0) is a magnesium end member, not the published structure,
whose own iron is 0.0877 in the 2:1 sheet and 0.0580 in the hydroxide sheet.  The
build named that entry `chlorite`, with a comment saying the published structure
keeps its plain name, and the composition series held the 2:1 sheet at zero
throughout - so every entry was an invented composition and the one refined
against single-crystal data was not among them.
"""

import math
import pathlib
import re

import pytest

from clayquant.library import CHLORITE_IRON
from clayquant.models import CHLORITE_OCTAHEDRA, CHLORITE_PUBLISHED_IRON, chlorite_crystal

CIF = pathlib.Path(__file__).resolve().parents[1] / "structures" / "chlorite_ICSD_164234.cif"
requires_cif = pytest.mark.skipif(
    not CIF.exists(), reason="ICSD CIF files are licensed and not redistributed")


def test_the_published_composition_is_on_the_axis():
    assert CHLORITE_PUBLISHED_IRON in CHLORITE_IRON, (
        "the composition series does not contain the structure it was built "
        "around, so the refinement itself cannot be fitted"
    )


def test_it_is_the_entry_that_keeps_the_plain_name():
    """Whatever carries the bare name `chlorite` must be the refinement.

    A name is a claim about provenance, and this one was being made for a
    composition nobody measured.
    """
    plain = [pair for pair in CHLORITE_IRON if pair == CHLORITE_PUBLISHED_IRON]
    assert len(plain) == 1
    assert CHLORITE_IRON[0] == CHLORITE_PUBLISHED_IRON


def test_zero_iron_is_not_the_published_structure():
    """The premise of the whole fix, asserted so it cannot quietly stop being true."""
    assert CHLORITE_PUBLISHED_IRON != (0.0, 0.0)
    published = chlorite_crystal(*CHLORITE_PUBLISHED_IRON)
    magnesium = chlorite_crystal(0.0, 0.0)
    iron_of = lambda crystal: {s.label: s.occupancy for s in crystal.sites
                               if s.label.startswith("Fe")}
    assert any(value > 0.0 for value in iron_of(published).values())
    assert all(value == 0.0 for value in iron_of(magnesium).values())


@requires_cif
def test_the_constant_matches_the_file_it_describes():
    """Re-derive the two numbers from the CIF, weighting by Wyckoff multiplicity.

    Hard-coding them is a judgement call; letting them drift from the file is
    not, so this reads the occupancies back out.
    """
    text = CIF.read_text()
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 9 and parts[0][:2] in {"Mg", "Fe"} and parts[0][-1].isdigit():
            label, multiplicity, occupancy = parts[0], int(parts[2]), parts[-1]
            rows[label] = (multiplicity, float(re.sub(r"\(.*?\)", "", occupancy)))
    assert rows, "no octahedral sites parsed out of the CIF"
    for sheet, expected in zip(("2:1", "hydroxide"), CHLORITE_PUBLISHED_IRON):
        labels = [l for l in CHLORITE_OCTAHEDRA[sheet] if l.startswith("Fe")]
        weight = sum(rows[l][0] for l in labels)
        iron = sum(rows[l][0] * rows[l][1] for l in labels) / weight
        assert math.isclose(iron, expected, abs_tol=5e-4), (
            f"{sheet} sheet: the CIF says {iron:.4f}, the constant says {expected}"
        )
