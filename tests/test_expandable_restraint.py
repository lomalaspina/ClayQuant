"""The air-dried ceiling as a restraint, not as a deletion.

`expandable_bound` measures how much expandable layer content the two mounts
allow, and it was applied by removing every entry above that from the library.
A ceiling on *composition* enforced as a deletion of *shape* is not the same
statement: a mixed-layer entry is both a composition the ceiling may exclude and
a profile it has no business excluding, and in this library the broad mixed-layer
entries are the only ones wide enough to carry a real clay's low-angle intensity
(Sec. A.73).
"""
import numpy as np
import pytest

from clayquant.library import PatternLibrary
from clayquant.pattern import Pattern
from clayquant.treatment import (EXPANDABLE_CONSTRAINT_WEIGHT, ExpandableBound,
                                 expandable_mass_constraint)

GRID = np.arange(4.0, 20.0, 0.05)


def a_peak(centre, width=0.3):
    return np.exp(-0.5 * ((GRID - centre) / width) ** 2)


@pytest.fixture
def library():
    """Two discrete entries and three carrying expandable layers."""
    lib = PatternLibrary(two_theta=GRID)
    lib.add(Pattern(two_theta=GRID, intensity=a_peak(8.8), name="illite"),
            phase="illite", fraction=1.0, strip_continuum=False)
    lib.add(Pattern(two_theta=GRID, intensity=a_peak(12.4), name="kaolinite"),
            phase="kaolinite_1M", strip_continuum=False)
    for name, fraction in (("I/S 0.98", 0.98), ("I/S 0.80", 0.80),
                           ("I/S 0.40", 0.40)):
        lib.add(Pattern(two_theta=GRID, intensity=a_peak(8.0, 0.9), name=name),
                phase="I/S", fraction=fraction, strip_continuum=False)
    return lib


def a_bound(maximum, library):
    allowed, excluded = [], []
    for index, entry in enumerate(library.entries):
        expandable = 0.0 if entry.fraction is None else 1.0 - float(entry.fraction)
        (allowed if expandable <= maximum + 1e-9 else excluded).append(index)
    return ExpandableBound(maximum=maximum, allowed=allowed, excluded=excluded,
                           unrestricted=False, status="")


def test_nothing_is_removed_from_the_library(library):
    """The whole point: every shape stays reachable."""
    bound = a_bound(0.03, library)
    block = expandable_mass_constraint(library, bound)
    assert block is not None
    assert block.n_entries == len(library.entries)


def test_only_the_entries_above_the_ceiling_are_penalised(library):
    """The restraint acts on what the measurement excludes, not on expandable
    clay as such: an entry inside the ceiling carries no penalty."""
    bound = a_bound(0.03, library)   # excludes 0.80 and 0.40, keeps 0.98 (2 %)
    row = expandable_mass_constraint(library, bound).design[0]
    names = library.names
    assert row[names.index("illite")] == 0.0
    assert row[names.index("kaolinite")] == 0.0
    assert row[names.index("I/S 0.98")] == 0.0, "inside the ceiling, so not penalised"
    assert row[names.index("I/S 0.80")] > 0.0
    assert row[names.index("I/S 0.40")] > 0.0


def test_the_penalty_grows_with_the_expandable_content(library):
    bound = a_bound(0.03, library)
    row = expandable_mass_constraint(library, bound).design[0]
    names = library.names
    assert row[names.index("I/S 0.40")] > row[names.index("I/S 0.80")]
    # 0.60 expandable against 0.20, on patterns of equal area
    assert row[names.index("I/S 0.40")] == pytest.approx(
        3.0 * row[names.index("I/S 0.80")], rel=1e-6)


def test_it_pulls_towards_zero_and_not_towards_the_ceiling(library):
    """The mounts say *at most* this much, which is one-sided.

    A row targeting the ceiling would pull the expandable content up to it even
    where the data want none.
    """
    block = expandable_mass_constraint(library, a_bound(0.03, library))
    assert block.target.shape == (1,)
    assert float(block.target[0]) == 0.0


def test_no_restraint_where_there_is_nothing_to_restrain(library):
    """Movement seen, or a ceiling that excludes nothing: the bound says
    nothing, so neither does this."""
    assert expandable_mass_constraint(
        library, ExpandableBound(1.0, list(range(5)), [], True, "")) is None
    assert expandable_mass_constraint(library, a_bound(1.0, library)) is None


def test_the_data_can_still_outvote_it(library):
    """A fit that needs an excluded entry's *width* can reach it by paying.

    Deletion made that impossible, which is what cost 2651 entries and left one
    separate explaining 7 per cent of its low-angle region.
    """
    from clayquant.nnls import nnls_fit
    # a broad feature at 8.0 deg that only the mixed-layer entries can carry
    measured = Pattern(two_theta=GRID, name="m",
                       intensity=500.0 + 9000.0 * a_peak(8.0, 0.9))
    bound = a_bound(0.03, library)
    block = expandable_mass_constraint(library, bound,
                                       weight=EXPANDABLE_CONSTRAINT_WEIGHT)
    restrained = nnls_fit(measured, library, range_two_theta=(4.0, 20.0),
                          constraints=[block])
    carried = sum(float(c) for name, c in
                  zip(restrained.names, restrained.coefficients)
                  if name.startswith("I/S"))
    assert carried > 0.0, "the restraint has become a deletion"
    assert restrained.r_wp < 0.5


def test_the_weight_is_a_restraint_not_a_deletion():
    assert 0.0 < EXPANDABLE_CONSTRAINT_WEIGHT < 1e3
