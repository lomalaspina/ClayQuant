"""Every element, and the species spellings real structure files carry."""
from __future__ import annotations

import pytest

from clayquant.absorption import MASS_ATTENUATION_CU_KA, mass_attenuation
from clayquant.masses import atomic_weight, element_of


def test_the_table_covers_hydrogen_to_uranium():
    """No structure should stop a fit for want of a coefficient."""
    assert len(MASS_ATTENUATION_CU_KA) >= 92
    for element in ("H", "Li", "B", "Si", "Fe", "Sn", "La", "W", "Pb", "U"):
        assert MASS_ATTENUATION_CU_KA[element] > 0.0


def test_the_two_absorption_edges_that_cross_cu_k_alpha_survive():
    """The series must not be monotonic in Z, and must never be interpolated.

    The K edge falls between cobalt and nickel, the L3 edge among the heavy
    elements.  A table built by interpolating between neighbours would erase
    both, and be wrong by a factor of six at nickel.
    """
    m = MASS_ATTENUATION_CU_KA
    assert m["Ni"] < 0.2 * m["Co"]
    assert m["W"] < m["La"]
    assert m["Pb"] > m["W"]


def test_one_source_throughout():
    """Values sit within about 12 % of International Tables Volume C, which is
    the spread between published tabulations - close enough to be the same
    quantity, far enough apart that mixing the two would put a step in the
    middle of the table."""
    assert 280.0 < MASS_ATTENUATION_CU_KA["Fe"] < 320.0
    assert 45.0 < MASS_ATTENUATION_CU_KA["Al"] < 52.0
    assert 320.0 < MASS_ATTENUATION_CU_KA["Ba"] < 360.0


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("Fe", "Fe"), ("Fe3+", "Fe"), ("Fe+3", "Fe"), ("O2-", "O"), ("O-2", "O"),
        ("Sn+4", "Sn"), ("K+", "K"),
        ("Ti(4+)", "Ti"),   # a parenthesised charge
        ("Fe2", "Fe"),      # a site label left in the species field
        ("O3", "O"),
        ("D", "H"),         # deuterium scatters as hydrogen
        ("FE", "Fe"),       # the wrong case
        ("si", "Si"),
    ],
)
def test_the_spellings_real_files_carry_all_resolve(spelling, expected):
    assert element_of(spelling) == expected
    assert mass_attenuation(spelling) == MASS_ATTENUATION_CU_KA[expected]


@pytest.mark.parametrize("spelling", ["OH", "Zz9", "Xx", ""])
def test_what_cannot_be_read_is_still_refused(spelling):
    """The relaxations must not become guesses.  "OH" is two elements and
    choosing one would put a wrong mass into a weight percent."""
    with pytest.raises(KeyError):
        element_of(spelling)


def test_deuterium_keeps_hydrogen_s_weight_not_its_own():
    """It resolves to hydrogen for *scattering*; the mass is a separate question
    and this records which answer is being given, so that a neutron-refined
    structure's masses are read knowing it."""
    assert atomic_weight("D") == pytest.approx(atomic_weight("H"))


def test_the_two_sources_are_kept_apart_and_do_not_overlap():
    """Which tabulation a coefficient came from should be answerable by looking.

    The two differ by up to 11 % on individual elements, so merging them by hand
    would bury a seam that matters: a value from the Elam set carries the
    accuracy of that source, not of the compound check the International Tables
    subset passes.
    """
    from clayquant.absorption import _CHECKED_CU_KA, _ELAM_CU_KA

    assert not set(_CHECKED_CU_KA) & set(_ELAM_CU_KA)
    assert len(_CHECKED_CU_KA) + len(_ELAM_CU_KA) == len(MASS_ATTENUATION_CU_KA)
    # the checked subset is the rock-forming one, and it wins where both could
    for element in ("Si", "Al", "Fe", "K", "Mg", "Ca", "O"):
        assert MASS_ATTENUATION_CU_KA[element] == _CHECKED_CU_KA[element]


def test_the_checked_values_were_not_disturbed_by_filling_the_gaps():
    """Adding 65 elements must not move the ones the compound check rests on."""
    from clayquant.absorption import _CHECKED_CU_KA

    assert _CHECKED_CU_KA["Fe"] == pytest.approx(308.0)
    assert _CHECKED_CU_KA["Si"] == pytest.approx(60.6)
    assert _CHECKED_CU_KA["Al"] == pytest.approx(48.6)
