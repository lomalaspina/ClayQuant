"""Mass attenuation coefficients, and the external-standard scale they serve.

A weight percent from a fitted scale factor divides one mount between its
phases and says nothing about the absolute amount.  A standard measured through
the same optics fixes that, through W = S(ZMV)mu/K, and the absorption term is
there because the relation is for a specimen thick enough that its own
absorption decides how much material diffracts.  Two things therefore have to be
right: the coefficients, and the thickness assumption - which is why
penetration_depth is tested as carefully as the table.

The table is checked against compounds whose coefficients are published, since a
per-element table transcribed by hand is exactly the kind of thing that is
wrong in one entry and plausible everywhere.
"""

from __future__ import annotations

import math

import pytest

from clayquant.absorption import (
    MASS_ATTENUATION_CU_KA,
    mass_attenuation,
    mass_attenuation_of,
    penetration_depth,
    thick_enough,
)


def test_a_species_is_read_the_way_a_cif_writes_it():
    assert mass_attenuation("Fe") == mass_attenuation("Fe3+") == mass_attenuation("Fe2+")
    assert mass_attenuation("O-2") == MASS_ATTENUATION_CU_KA["O"]
    assert mass_attenuation("Si4+") == MASS_ATTENUATION_CU_KA["Si"]


def test_an_element_that_is_not_tabulated_raises_rather_than_guessing():
    """A wrong coefficient propagates silently into a weight percent."""
    with pytest.raises(KeyError, match="not guessed at"):
        mass_attenuation("Au")


def test_iron_absorbs_far_more_than_the_magnesium_it_replaces():
    """The reason an Fe-rich clay is not interchangeable with an Fe-poor one.

    Iron's K edge lies just below the Cu K-alpha energy and nickel's just above,
    so the two sit on opposite sides of a discontinuity an order of magnitude
    tall.  A table that had smoothed over it would fail here.
    """
    assert mass_attenuation("Fe") > 7 * mass_attenuation("Mg")
    assert mass_attenuation("Fe") > 5 * mass_attenuation("Ni")
    assert mass_attenuation("Cu") < mass_attenuation("Ca")


class FakeSite:
    def __init__(self, species, occupancy=1.0):
        self.species = species
        self.occupancy = occupancy


class FakeCrystal:
    def __init__(self, sites, name="test"):
        self._sites = sites
        self.name = name

    def expanded_sites(self):
        return list(self._sites)


@pytest.mark.parametrize(
    "formula, published",
    [
        # Corundum, quartz and calcite have well-known coefficients at Cu K-alpha.
        ([("Al", 2), ("O", 3)], 31.1),
        ([("Si", 1), ("O", 2)], 34.4),
        ([("Ca", 1), ("C", 1), ("O", 3)], 74.8),
    ],
)
def test_a_compound_agrees_with_its_published_coefficient(formula, published):
    sites = [FakeSite(element) for element, count in formula for _ in range(count)]
    assert mass_attenuation_of(FakeCrystal(sites)) == pytest.approx(published, rel=0.01)


def test_a_structure_with_no_mass_is_refused():
    with pytest.raises(ValueError, match="no mass"):
        mass_attenuation_of(FakeCrystal([FakeSite("O", occupancy=0.0)]))


def test_the_penetration_depth_shrinks_towards_low_angle():
    """The counter-intuitive part, and the one that matters for clays.

    A basal reflection at 6 deg comes from a few micrometres of the mount, while
    a 35 deg reflection samples five times deeper - so the reflections a clay
    analysis lives on are the ones most exposed to a film being too thin.
    """
    shallow = penetration_depth(6.2, 30.0, 2.0)
    deep = penetration_depth(35.1, 30.0, 2.0)
    assert shallow < deep
    assert deep / shallow == pytest.approx(
        math.sin(math.radians(35.1 / 2)) / math.sin(math.radians(6.2 / 2)), rel=1e-9
    )
    assert 3.0 < shallow < 6.0, "a few micrometres, for a clay at Cu K-alpha"


def test_a_more_absorbing_specimen_is_sampled_less_deeply():
    assert penetration_depth(20.0, 60.0, 2.0) == pytest.approx(
        penetration_depth(20.0, 30.0, 2.0) / 2.0
    )
    assert penetration_depth(20.0, 30.0, 4.0) == pytest.approx(
        penetration_depth(20.0, 30.0, 2.0) / 2.0
    )


def test_a_zero_attenuation_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        penetration_depth(20.0, 0.0, 2.0)


def test_thick_enough_is_angle_dependent():
    """One film is effectively infinite at low angle and not at high.

    Which is what makes a thin film distinguishable from a texture factor: a
    thickness deficit varies across the pattern, a texture factor does not.
    """
    assert thick_enough(60.0, 8.8, 30.0, 2.0)
    assert not thick_enough(60.0, 35.1, 30.0, 2.0)
    assert not thick_enough(5.0, 8.8, 30.0, 2.0)
    # And a pressed powder is infinite everywhere in range.
    assert all(thick_enough(1000.0, angle, 31.1, 2.0) for angle in (4.0, 20.0, 39.0))
