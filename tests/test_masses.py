"""Atomic weights, and the cell and layer masses built from them.

These are the bridge from a fitted scale factor to a mass, so an error here
would bias every weight percent by the same factor without looking wrong.  The
check is against formula masses that can be looked up independently.
"""

from __future__ import annotations

import pytest

from clayquant.masses import atomic_weight, element_of
from clayquant.models import available_phases, eg_smectite_layer, load_crystal, load_layer

requires_cifs = pytest.mark.skipif(
    not all(available_phases().values()), reason="ICSD CIF files are not installed"
)


def test_the_charge_is_ignored():
    assert element_of("Fe3+") == "Fe"
    assert element_of("O2-") == "O"
    assert element_of("K1+") == "K"
    assert atomic_weight("Si4+") == atomic_weight("Si")


@pytest.mark.parametrize(
    "formula, literature",
    [
        # (counts per formula unit, mass in g/mol from the IUPAC 2021 weights)
        ({"Si": 1, "O": 2}, 60.083),                                  # quartz
        ({"Ca": 1, "C": 1, "O": 3}, 100.086),                         # calcite
        ({"Al": 2, "O": 3}, 101.961),                                 # corundum
        ({"Al": 2, "Si": 2, "O": 9, "H": 4}, 258.157),                # kaolinite
        ({"K": 1, "Al": 3, "Si": 3, "O": 12, "H": 2}, 398.308),       # muscovite
        ({"C": 2, "H": 6, "O": 2}, 62.068),                           # ethylene glycol
    ],
)
def test_formula_masses(formula, literature):
    total = sum(count * atomic_weight(element) for element, count in formula.items())
    assert total == pytest.approx(literature, abs=0.01)


def test_an_unknown_element_is_reported_rather_than_guessed():
    """Whatever fails first - the scattering table or the mass table - it says so."""
    with pytest.raises(KeyError, match="Xx"):
        atomic_weight("Xx")


@requires_cifs
def test_the_cell_mass_is_the_formula_mass_times_the_formula_units():
    """Kaolinite 1M: C1 cell with two Al2Si2O5(OH)4 units."""
    crystal = load_crystal("kaolinite_1M")
    formula = 258.157
    assert crystal.cell_mass == pytest.approx(2 * formula, rel=0.002)


@requires_cifs
def test_a_structure_without_hydrogens_is_lighter_by_exactly_them():
    """Gruner's 1932 kaolinite 2M has no hydrogen positions, and it shows.

    Per formula unit: the 1M cell holds two, the 2M cell four.
    """
    with_h = load_crystal("kaolinite_1M").cell_mass / 2
    without_h = load_crystal("kaolinite_2M").cell_mass / 4
    assert with_h - without_h == pytest.approx(4 * atomic_weight("H"), abs=0.2)


@requires_cifs
def test_the_layer_mass_is_the_cell_mass_divided_by_the_layers_in_it():
    from clayquant.models import CIF_SOURCES

    for key, source in CIF_SOURCES.items():
        crystal = load_crystal(key)
        layer = load_layer(key)
        assert layer.mass == pytest.approx(
            crystal.cell_mass / source.layers_per_cell, rel=1e-6
        ), key


def test_the_glycol_smectite_layer_masses_its_own_contents():
    """2 x {(Al,Fe,Mg)2 Si4 O10(OH)2 . 1.7 C2H6O2 . 0.8 H2O . 0.2 Ca}."""
    layer = eg_smectite_layer()
    pyrophyllite_less_octahedra = (
        4 * atomic_weight("Si") + 10 * atomic_weight("O") + 2 * atomic_weight("O")
        + 2 * atomic_weight("H")
    )
    octahedra = (3.08 * atomic_weight("Al") + 0.32 * atomic_weight("Fe")
                 + 0.66 * atomic_weight("Mg"))
    glycol = 1.7 * (2 * atomic_weight("C") + 6 * atomic_weight("H") + 2 * atomic_weight("O"))
    water = 0.8 * (atomic_weight("O") + 2 * atomic_weight("H"))
    calcium = 0.2 * atomic_weight("Ca")
    expected = 2 * (pyrophyllite_less_octahedra + glycol + water + calcium) + octahedra
    assert layer.mass == pytest.approx(expected, rel=0.002)


@requires_cifs
def test_every_library_phase_has_a_mass():
    from clayquant.models import CIF_SOURCES

    for key in CIF_SOURCES:
        assert load_crystal(key).cell_mass > 0
        assert load_layer(key).mass > 0
    assert eg_smectite_layer().mass > 0
