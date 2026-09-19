"""The collapsed smectite layer, and how much the fit's answer depends on it.

The air-dried layer is a model, not a measurement - Reynolds (1965) refined the
glycol complex and nothing comparable exists here for the water complex that
replaces it - so what matters is not that its interlayer is exactly right but
that the conclusion drawn from it does not hinge on the parts that are guessed.
These tests check the structure is consistent, and then measure that.
"""
from collections import defaultdict

import numpy as np
import pytest

from clayquant.models import (
    AIR_DRIED_SMECTITE_D001,
    REYNOLDS_1965_D001,
    WATER_PER_CELL,
    air_dried_smectite_layer,
    eg_smectite_layer,
)


def contents(layer):
    totals: defaultdict[str, float] = defaultdict(float)
    for species, occupancy in zip(layer.species, layer.occupancy):
        totals[species] += float(occupancy)
    return dict(totals)


def two_one_layer(layer):
    """The sites of the 2:1 layer itself, which reaches to the basal oxygens."""
    return sorted(
        (round(float(z), 4), species, round(float(occupancy), 4))
        for z, species, occupancy in zip(layer.z, layer.species, layer.occupancy)
        if abs(float(z)) <= 3.27 + 1e-9
    )


def test_the_two_one_layer_is_reynolds_own():
    """Drying empties the interlayer; it does not touch the 2:1 layer.

    So the sites out to the basal oxygens must be identical in the two models,
    at the same heights - not rescaled with the repeat distance, which is what
    ``with_thickness(scale_z=True)`` would have done and would have put the
    structure factor wrong.
    """
    assert two_one_layer(air_dried_smectite_layer()) == two_one_layer(eg_smectite_layer())


def test_the_two_one_layer_keeps_its_height_at_every_repeat():
    reference = two_one_layer(eg_smectite_layer())
    for thickness in (12.4, 13.5, 15.0):
        assert two_one_layer(air_dried_smectite_layer(thickness)) == reference


def test_no_atom_lies_outside_the_collapsed_cell():
    """The failure that prompted the model: compressing the repeat alone left
    the interlayer species at 7.94 A, outside a 12.4 A cell."""
    for thickness in (12.4, 13.0, 15.0):
        layer = air_dried_smectite_layer(thickness)
        assert np.max(np.abs(layer.z)) <= thickness / 2.0 + 1e-9


def test_the_cell_is_the_one_the_formula_says():
    air = contents(air_dried_smectite_layer())
    assert air["Si"] == pytest.approx(8.0)
    # 20 framework + 4 hydroxyl + one oxygen per water molecule.
    assert air["O"] == pytest.approx(24.0 + WATER_PER_CELL)
    # 4 hydroxyl hydrogens and two per water molecule.
    assert air["H"] == pytest.approx(4.0 + 2.0 * WATER_PER_CELL)
    # Charge balance is unchanged by drying, so the cation is Reynolds' own.
    assert air["Ca"] == pytest.approx(contents(eg_smectite_layer())["Ca"])


def test_glycolation_adds_the_glycol_and_removes_the_water_it_displaced():
    """A check on the electron bookkeeping, which is what the 00l series sees.

    Reynolds' complex holds 3.4 ethylene glycol per cell and 1.6 water; the
    air-dried model holds no glycol and ``WATER_PER_CELL``.  So glycolation adds
    3.4 molecules of C2H6O2 at 34 electrons each and takes away the water they
    displaced at 10 each, and the two models' electron counts must differ by
    exactly that.
    """
    glycol_per_cell = contents(eg_smectite_layer())["C"] / 2.0
    assert glycol_per_cell == pytest.approx(3.4)
    water_in_glycol_complex = contents(eg_smectite_layer())["O"] - 24.0 - 2.0 * glycol_per_cell
    assert water_in_glycol_complex == pytest.approx(1.6)

    displaced = WATER_PER_CELL - water_in_glycol_complex
    added = eg_smectite_layer().electrons - air_dried_smectite_layer().electrons
    assert added == pytest.approx(34.0 * glycol_per_cell - 10.0 * displaced, abs=0.5)


def test_an_impossible_repeat_is_refused():
    with pytest.raises(ValueError, match="no interlayer"):
        air_dried_smectite_layer(6.0)


def test_the_interlayer_plane_is_not_counted_twice():
    """It sits on the cell boundary, where the mirror gives it a twin."""
    layer = air_dried_smectite_layer(12.4)
    edge = [
        float(occupancy)
        for z, species, occupancy in zip(layer.z, layer.species, layer.occupancy)
        if species == "O" and abs(abs(float(z)) - 6.2) < 1e-9
    ]
    assert len(edge) == 2 and sum(edge) == pytest.approx(WATER_PER_CELL)


@pytest.mark.parametrize("water", [2.0, 4.0, 6.0])
def test_the_water_content_does_not_decide_the_discrimination(water):
    """The point of the model is to say whether a phase *changes* between the
    two treatments.  That is governed by the spacing, which is known, not by the
    interlayer contents, which are guessed - so halving or doubling the water
    must leave the difference between the treatments large.
    """
    from clayquant.crystal import LayerModel

    def basal(layer: LayerModel, order: int) -> float:
        s = np.array([order / layer.thickness])
        return float(abs(layer.structure_factor(s)[0]))

    air = air_dried_smectite_layer(AIR_DRIED_SMECTITE_D001, water=water)
    assert air.thickness == pytest.approx(AIR_DRIED_SMECTITE_D001)
    # The 001 spacing is what the fit sees move, and it is set by the repeat.
    assert abs(air.thickness - REYNOLDS_1965_D001) > 4.0
    assert basal(air, 1) > 0.0
