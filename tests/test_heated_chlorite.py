"""The chlorite after 1.5 h at 550 C, which is not the same mineral.

The interlayer hydroxide sheet gives up its water, 2 OH(-) -> O(2-) + H2O.  A
chlorite's 001 reflects the contrast between its two sheets and its even orders
their sum, so emptying one sheet raises the 001 several times over while the
even orders collapse - which is exactly what the two heated standards do, and
why a heated mount cannot be fitted with air-dried reference patterns.
"""

import numpy as np
import pytest

from clayquant.mixed_layer import MixedLayerStack, lognormal_csds
from clayquant.models import (
    CHLORITE_HYDROXYL,
    chlorite_crystal,
    chlorite_layer,
    load_crystal,
)
from clayquant.optics import Divergence
from clayquant.pattern import Instrument, basal_pattern
from clayquant.profile import PeakShape

MEASURED_HEATED = {"Chlorite_16": 0.169, "Prochlorite_15": 0.749}
"""002/001 of the two heated standards, against 1.798 and 2.245 air dried."""


def occupancy(crystal, label):
    return next(site.occupancy for site in crystal.sites if site.label == label)


def test_no_dehydroxylation_is_the_structure_as_published():
    assert chlorite_crystal(0.0, 0.0).sites == chlorite_crystal(
        0.0, 0.0, dehydroxylation=0.0).sites


def test_half_the_oxygen_leaves_with_all_the_hydrogen():
    """The stoichiometry of 2 OH -> O + H2O, and nothing else."""
    dried = chlorite_crystal(0.0, 0.0, dehydroxylation=1.0)
    for label in CHLORITE_HYDROXYL["oxygen"]:
        assert occupancy(dried, label) == pytest.approx(0.5)
    for label in CHLORITE_HYDROXYL["hydrogen"]:
        assert occupancy(dried, label) == pytest.approx(0.0)
    half = chlorite_crystal(0.0, 0.0, dehydroxylation=0.5)
    assert occupancy(half, "O7") == pytest.approx(0.75)
    assert occupancy(half, "H2") == pytest.approx(0.5)


def test_the_layer_keeps_its_own_hydroxyl():
    """O6 and H1 belong to the 2:1 layer and survive this temperature."""
    dried = chlorite_crystal(0.0, 0.0, dehydroxylation=1.0)
    assert occupancy(dried, "O6") == pytest.approx(occupancy(load_crystal("chlorite"), "O6"))
    assert occupancy(dried, "H1") == pytest.approx(occupancy(load_crystal("chlorite"), "H1"))


def test_the_water_is_missing_from_the_mass():
    """A weight percent from a heated mount is on the dehydroxylated mass."""
    ratio = (chlorite_crystal(0.0, 0.0, dehydroxylation=1.0).cell_mass
             / chlorite_crystal(0.0, 0.0).cell_mass)
    # Six of the eight hydroxyls, so about a tenth of the mineral's weight.
    assert 0.88 < ratio < 0.93


def test_an_impossible_dehydroxylation_is_refused():
    for value in (-0.1, 1.5):
        with pytest.raises(ValueError):
            chlorite_crystal(0.0, 0.0, dehydroxylation=value)


def test_the_composition_is_recorded_in_the_name_and_the_source():
    crystal = chlorite_crystal(0.0, 0.05, dehydroxylation=0.75)
    assert "dehydrox 0.75" in crystal.name
    assert "H2O" in crystal.source
    assert "dehydrox" not in chlorite_crystal(0.0, 0.05).name


def basal_series(delta, iron=0.045):
    instrument = Instrument(
        peak_shape=PeakShape(u=0.004, v=-0.001, w=0.002, eta=0.5, size_ab=600.0),
        divergence=Divergence(specimen_length=35.0, goniometer_radius=240.0,
                              divergence=0.5),
    )
    layer = chlorite_layer(0.0, iron, dehydroxylation=delta)
    grid = np.arange(3.0, 40.0, 0.01)
    y = np.asarray(basal_pattern(
        MixedLayerStack(layer, layer, 1.0, csds=lognormal_csds(10.0)), grid,
        instrument).intensity)

    def area(centre, half=0.6):
        inside = np.abs(grid - centre) < half
        return float(np.trapezoid(y[inside], grid[inside]))

    first = area(6.22)
    return {1: first, 2: area(12.46) / first}


def test_the_001_grows_while_the_002_collapses():
    """The signature of heating a chlorite, and the whole reason to model it."""
    air = basal_series(0.0)
    heated = basal_series(0.75)
    assert heated[2] < 0.2 * air[2]
    assert heated[1] > 3.0 * air[1]


def test_the_standards_sit_inside_the_range():
    """Neither standard is fully dehydroxylated after 1.5 h, and both are
    reached: the parameter has to be spanned, not assumed."""
    air = basal_series(0.0)[2]
    full = basal_series(1.0)[2]
    for measured in MEASURED_HEATED.values():
        assert full < measured < air
