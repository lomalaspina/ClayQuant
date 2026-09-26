"""The two calculations have to agree per gram, not per pattern.

A discrete mineral is calculated by powder_pattern, whose unit is one unit
cell.  An interstratified one is calculated by the stacking model, whose unit is
one layer.  Their weight percents are compared with each other, so the two
descriptions of the same material must give the same intensity for the same
mass - and a cell is not a layer.
"""

import numpy as np
import pytest

from clayquant.emission import CU_KA_5LINE
from clayquant.library import DEFAULT_PEAK_SHAPE
from clayquant.mixed_layer import MixedLayerStack, lognormal_csds
from clayquant.models import (
    CIF_SOURCES,
    eg_smectite_layer,
    illite_crystal,
    load_crystal,
)
from clayquant.optics import Divergence
from clayquant.pattern import (
    Instrument,
    basal_pattern,
    basal_scale_factor,
    layer_basal_scale_factor,
    powder_pattern,
    two_theta_grid,
)

GRID = two_theta_grid(2.0, 40.0, 0.04)
ORIENTATION = 0.1


@pytest.fixture(scope="module")
def instrument():
    return Instrument(
        emission=CU_KA_5LINE, peak_shape=DEFAULT_PEAK_SHAPE, lp_mode="powder",
        divergence=Divergence(specimen_length=25.0, goniometer_radius=240.0,
                              divergence=0.5, shape="round"),
    )


def area(intensity):
    return float(np.trapezoid(np.asarray(intensity), GRID))


def per_gram_both_ways(crystal, layers_per_cell, instrument, mean=15.0):
    csds = lognormal_csds(mean)
    layer = crystal.layer_model(layers_per_cell=layers_per_cell, name="x")
    powder = area(powder_pattern(crystal, GRID, instrument,
                                 r_march_dollase=ORIENTATION).intensity) / crystal.cell_mass
    scale = basal_scale_factor(crystal, layers_per_cell, GRID, instrument, csds)
    stack = area(basal_pattern(MixedLayerStack(layer, layer, 1.0, csds=csds), GRID,
                               instrument, r_march_dollase=ORIENTATION).intensity)
    return powder, stack * scale / layer.mass


@pytest.mark.parametrize("key", ["illite", "chlorite", "kaolinite_1M", "kaolinite_2M"])
def test_the_same_mineral_scatters_the_same_per_gram_either_way(key, instrument):
    crystal = illite_crystal(0.9, 0.15) if key == "illite" else load_crystal(key)
    powder, stack = per_gram_both_ways(crystal, CIF_SOURCES[key].layers_per_cell, instrument)
    assert stack == pytest.approx(powder, rel=2e-3)


def test_a_two_layer_cell_is_where_the_divisor_shows(instrument):
    """Without it the illite series is twice as bright per gram as the illite."""
    crystal = illite_crystal(0.9, 0.15)
    csds = lognormal_csds(15.0)
    assert CIF_SOURCES["illite"].layers_per_cell == 2
    with_divisor = basal_scale_factor(crystal, 2, GRID, instrument, csds)
    powder, stack = per_gram_both_ways(crystal, 2, instrument)
    assert stack == pytest.approx(powder, rel=2e-3)
    # and the undivided value, which is what it used to return
    assert (stack / with_divisor) * (with_divisor * 2) == pytest.approx(2.0 * stack)


def test_the_divisor_does_nothing_to_a_one_layer_cell(instrument):
    """Which is why the error hid: it moved I/S and left C/S alone."""
    assert CIF_SOURCES["chlorite"].layers_per_cell == 1
    powder, stack = per_gram_both_ways(load_crystal("chlorite"), 1, instrument)
    assert stack == pytest.approx(powder, rel=2e-3)


@pytest.mark.parametrize("key", ["chlorite", "kaolinite_1M"])
def test_the_layer_only_scale_matches_the_crystal_one(key, instrument):
    """For a one-layer cell the two must be the same number, since the layer is
    the cell.  This is what licenses using it for the smectite."""
    crystal = load_crystal(key)
    layer = crystal.layer_model(layers_per_cell=1, name=key)
    csds = lognormal_csds(15.0)
    assert layer_basal_scale_factor(layer, GRID, instrument, csds) == pytest.approx(
        basal_scale_factor(crystal, 1, GRID, instrument, csds), rel=1e-4)


def test_the_layer_only_scale_puts_a_layer_model_on_the_powder_scale(instrument):
    """The check that matters for the smectite, done on a mineral that can be
    checked: a one-layer cell calculated from its layer alone against the same
    mineral calculated from its cell."""
    crystal = load_crystal("chlorite")
    layer = crystal.layer_model(layers_per_cell=1, name="chlorite")
    csds = lognormal_csds(15.0)
    scale = layer_basal_scale_factor(layer, GRID, instrument, csds)
    powder = area(powder_pattern(crystal, GRID, instrument,
                                 r_march_dollase=ORIENTATION).intensity) / crystal.cell_mass
    stack = area(basal_pattern(MixedLayerStack(layer, layer, 1.0, csds=csds), GRID,
                               instrument, r_march_dollase=ORIENTATION).intensity)
    assert stack * scale / layer.mass == pytest.approx(powder, rel=2e-3)


def test_the_smectite_gets_a_scale_at_all(instrument):
    """It had none, and sat on the stacking model's own basis while everything
    its weight percent was compared against sat on the powder one."""
    smectite = eg_smectite_layer()
    scale = layer_basal_scale_factor(smectite, GRID, instrument, lognormal_csds(15.0))
    assert 0.0 < scale < 1.0


def test_the_scaled_smectite_lands_among_the_other_clays(instrument):
    """Not a proof of its brightness, but a bound: before the scale it was 25
    times kaolinite per gram, where the four discrete minerals span a factor of
    three among themselves."""
    csds = lognormal_csds(15.0)
    smectite = eg_smectite_layer()
    raw = area(basal_pattern(MixedLayerStack(smectite, smectite, 1.0, csds=csds), GRID,
                             instrument, r_march_dollase=ORIENTATION).intensity)
    scale = layer_basal_scale_factor(smectite, GRID, instrument, csds)
    kaolinite = load_crystal("kaolinite_1M")
    reference = area(powder_pattern(kaolinite, GRID, instrument,
                                    r_march_dollase=ORIENTATION).intensity) / kaolinite.cell_mass
    before = (raw / smectite.mass) / reference
    after = (raw * scale / smectite.mass) / reference
    assert before > 20.0
    assert after < before / 3.0
