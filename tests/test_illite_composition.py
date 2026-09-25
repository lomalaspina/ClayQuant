"""The illite's composition, which the published structure gets wrong twice.

ICSD 90144 carries K at full occupancy - a muscovite - and no octahedral iron,
and it calculates a basal series three times too strong at 5 A.  Both
substitutions move that series, and they move it differently: the K sits at
z = 1/4 of the two-layer cell, exactly between the layers, so its phase factor
alternates from order to order, while the iron sits at the middle of the 2:1
layer, the plane the layer is nearly symmetric about.
"""

import numpy as np
import pytest

from clayquant.emission import CU_KA_5LINE
from clayquant.library import ILLITE_COMPOSITION
from clayquant.models import illite_crystal, load_crystal
from clayquant.optics import Divergence
from clayquant.pattern import Instrument, powder_pattern, two_theta_grid
from clayquant.profile import PeakShape

MEASURED = {2: 0.164, 3: 0.477}
"""The Illite_10 standard's basal series, as order/001 areas, air dried."""


def occupancies(crystal, element):
    return [site.occupancy for site in crystal.sites if site.species.startswith(element)]


def test_the_published_structure_is_an_iron_free_muscovite():
    published = load_crystal("illite")
    assert occupancies(published, "K") == [1.0]
    assert occupancies(published, "Fe") == []
    assert illite_crystal().sites == published.sites


def test_the_occupancy_is_set_and_the_geometry_is_not():
    published = load_crystal("illite")
    changed = illite_crystal(0.8)
    assert occupancies(changed, "K") == [0.8]
    assert changed.a == published.a and changed.c == published.c
    assert len(changed.sites) == len(published.sites)
    for before, after in zip(published.sites, changed.sites):
        assert before.label == after.label
        assert (before.x, before.y, before.z) == (after.x, after.y, after.z)


def test_the_iron_replaces_the_octahedral_aluminium_and_nothing_else():
    published = load_crystal("illite")
    changed = illite_crystal(1.0, 0.15)
    octahedral = [site for site in published.sites
                  if site.species.startswith("Al") and min(site.z % 0.5, 0.5 - site.z % 0.5) < 0.05]
    assert len(octahedral) == 1
    iron = [site for site in changed.sites if site.species.startswith("Fe")]
    assert len(iron) == 1
    assert iron[0].z == octahedral[0].z
    assert iron[0].occupancy == pytest.approx(0.15 * octahedral[0].occupancy)
    # The tetrahedral aluminium, at z = 0.137, is left alone.
    tetrahedral = [site for site in changed.sites
                   if site.label.startswith("Al1")]
    assert [site.occupancy for site in tetrahedral] == [1.0]


def test_the_two_substitutions_pull_the_mass_opposite_ways():
    published = load_crystal("illite").cell_mass
    assert illite_crystal(0.8).cell_mass < published
    assert illite_crystal(1.0, 0.15).cell_mass > published


def test_an_impossible_composition_is_refused():
    for value in (-0.1, 1.5):
        with pytest.raises(ValueError):
            illite_crystal(value)
        with pytest.raises(ValueError):
            illite_crystal(1.0, value)


def basal_ratios(crystal):
    grid = two_theta_grid(3.0, 40.0, 0.0167)
    instrument = Instrument(
        emission=CU_KA_5LINE,
        peak_shape=PeakShape(u=0.0, v=0.0, w=0.0025, eta=0.6, size_ab=400.0),
        lp_mode="powder",
        divergence=Divergence(specimen_length=35.0, goniometer_radius=240.0,
                              divergence=0.5),
    )
    pattern = powder_pattern(crystal, grid, instrument, r_march_dollase=0.3,
                             name="illite")
    x = np.asarray(pattern.two_theta)
    y = np.asarray(pattern.intensity)

    def area(centre, half=0.4):
        inside = (x >= centre - half) & (x <= centre + half)
        return float(np.trapezoid(np.clip(y[inside], 0.0, None), x[inside]))

    first = area(8.84)
    return {2: area(17.72) / first, 3: area(26.72) / first}


def test_removing_potassium_lowers_the_higher_orders():
    published = basal_ratios(load_crystal("illite"))
    poorer = basal_ratios(illite_crystal(0.8))
    assert poorer[2] < published[2]
    assert poorer[3] < published[3]


def test_the_iron_lowers_the_5_angstrom_order_hardest():
    """The reason the axis is a pair and not the potassium alone.

    Removing a fifth of the potassium and adding 0.15 of iron both lower the
    5 A order, but only the iron lowers it enough to reach the measurement, and
    it does so without flattening the 3.33 A order as far.
    """
    published = basal_ratios(load_crystal("illite"))
    by_potassium = basal_ratios(illite_crystal(0.8))
    by_iron = basal_ratios(illite_crystal(1.0, 0.15))
    assert by_iron[2] < by_potassium[2] < published[2]
    assert (published[2] - by_iron[2]) > 3.0 * (published[2] - by_potassium[2])


def test_the_iron_bearing_composition_reaches_the_measured_series():
    ratios = basal_ratios(illite_crystal(1.0, 0.15))
    published = basal_ratios(load_crystal("illite"))
    for order in (2, 3):
        assert abs(ratios[order] - MEASURED[order]) < abs(published[order] - MEASURED[order])


def test_the_axis_spans_the_published_structure_and_both_substitutions():
    assert (1.0, 0.0) in ILLITE_COMPOSITION
    assert any(k < 1.0 for k, _ in ILLITE_COMPOSITION)
    assert any(fe > 0.0 for _, fe in ILLITE_COMPOSITION)
    for potassium, iron in ILLITE_COMPOSITION:
        assert 0.0 <= potassium <= 1.0 and 0.0 <= iron <= 1.0
        illite_crystal(potassium, iron)


def test_the_interstratified_host_is_built_from_the_corrected_illite():
    """The host of the I/S series must not be a muscovite while the discrete
    illite beside it is not: the split between the two would absorb the
    difference."""
    from clayquant.library import ILLITE_SMECTITE_HOST, build_library
    from clayquant.pattern import two_theta_grid

    def host_series(pair):
        library = build_library(
            grid=two_theta_grid(4.0, 30.0, 0.1), orientations=(0.3,),
            illite_smectite=(1.0,), chlorite_smectite=(), chlorite_iron=(),
            illite_composition=(), csds_means=(15.0,), host_thicknesses={},
            air_dried_thickness=None, illite_smectite_host=pair, progress=False)
        entry = next(e for e in library.entries if "I/S" in e.name)
        x = np.asarray(library.two_theta)
        y = np.asarray(entry.intensity)

        def area(centre, half=0.5):
            inside = (x >= centre - half) & (x <= centre + half)
            return float(np.trapezoid(np.clip(y[inside], 0.0, None), x[inside]))

        return area(17.72) / area(8.84)

    assert host_series(ILLITE_SMECTITE_HOST) < host_series(None)


def test_the_host_composition_is_recorded():
    from clayquant.library import ILLITE_SMECTITE_HOST, build_library
    from clayquant.pattern import two_theta_grid

    library = build_library(
        grid=two_theta_grid(4.0, 20.0, 0.2), orientations=(0.3,),
        illite_smectite=(0.5,), chlorite_smectite=(), chlorite_iron=(),
        illite_composition=(), csds_means=(15.0,), progress=False)
    assert library.metadata["illite_smectite_host"] == list(ILLITE_SMECTITE_HOST)
