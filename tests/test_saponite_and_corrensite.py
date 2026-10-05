"""The trioctahedral smectite and the chlorite/smectite phase built from it.

Neither has been checked against a measurement - the saponite standard in this
collection shows one broad 001 and no usable higher order, and there is no
corrensite in it at all - so what can be checked is that each is the mineral it
claims to be: the right layer content, the right spacings, and a density that
falls where it must.
"""
import math

import numpy as np
import pytest

from clayquant.library import CORRENSITE_FRACTION, ordered_transition
from clayquant.mixed_layer import MixedLayerStack, lognormal_csds
from clayquant.models import (
    AIR_DRIED_SMECTITE_D001,
    REYNOLDS_1965_D001,
    air_dried_saponite_layer,
    air_dried_smectite_layer,
    eg_smectite_layer,
    load_layer,
    saponite_layer,
    trioctahedral_two_one_rows,
)
from clayquant.optics import Divergence
from clayquant.pattern import Instrument, basal_pattern
from clayquant.profile import PeakShape

WAVELENGTH = 1.5405980


def _content():
    """Full 2:1 layer content, the mirror plane counted once."""
    totals: dict[str, float] = {}
    for z, species, occupancy, _ in trioctahedral_two_one_rows():
        totals[species] = totals.get(species, 0.0) + occupancy * (1 if z == 0 else 2)
    return totals


def test_the_two_one_sheet_is_trioctahedral():
    """Six octahedral cations, which is the whole point of taking it."""
    content = _content()
    octahedral = content["Mg2+"] + content["Fe3+"]
    assert octahedral == pytest.approx(6.0, abs=1e-6)


def test_the_two_one_sheet_is_a_two_one_layer():
    """Eight tetrahedral cations and O20(OH)4, or it is not one."""
    content = _content()
    assert content["Si4+"] + content["Al3+"] == pytest.approx(8.0, abs=1e-6)
    assert content["O2-"] == pytest.approx(24.0, abs=1e-6)
    assert content["H1+"] == pytest.approx(4.0, abs=1e-6)


def test_the_octahedral_plane_is_not_lost_to_rounding():
    """The bug this guards against made the layer dioctahedral by accident.

    The sheet's octahedral sites refine to z = 0.99987, which folds to a height
    of about -0.002 A.  Discarding that as negative kept four cations of the six
    and the layer silently became the wrong mineral.
    """
    rows = trioctahedral_two_one_rows()
    on_the_mirror = sum(occ for z, _, occ, _ in rows if z == 0.0)
    assert on_the_mirror == pytest.approx(6.0, abs=1e-6)


def test_the_sheet_agrees_with_reynolds_about_where_the_planes_are():
    """Two refinements of the same object, to within a twentieth of an angstrom."""
    ours = sorted({z for z, *_ in trioctahedral_two_one_rows()})
    reynolds = [0.00, 1.06, 2.70, 3.27]
    for theirs in reynolds:
        assert min(abs(z - theirs) for z in ours) < 0.07, theirs


def test_saponite_glycolates_and_collapses():
    assert saponite_layer().thickness == pytest.approx(REYNOLDS_1965_D001)
    assert air_dried_saponite_layer().thickness == pytest.approx(AIR_DRIED_SMECTITE_D001)


def test_a_collapse_that_leaves_no_interlayer_is_refused():
    with pytest.raises(ValueError, match="no interlayer"):
        air_dried_saponite_layer(6.0)


def _instrument():
    return Instrument(
        peak_shape=PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6, size_ab=400.0),
        divergence=Divergence(specimen_length=25.0, goniometer_radius=240.0,
                              divergence=0.5, shape="round", beam_width=14.0),
    )


def _orders(layer_a, layer_b, fraction, ordering, grid):
    stack = MixedLayerStack(layer_a, layer_b, fraction_a=fraction,
                            transition=ordered_transition(fraction, ordering),
                            csds=lognormal_csds(15.0, 0.35))
    return np.asarray(basal_pattern(stack, grid, _instrument(),
                                    r_march_dollase=1.0).intensity)


def _maxima(grid, y, floor=0.08):
    peak = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    keep = np.where(peak & (y[1:-1] >= floor * y.max()))[0] + 1
    return [WAVELENGTH / (2.0 * math.sin(math.radians(float(grid[k]) / 2.0))) for k in keep]


def test_the_two_smectites_are_near_duplicates_in_a_basal_scan():
    """The fact the build switch exists for, so it is pinned rather than assumed.

    What tells a trioctahedral smectite from a dioctahedral one is the 060 near
    1.53 A against 1.50 A, at about 60 deg 2theta on an unoriented mount.  An
    oriented basal scan never reaches it, so the two are all but the same column
    here and must not be offered to the fit as two.
    """
    grid = np.arange(2.0, 40.0, 0.02)
    for a, b in ((saponite_layer(), eg_smectite_layer()),
                 (air_dried_saponite_layer(), air_dried_smectite_layer())):
        tri = _orders(a, a, 1.0, 0.0, grid)
        di = _orders(b, b, 1.0, 0.0, grid)
        cosine = float(tri @ di / np.linalg.norm(tri) / np.linalg.norm(di))
        assert cosine > 0.99, cosine


def test_corrensite_is_a_rational_series_on_the_sum_of_its_layers():
    """Which is what makes it corrensite and not a chlorite/smectite mixture."""
    grid = np.arange(2.0, 36.0, 0.02)
    chlorite, saponite = load_layer("chlorite"), saponite_layer()
    expected = chlorite.thickness + saponite.thickness
    assert expected == pytest.approx(31.1, abs=0.5)
    spacings = _maxima(grid, _orders(chlorite, saponite, CORRENSITE_FRACTION, 1.0, grid))
    for order in (1, 2, 3, 4):
        assert min(abs(d - expected / order) for d in spacings) < 0.6, order


def test_corrensite_collapses_on_the_smectite_half_only():
    """The chlorite layer does not move, so the superlattice shifts by the rest."""
    grid = np.arange(2.0, 36.0, 0.02)
    chlorite = load_layer("chlorite")
    dried = _maxima(grid, _orders(chlorite, air_dried_saponite_layer(),
                                  CORRENSITE_FRACTION, 1.0, grid))
    expected = chlorite.thickness + AIR_DRIED_SMECTITE_D001
    assert min(abs(d - expected) for d in dried) < 0.6


def test_maximum_ordering_at_equal_proportions_is_strict_alternation():
    transition = ordered_transition(CORRENSITE_FRACTION, 1.0)
    assert transition[0][1] == pytest.approx(1.0)
    assert transition[1][0] == pytest.approx(1.0)


def _small(species):
    from clayquant.library import build_library
    return build_library(
        grid=np.arange(2.0, 36.0, 0.05), orientations=(0.3,), illite_smectite=(0.8,),
        chlorite_smectite=(0.9,), chlorite_iron=((0.0877, 0.0580),),
        illite_composition=((1.0, 0.0),), csds_means=(15.0,),
        host_thicknesses={}, domain_sizes={}, smectite_species=species,
    )


def test_the_species_is_a_build_switch_and_not_a_second_column():
    """Two columns 0.999 collinear would be divided between on noise alone."""
    for species in ("dioctahedral", "trioctahedral"):
        library = _small(species)
        phases = sorted({entry.phase for entry in library.entries})
        assert phases.count("smectite_EG") == 1
        assert "saponite_EG" not in phases
        assert library.metadata["smectite_species"] == species


def test_the_switch_reaches_the_expandable_entries():
    dio = [e for e in _small("dioctahedral").entries if e.phase == "smectite_EG"][0]
    tri = [e for e in _small("trioctahedral").entries if e.phase == "smectite_EG"][0]
    assert tri.unit_mass != pytest.approx(dio.unit_mass)
    assert tri.unit_mass == pytest.approx(saponite_layer().mass)
    assert dio.unit_mass == pytest.approx(eg_smectite_layer().mass)


def test_corrensite_is_trioctahedral_whichever_way_the_switch_is_set():
    """Its smectite half is a saponite by definition, not by choice."""
    dio = [e for e in _small("dioctahedral").entries if e.phase == "corrensite"][0]
    tri = [e for e in _small("trioctahedral").entries if e.phase == "corrensite"][0]
    assert dio.unit_mass == pytest.approx(tri.unit_mass)
    assert np.allclose(dio.intensity, tri.intensity)


def test_an_unknown_species_is_refused():
    with pytest.raises(ValueError, match="smectite_species"):
        _small("octahedral")


def test_corrensite_carries_its_composition_and_its_ordering():
    entry = [e for e in _small("dioctahedral").entries if e.phase == "corrensite"][0]
    assert entry.fraction == pytest.approx(CORRENSITE_FRACTION)
    assert entry.ordering == pytest.approx(1.0)
