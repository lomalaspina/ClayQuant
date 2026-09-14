"""Fitting a chlorite's octahedral iron to a pattern of the pure mineral.

A chlorite's two octahedral sheets take iron for magnesium in proportions that
differ between deposits, iron scatters about twice as strongly as the magnesium
it replaces, and the sheets sit at different heights in the layer - so the
composition acts on the basal intensities the quantification is read from, and
one published clinochlore cannot describe two chlorites of different iron
content.

The tests below are of the arithmetic and of the recovery: a pattern calculated
from a known composition must give that composition back.  What they cannot test
is whether a real chlorite's iron is distributed between the two sheets the way
this two-parameter model says; that is what a measurement of the pure mineral is
for, and what the residual reports.
"""

from __future__ import annotations

import numpy as np
import pytest

from clayquant.composition import (
    ChloriteComposition,
    fit_chlorite_iron,
    measure_basal_series,
    two_theta_of,
)
from clayquant.mixed_layer import MixedLayerStack, lognormal_csds
from clayquant.models import CHLORITE_OCTAHEDRA, available_phases, chlorite_crystal, chlorite_layer
from clayquant.optics import Divergence
from clayquant.pattern import Instrument, Pattern, basal_pattern
from clayquant.profile import PeakShape

pytestmark = pytest.mark.skipif(
    not available_phases()["chlorite"], reason="the chlorite CIF is not installed"
)

INSTRUMENT = Instrument(
    peak_shape=PeakShape(u=0.004, v=-0.001, w=0.002, eta=0.5, size_ab=600.0),
    divergence=Divergence(specimen_length=20.0, goniometer_radius=240.0, divergence=0.5),
)
GRID = np.arange(3.0, 40.0, 0.0167)


def synthetic(iron_2to1, iron_hydroxide, mean_layers=25.0, name="synthetic chlorite"):
    """A pattern of a pure chlorite of known composition, with a flat background."""
    layer = chlorite_layer(iron_2to1, iron_hydroxide)
    stack = MixedLayerStack(layer, layer, 1.0, csds=lognormal_csds(mean_layers))
    intensity = basal_pattern(stack, GRID, INSTRUMENT).intensity
    return Pattern(GRID, 50.0 + 1000.0 * intensity / intensity.max(), name=name)


def test_the_iron_goes_where_it_is_asked_to_go():
    crystal = chlorite_crystal(0.4, 0.15)
    occupancy = {site.label: site.occupancy for site in crystal.sites}
    for label in CHLORITE_OCTAHEDRA["2:1"]:
        expected = 0.4 if label.startswith("Fe") else 0.6
        assert occupancy[label] == pytest.approx(expected)
    for label in CHLORITE_OCTAHEDRA["hydroxide"]:
        expected = 0.15 if label.startswith("Fe") else 0.85
        assert occupancy[label] == pytest.approx(expected)
    # Magnesium and iron on one site must still add to a full site.
    for sheet in CHLORITE_OCTAHEDRA.values():
        pairs = {}
        for label in sheet:
            site = next(s for s in crystal.sites if s.label == label)
            pairs.setdefault(label[2:], 0.0)
            pairs[label[2:]] += site.occupancy
        assert all(total == pytest.approx(1.0) for total in pairs.values())


def test_more_iron_weighs_more_and_leaves_the_cell_alone():
    light = chlorite_crystal(0.0, 0.0)
    heavy = chlorite_crystal(1.0, 1.0)
    assert heavy.cell_mass > light.cell_mass
    assert heavy.volume == pytest.approx(light.volume), "a substitution is not a cell change"
    assert heavy.d001 == pytest.approx(light.d001)
    # Fe(55.85) for Mg(24.31) over the cell's twelve octahedral positions, six
    # in the 2:1 sheet and six in the hydroxide sheet.
    assert heavy.cell_mass - light.cell_mass == pytest.approx(12 * (55.845 - 24.305), rel=0.01)


@pytest.mark.parametrize("value", [-0.01, 1.01, 2.0])
def test_an_occupancy_outside_zero_to_one_is_refused(value):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        chlorite_crystal(value, 0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        chlorite_crystal(0.1, value)


def test_iron_changes_the_basal_ratios_it_is_fitted_from():
    """If it did not, there would be nothing to fit."""
    def ratios(a, b):
        pattern = synthetic(a, b)
        return measure_basal_series(pattern, 14.2782).normalised()

    poor, rich = ratios(0.05, 0.05), ratios(0.7, 0.35)
    assert poor.keys() == rich.keys()
    changed = max(abs(poor[l] - rich[l]) / max(poor[l], 1e-9) for l in poor if l != 2)
    assert changed > 0.2, "the basal ratios must move with the iron content"


def test_the_basal_series_is_integrated_where_the_spacing_puts_it():
    pattern = synthetic(0.3, 0.15)
    series = measure_basal_series(pattern, 14.2782)
    assert series.reference == 2
    assert series.orders[0] == 1
    for order, angle in zip(series.orders, series.two_theta):
        assert angle == pytest.approx(two_theta_of(14.2782, order), abs=1e-6)
    assert series.normalised()[2] == 1.0
    assert all(value >= 0.0 for value in series.normalised().values())


def test_an_order_outside_the_measured_range_is_left_out_not_zeroed():
    pattern = synthetic(0.3, 0.15)
    # 003 sits at 18.63 deg, so a pattern ending at 18.8 contains its centre but
    # not the window its area would be integrated over.
    keep = pattern.two_theta < 18.8
    narrow = Pattern(pattern.two_theta[keep], pattern.intensity[keep], name="narrow")
    series = measure_basal_series(narrow, 14.2782)
    assert series.orders == (1, 2), "003's centre is inside the range, its window is not"


def test_a_reference_order_that_was_not_measured_is_an_error():
    pattern = synthetic(0.3, 0.15)
    with pytest.raises(ValueError, match="nothing to"):
        measure_basal_series(pattern, 14.2782, reference=9)


@pytest.mark.slow
@pytest.mark.parametrize("iron", [(0.30, 0.15), (0.60, 0.30)])
def test_a_known_composition_is_recovered(iron):
    """The test that would catch the fit converging on the wrong answer."""
    pattern = synthetic(*iron)
    fit = fit_chlorite_iron(pattern, INSTRUMENT, mean_layers=(25.0,))
    assert isinstance(fit, ChloriteComposition)
    assert fit.iron_2to1 == pytest.approx(iron[0], abs=0.06)
    assert fit.iron_hydroxide == pytest.approx(iron[1], abs=0.10)
    assert fit.residual < 0.02
    assert fit.residual <= fit.published_residual


def test_the_description_reports_what_was_fitted_and_what_it_beats():
    fit = ChloriteComposition(
        iron_2to1=0.35, iron_hydroxide=0.2, mean_layers=25.0, residual=0.01,
        observed={2: 1.0, 3: 0.5}, calculated={2: 1.0, 3: 0.52},
        published_residual=0.04, measurement="Chlorite_16",
    )
    assert fit.improvement == pytest.approx(4.0)
    text = fit.describe()
    assert "Chlorite_16" in text and "0.350" in text and "0.200" in text
    assert "4.0 times better" in text


def test_the_integration_window_must_not_truncate_the_tails():
    """The defect that produced a confident conclusion from an artefact.

    A basal reflection of a real clay has a full width at half maximum near
    0.1-0.2 deg, so a window of +-0.35 deg looks more than ample.  It is not: the
    diffuse tail from stacking disorder carries much of the area, and a narrow
    window keeps a different fraction of each order - which is exactly the
    quantity a ratio of orders is sensitive to.  Measured on a calculated
    pattern here, where the truth is known; on a real chlorite the same window
    moved the 005 ratio from 0.77 to 1.34.
    """
    pattern = synthetic(0.3, 0.15, mean_layers=8.0)
    narrow = measure_basal_series(pattern, 14.2782, half_width=0.35)
    wide = measure_basal_series(pattern, 14.2782, half_width=1.0)

    kept = {}
    for order, narrow_area, wide_area in zip(narrow.orders, narrow.area, wide.area):
        assert wide_area >= narrow_area, "a wider window cannot hold less"
        kept[order] = narrow_area / wide_area

    assert min(kept.values()) < 0.95, "if the narrow window lost nothing there is nothing to test"
    spread = max(kept.values()) - min(kept.values())
    assert spread > 0.02, (
        "the point of the test: the narrow window keeps a different fraction of "
        f"each order, here {kept}"
    )
    # And the default is the wide one, so a caller who thinks about none of this
    # is not silently handed the truncated answer.
    assert measure_basal_series(pattern, 14.2782).area == wide.area


def test_magnesium_and_iron_on_one_site_share_its_displacement_parameter():
    """They occupy the same site, so they cannot have different B.

    Letting them differ would be unphysical - one site, one environment - and
    degenerate with the occupancy that shares it, since raising B and raising
    the iron fraction both change the same scattering contribution.  The
    published structure already pairs them; this keeps it true of anything
    derived from it, including a B that is being fitted.
    """
    published = chlorite_crystal(0.3, 0.15)
    for sheet in CHLORITE_OCTAHEDRA.values():
        for magnesium, iron in (sheet[0:2], sheet[2:4]):
            a = next(s for s in published.sites if s.label == magnesium)
            b = next(s for s in published.sites if s.label == iron)
            assert a.species.startswith("Mg") and b.species.startswith("Fe")
            assert a.b_iso == pytest.approx(b.b_iso)

    imposed = chlorite_crystal(0.3, 0.15, octahedral_b=1.4)
    octahedral = set(CHLORITE_OCTAHEDRA["2:1"]) | set(CHLORITE_OCTAHEDRA["hydroxide"])
    assert {s.b_iso for s in imposed.sites if s.label in octahedral} == {1.4}
    # And nothing else is touched.
    for a, b in zip(published.sites, imposed.sites):
        if a.label not in octahedral:
            assert a.b_iso == pytest.approx(b.b_iso)
    assert "octahedral B" in imposed.source


def test_a_negative_displacement_parameter_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        chlorite_crystal(0.3, 0.15, octahedral_b=-0.5)


def test_the_octahedral_b_barely_moves_the_higher_basal_orders():
    """Recorded because it was proposed as the cause of a 30 % shortfall.

    exp(-B q^2) is a weak factor at these angles - at 005 of a 14.28 A spacing,
    q^2 = 0.031 A^-2 - but F(00l) is a sum of terms with opposing signs, so the
    sensitivity does not follow from the size of the exponential and has to be
    computed.  Computed, it is small: the whole range from B = 0 to B = 6 moves
    the 004/002 ratio by under a fifth, and in the direction opposite to the one
    a shortfall would need.
    """
    def ratio(b):
        series = measure_basal_series(
            Pattern(GRID, _calculated(chlorite_layer(0.09, 0.06, octahedral_b=b))),
            14.2782, orders=(2, 4), reference=2,
        )
        return series.normalised()[4]

    low, high = ratio(0.0), ratio(6.0)
    assert high < low, "more thermal motion lowers 004 against 002, it does not raise it"
    assert abs(high - low) / low < 0.2, (
        f"B moves 004/002 from {low:.3f} to {high:.3f}; a 30 % shortfall is out of its reach"
    )


def _calculated(layer, mean_layers=8.0):
    stack = MixedLayerStack(layer, layer, 1.0, csds=lognormal_csds(mean_layers))
    return basal_pattern(stack, GRID, INSTRUMENT).intensity
