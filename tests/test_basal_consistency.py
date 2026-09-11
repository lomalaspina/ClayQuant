"""Cross-check the one-dimensional layer models against the full 3D structures.

The basal series of a phase can be computed two independent ways: from the
complete crystal structure as ordinary ``hkl`` reflections, or from the layer
projected onto ``c*`` and stacked by the interstratification model.  They must
agree.  This catches errors in the projection of multi-layer polytypes (illite
2M1 and kaolinite 2M have two layers per cell), in the layer origin convention,
and in the single-component limit of the stacking model.

Skipped when the ICSD CIF files are not installed; see clayquant.models.
"""

from __future__ import annotations

import numpy as np
import pytest

from clayquant.emission import CU_KA1
from clayquant.mixed_layer import MixedLayerStack, fixed_csds
from clayquant.models import CIF_SOURCES, available_phases, load_crystal, load_layer
from clayquant.pattern import (
    Instrument,
    _build_from_reflections,
    basal_pattern,
    reflections,
    two_theta_grid,
)
from clayquant.profile import PeakShape

N_LAYERS = 40
GRID = two_theta_grid(2.0, 45.0, 0.005)

PHASES = [key for key in CIF_SOURCES if available_phases()[key]]
requires_cifs = pytest.mark.skipif(not PHASES, reason="ICSD CIF files are not installed")


@requires_cifs
@pytest.mark.parametrize("key", PHASES)
def test_basal_series_matches_three_dimensional_calculation(key):
    crystal = load_crystal(key)
    layer = load_layer(key)

    from_layer = basal_pattern(
        MixedLayerStack(layer, layer, 1.0, csds=fixed_csds(N_LAYERS)),
        GRID,
        Instrument(emission=CU_KA1, peak_shape=PeakShape(w=0.002, eta=0.0)),
    ).intensity

    instrument_3d = Instrument(
        emission=CU_KA1,
        peak_shape=PeakShape(
            w=0.002, eta=0.0, size_c=N_LAYERS * layer.thickness, size_ab=1.0e6
        ),
    )
    d_min = CU_KA1.principal_wavelength / (2.0 * np.sin(np.radians(GRID[-1] / 2.0)))
    found = reflections(crystal, d_min)
    from_crystal = _build_from_reflections(found.select(found.is_basal), GRID, instrument_3d, 1.0)

    areas = []
    for order in range(1, 12):
        d = layer.thickness / order
        argument = CU_KA1.principal_wavelength / (2.0 * d)
        if argument >= 1.0:
            break
        two_theta = np.degrees(2.0 * np.arcsin(argument))
        if not GRID[0] + 0.5 < two_theta < GRID[-1] - 0.5:
            continue
        window = (GRID > two_theta - 0.35) & (GRID < two_theta + 0.35)
        areas.append(
            (
                float(np.trapezoid(from_crystal[window], GRID[window])),
                float(np.trapezoid(from_layer[window], GRID[window])),
            )
        )

    assert len(areas) >= 3, f"{key}: too few basal orders in range to compare"
    reference = np.array([a for a, _ in areas])
    model = np.array([b for _, b in areas])
    scale = reference.sum() / model.sum()
    ratios = scale * model / reference
    assert np.allclose(ratios, 1.0, atol=0.08), f"{key}: basal intensity ratios {ratios}"


@requires_cifs
@pytest.mark.parametrize("key", PHASES)
def test_layer_thickness_matches_basal_spacing(key):
    crystal = load_crystal(key)
    layer = load_layer(key)
    expected = crystal.d001 / CIF_SOURCES[key].layers_per_cell
    assert layer.thickness == pytest.approx(expected, rel=1e-9)


@requires_cifs
def test_known_basal_spacings():
    # Textbook basal spacings of the clay minerals, to catch a metric error.
    assert load_layer("illite").thickness == pytest.approx(10.0, abs=0.1)
    assert load_layer("chlorite").thickness == pytest.approx(14.2, abs=0.2)
    assert load_layer("kaolinite_1M").thickness == pytest.approx(7.15, abs=0.1)
    assert load_layer("kaolinite_2M").thickness == pytest.approx(7.15, abs=0.1)
