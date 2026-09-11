"""End-to-end test of the quantification workflow on a synthetic measurement.

A pattern of known composition is built from the library, given a quartz peak, a
2theta zero error, a polynomial-plus-1/x background and Poisson noise.  The test
then runs the pipeline the GUI drives - zero-error calibration, background
fitting, non-negative least squares - and checks that it recovers what went in.

Skipped when the ICSD CIF files are not installed; see clayquant.models.
"""

from __future__ import annotations

import numpy as np
import pytest

from clayquant.background import BackgroundModel
from clayquant.calibration import (
    QUARTZ_100_D,
    apply_zero_error,
    estimate_zero_error,
    reference_two_theta,
)
from clayquant.diagnostics import kaolinite_collapse, measure_peak, smectite_swelling
from clayquant.library import build_library
from clayquant.models import available_phases
from clayquant.nnls import nnls_fit
from clayquant.optics import Divergence
from clayquant.pattern import Instrument, Pattern, two_theta_grid
from clayquant.profile import PeakShape, pseudo_voigt

pytestmark = pytest.mark.skipif(
    not all(available_phases().values()), reason="ICSD CIF files are not installed"
)

ZERO_ERROR = 0.06
FIT_RANGE = (4.0, 34.0)
COMPOSITION = {"illite PO=0.2": 0.50, "I/S 0.80/0.20 PO=0.2": 0.30, "kaolinite_1M PO=0.2": 0.20}


@pytest.fixture(scope="module")
def library():
    return build_library(
        grid=two_theta_grid(2.0, 36.0, 0.02),
        instrument=Instrument(
            peak_shape=PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6, size_ab=400.0),
            divergence=Divergence(),
        ),
        orientations=(0.2,),
        illite_smectite=(0.80,),
        chlorite_smectite=(0.90,),
        csds_means=(10.0,),
        # One spacing per phase keeps the fixture small and its entry names
        # free of the d= tag.
        host_thicknesses={},
    )


def add_quartz(two_theta: np.ndarray, height: float = 400.0) -> np.ndarray:
    center = reference_two_theta(QUARTZ_100_D)
    return height * pseudo_voigt(two_theta - center, 0.12, 0.5) * 0.12


def true_background(two_theta: np.ndarray) -> np.ndarray:
    return 40.0 + 0.6 * (two_theta - 19.0) ** 2 * 0.02 + 900.0 / two_theta


@pytest.fixture(scope="module")
def synthetic(library):
    two_theta = library.two_theta
    clay = np.zeros_like(two_theta)
    for name, weight in COMPOSITION.items():
        clay += weight * library.entries[library.names.index(name)].intensity

    signal = 6000.0 * clay + add_quartz(two_theta)
    background = true_background(two_theta)
    rng = np.random.default_rng(20260911)
    counts = rng.poisson(np.clip(signal + background, 0.0, None)).astype(float)

    return Pattern(two_theta + ZERO_ERROR, counts, name="synthetic EG mount")


def test_zero_error_is_recovered(synthetic):
    result = estimate_zero_error(synthetic, window=0.8)
    assert result.shift == pytest.approx(ZERO_ERROR, abs=0.02)
    corrected = apply_zero_error(synthetic, result.shift)
    quartz = measure_peak(corrected, (reference_two_theta(QUARTZ_100_D) - 0.5,
                                      reference_two_theta(QUARTZ_100_D) + 0.5))
    assert quartz.position == pytest.approx(reference_two_theta(QUARTZ_100_D), abs=0.03)


def test_background_with_accumulated_inverse_term(synthetic):
    corrected = apply_zero_error(synthetic, estimate_zero_error(synthetic, window=0.8).shift)
    window = (corrected.two_theta >= FIT_RANGE[0]) & (corrected.two_theta <= FIT_RANGE[1])
    two_theta = corrected.two_theta[window]
    intensity = corrected.intensity[window]
    truth = true_background(two_theta)

    plain = BackgroundModel(polynomial_degree=3)
    with_inverse = BackgroundModel(polynomial_degree=3, inverse=True)
    assert with_inverse.n_terms == plain.n_terms + 1
    assert with_inverse.components == ["polynomial", "inverse"]

    fit_plain = plain.fit(two_theta, intensity)
    fit_inverse = with_inverse.fit(two_theta, intensity)

    # The synthetic background contains a genuine 1/x term, so accumulating that
    # component on top of the polynomial must describe the stripped background
    # better.
    assert fit_inverse.r_squared(two_theta) > fit_plain.r_squared(two_theta)
    assert fit_inverse.r_squared(two_theta) > 0.99

    # Whatever the components, a background must stay under the data and must
    # not rise above the true background into the peaks.
    for fit in (fit_plain, fit_inverse):
        values = fit(two_theta)
        assert np.mean(values <= intensity) > 0.99
        assert np.all(values <= truth + 0.05 * truth.max())
        assert np.all(values >= 0.0)


def test_background_components_can_all_be_combined(synthetic):
    corrected = apply_zero_error(synthetic, estimate_zero_error(synthetic, window=0.8).shift)
    window = (corrected.two_theta >= FIT_RANGE[0]) & (corrected.two_theta <= FIT_RANGE[1])
    two_theta = corrected.two_theta[window]
    intensity = corrected.intensity[window]

    combined = BackgroundModel(
        polynomial_degree=2, chebyshev_degree=2, exponential_decay=0.2, inverse=True
    )
    assert combined.components == ["polynomial", "chebyshev", "exponential", "inverse"]
    assert combined.n_terms == 3 + 3 + 1 + 1
    fit = combined.fit(two_theta, intensity)
    assert np.all(np.isfinite(fit(two_theta)))
    assert np.all(fit(two_theta) >= 0.0)


def test_nnls_recovers_the_composition(synthetic, library):
    corrected = apply_zero_error(synthetic, estimate_zero_error(synthetic, window=0.8).shift)
    model = BackgroundModel(polynomial_degree=3, inverse=True)
    window = (corrected.two_theta >= FIT_RANGE[0]) & (corrected.two_theta <= FIT_RANGE[1])
    background = model.fit(corrected.two_theta[window], corrected.intensity[window])

    result = nnls_fit(corrected, library, background=background, range_two_theta=FIT_RANGE)
    assert result.r_wp < 0.20

    # The synthetic mixture was built from patterns normalised to unit maximum,
    # so it is the amplitude fractions that should reproduce its weights.
    shares = dict(zip(result.names, result.amplitude_fraction))
    recovered = {name: shares.get(name, 0.0) for name in COMPOSITION}
    assert sum(recovered.values()) > 0.95, f"most of the fit unassigned: {recovered}"

    order_in = sorted(COMPOSITION, key=COMPOSITION.get, reverse=True)
    order_out = sorted(recovered, key=recovered.get, reverse=True)
    assert order_in == order_out, f"wrong ranking: {recovered}"
    for name, expected in COMPOSITION.items():
        assert recovered[name] == pytest.approx(expected, abs=0.06), f"{name}: {recovered}"

    # Kaolinite 1M and 2M have identical basal series, so the fit must not be
    # asked to tell them apart; it is enough that the kaolinite total is right.
    kaolinite = sum(
        share
        for name, share in zip(result.names, result.amplitude_fraction)
        if name.startswith("kaolinite")
    )
    assert kaolinite == pytest.approx(COMPOSITION["kaolinite_1M PO=0.2"], abs=0.06)


def test_kaolinite_collapse_is_detected(library):
    two_theta = library.two_theta
    illite = library.entries[library.names.index("illite PO=0.2")].intensity
    kaolinite = library.entries[library.names.index("kaolinite_1M PO=0.2")].intensity
    quartz = add_quartz(two_theta)

    air = Pattern(two_theta, 5000.0 * (0.6 * illite + 0.4 * kaolinite) + quartz, name="air")
    # Heating destroys the kaolinite and leaves everything else, on a mount with
    # a different amount of clay: the quartz reference must absorb the scale.
    heated = Pattern(two_theta, 0.7 * (5000.0 * 0.6 * illite) + 0.7 * quartz, name="heated")

    result = kaolinite_collapse(air, heated)
    assert result.scale == pytest.approx(1.0 / 0.7, rel=0.05)
    assert result.kaolinite_detected
    assert result.collapse_fraction > 0.9
    assert not result.chlorite_indicated


def test_chlorite_survives_heating(library):
    two_theta = library.two_theta
    chlorite = library.entries[library.names.index("chlorite PO=0.2")].intensity
    quartz = add_quartz(two_theta)
    air = Pattern(two_theta, 5000.0 * chlorite + quartz, name="air")
    heated = Pattern(two_theta, 5000.0 * chlorite + quartz, name="heated")

    result = kaolinite_collapse(air, heated)
    assert not result.kaolinite_detected
    assert result.chlorite_indicated


def test_smectite_swelling_is_detected(library):
    two_theta = library.two_theta
    illite = library.entries[library.names.index("illite PO=0.2")].intensity
    glycolated = library.entries[library.names.index("I/S 0.80/0.20 PO=0.2")].intensity
    quartz = add_quartz(two_theta)

    # Air-dried: a 15 A reflection from two water layers; glycolated: the
    # calculated I/S pattern, whose low-angle reflection sits at higher d.
    air_peak = 3000.0 * pseudo_voigt(two_theta - 5.89, 0.5, 0.5) * 0.5
    air = Pattern(two_theta, 4000.0 * illite + air_peak + quartz, name="air")
    glycol = Pattern(two_theta, 4000.0 * glycolated + quartz, name="glycol")

    result = smectite_swelling(air, glycol, window=(4.2, 8.0))
    assert result.expandable_detected
    assert result.shift_d > 0.5
    assert result.air.d_spacing == pytest.approx(15.0, abs=0.3)
    assert result.glycol.d_spacing > 12.5
