"""What the background estimator must and must not do to a clay pattern.

The low-angle end of an oriented mount is air scatter and the shoulder of the
direct beam: a steep, smooth, *convex* rise carrying thousands of counts.  It is
background, it is not a peak, and an estimator that removes it leaves that
intensity behind as signal the fit is then asked to explain as diffraction.

The tests below pin the property that prevents this.  Where a background is
convex, the mean of two symmetric neighbours is never below the point itself, so
the stripping operation must leave it exactly unchanged - whatever width is
asked for.  An earlier implementation invented the neighbours it lacked at the
ends of the scan and let that invention propagate inwards, and on a known tail
of 1820 counts it returned 546.
"""

from __future__ import annotations

import numpy as np
import pytest

from clayquant import background
from clayquant.background import (
    BackgroundModel,
    snip_baseline,
    snip_edge_width,
)

STEP = 0.0167
TWO_THETA = np.arange(3.0, 40.0, STEP)


def direct_beam_tail(two_theta: np.ndarray) -> np.ndarray:
    """A background of the shape an oriented mount actually measures."""
    return (
        300.0
        + 4000.0 / (two_theta - 1.0) ** 1.6
        + 200.0 * np.exp(-(two_theta - 3.0) / 12.0)
    )


def clay_peaks(two_theta: np.ndarray) -> np.ndarray:
    lines = [
        (6.2, 1800.0, 0.35),   # a broad expandable band
        (8.85, 6000.0, 0.12),  # illite 001
        (12.4, 3200.0, 0.12),  # kaolinite 001 / chlorite 002
        (17.8, 1700.0, 0.10),
        (19.0, 800.0, 0.10),
        (20.85, 700.0, 0.09),  # quartz 100
        (24.9, 900.0, 0.09),
        (26.65, 7000.0, 0.08),  # quartz 101
        (27.9, 1500.0, 0.09),
        (35.0, 400.0, 0.10),
    ]
    return sum(h * np.exp(-0.5 * ((two_theta - c) / w) ** 2) for c, h, w in lines)


@pytest.mark.parametrize("window", [1.0, 2.0, 4.0, 6.0, 8.0])
def test_a_convex_background_survives_untouched(window):
    """With no peaks present there is nothing to strip, at any width."""
    truth = direct_beam_tail(TWO_THETA)
    estimate = snip_baseline(TWO_THETA, truth, window=window)
    assert np.allclose(estimate, truth, rtol=1e-6, atol=1.0)


@pytest.mark.parametrize("window", [2.0, 4.0, 6.0])
def test_the_tail_survives_under_peaks(window):
    data = direct_beam_tail(TWO_THETA) + clay_peaks(TWO_THETA)
    truth = direct_beam_tail(TWO_THETA)
    estimate = snip_baseline(TWO_THETA, data, window=window)

    # Away from the ends, where the comparison is two-sided, the estimate is the
    # background to a few counts in thousands.
    edge = snip_edge_width(TWO_THETA, window)
    interior = slice(edge, len(TWO_THETA) - edge)
    assert np.max(np.abs(estimate[interior] - truth[interior])) < 0.02 * truth.max()

    # And the low-angle tail is kept, rather than being stripped as if it were a
    # very broad peak: this is the failure that put 1300 counts of direct beam
    # into the quantified signal.
    assert estimate[0] > 0.9 * truth[0]
    assert np.all(estimate[:edge] > 0.75 * truth[:edge])


def test_peaks_are_still_removed():
    data = direct_beam_tail(TWO_THETA) + clay_peaks(TWO_THETA)
    estimate = snip_baseline(TWO_THETA, data, window=4.0)
    for centre in (8.85, 12.4, 26.65):
        i = int(np.argmin(np.abs(TWO_THETA - centre)))
        # The whole peak is background-free: what is left under it is the tail.
        assert estimate[i] < 0.35 * data[i]


def test_a_narrow_window_keeps_more_of_the_peak_wings():
    """The width control means peak width, and only peak width."""
    data = direct_beam_tail(TWO_THETA) + clay_peaks(TWO_THETA)
    narrow = snip_baseline(TWO_THETA, data, window=0.5)
    wide = snip_baseline(TWO_THETA, data, window=6.0)
    i = int(np.argmin(np.abs(TWO_THETA - 6.2)))  # the broad band
    assert narrow[i] > wide[i]
    # It does not mean tail: both keep the low-angle rise.
    assert narrow[0] == pytest.approx(wide[0], rel=0.05)


def test_the_estimate_never_exceeds_the_measurement():
    rng = np.random.default_rng(20260913)
    data = rng.poisson(direct_beam_tail(TWO_THETA) + clay_peaks(TWO_THETA)).astype(float)
    assert np.all(snip_baseline(TWO_THETA, data, window=4.0) <= data)


def test_the_fitted_model_follows_the_tail():
    """The whole point: the background the user sees must reach the data at 3 deg."""
    data = direct_beam_tail(TWO_THETA) + clay_peaks(TWO_THETA)
    fit = BackgroundModel(chebyshev_degree=4, inverse=True, inverse_offset=1.0).fit(
        TWO_THETA, data, snip_window=4.0
    )
    values = fit(TWO_THETA)
    truth = direct_beam_tail(TWO_THETA)
    assert values[0] > 0.75 * truth[0]
    low = TWO_THETA <= 6.0
    assert np.mean(np.abs(values[low] - truth[low])) < 0.1 * truth[low].mean()


# --- which estimate the main component is fitted to -------------------------


def a_clay_pattern():
    """A structured background with sharp peaks on it, as a clay mount gives."""
    x = np.arange(3.0, 40.0, 0.0167)
    background = 300.0 + 4000.0 / (x + 1.0) + 180.0 * np.exp(-0.5 * ((x - 24.0) / 7.0) ** 2)
    y = background.copy()
    for centre, height in ((6.2, 1200.0), (8.9, 7000.0), (12.5, 3000.0), (17.8, 1800.0),
                           (20.9, 900.0), (25.1, 1400.0), (26.6, 8000.0), (27.9, 1900.0)):
        y = y + height * np.exp(-0.5 * ((x - centre) / 0.05) ** 2)
    return x, y, background


def test_the_polynomial_and_chebyshev_bases_give_the_same_fit():
    """Pinned because it is a surprise and because it is stated in the interface.

    Both are mapped onto the same variable over the same range and both span
    every polynomial up to their degree, so an unconstrained least squares of
    either against the same target is the same curve.  They differ in
    conditioning and in nothing else.  Other software offers two options that do
    differ, which is why saying so where the choice is made is worth the words.
    """
    x, y, _ = a_clay_pattern()
    for degree in (2, 4, 7, 11):
        monomial = background.BackgroundModel(
            polynomial_degree=degree, inverse=True, inverse_offset=1.0,
        ).fit(x, y, snip_window=4.0)
        chebyshev = background.BackgroundModel(
            polynomial_degree=None, chebyshev_degree=degree,
            inverse=True, inverse_offset=1.0,
        ).fit(x, y, snip_window=4.0)
        assert monomial(x) == pytest.approx(chebyshev(x), rel=1e-6, abs=1e-6)


def test_every_estimator_is_reachable_and_stays_under_the_peaks():
    x, y, truth = a_clay_pattern()
    for name in background.ESTIMATORS:
        estimate = background.baseline_estimate(x, y, window=4.0, estimator=name)
        assert estimate.shape == x.shape
        assert np.all(np.isfinite(estimate))
        # Nowhere near the peak tops, and not below zero.
        assert estimate.max() < 0.5 * y.max()
        assert estimate.min() > -1.0


def test_asymmetric_least_squares_follows_a_structured_background():
    x, y, truth = a_clay_pattern()
    estimate = background.als_baseline(y)
    middle = (x > 15.0) & (x < 35.0)
    assert np.mean(np.abs(estimate[middle] - truth[middle])) < 80.0


def test_the_rolling_percentile_follows_a_structured_background():
    x, y, truth = a_clay_pattern()
    estimate = background.percentile_baseline(x, y, window=4.0)
    middle = (x > 15.0) & (x < 35.0)
    assert np.mean(np.abs(estimate[middle] - truth[middle])) < 80.0


def test_a_peak_strip_of_a_clay_pattern_is_the_smooth_one():
    """Why the choice of estimate matters more than the polynomial degree.

    The main component is fitted to the estimate, so how much structure the
    estimate has decides how much a higher degree can do.  A peak strip over a
    window wide enough for a clay basal reflection smooths the background's own
    structure away with them, and then a low degree describes what is left.
    """
    x, y, truth = a_clay_pattern()
    stripped = background.baseline_estimate(x, y, window=4.0, estimator="snip")
    percentile = background.baseline_estimate(x, y, window=4.0, estimator="percentile")
    middle = (x > 15.0) & (x < 35.0)
    assert np.mean(np.abs(stripped[middle] - truth[middle])) > np.mean(
        np.abs(percentile[middle] - truth[middle])
    )


def test_a_bad_percentile_is_refused():
    x, y, _ = a_clay_pattern()
    with pytest.raises(ValueError, match="percentile"):
        background.percentile_baseline(x, y, percentile=0.0)
    with pytest.raises(ValueError, match="percentile"):
        background.percentile_baseline(x, y, percentile=100.0)


def test_a_bad_asymmetry_or_smoothness_is_refused():
    _, y, _ = a_clay_pattern()
    with pytest.raises(ValueError, match="asymmetry"):
        background.als_baseline(y, asymmetry=0.0)
    with pytest.raises(ValueError, match="smoothness"):
        background.als_baseline(y, smoothness=0.0)
