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
