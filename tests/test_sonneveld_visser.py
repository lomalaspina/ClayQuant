"""The Sonneveld & Visser (1975) background estimator and noise level.

The properties tested are the ones the method is claimed to have in the paper -
a straight background survives, a curved one survives up to a stated curvature,
peaks do not - plus the two things a 1975 paper could not state in terms this
implementation needs: how far the erosion reaches, and what the curvature
allowance is worth on data of 10^5 counts rather than 255.
"""

import numpy as np
import pytest

from clayquant.background import (
    ESTIMATOR_LABELS,
    ESTIMATORS,
    SONNEVELD_VISSER_CURVATURE,
    BackgroundModel,
    StrippedBackground,
    baseline_estimate,
    noise_level,
    snip_baseline,
    sonneveld_visser_baseline,
    sonneveld_visser_iterations,
)

STEP = 0.0167


def grid(low: float = 3.0, high: float = 40.0) -> np.ndarray:
    return np.arange(low, high, STEP)


def gaussian(x, centre, height, fwhm):
    return height * np.exp(-0.5 * ((x - centre) / (fwhm / 2.3548)) ** 2)


# --- what must survive ----------------------------------------------------


def test_a_straight_background_is_returned_unchanged():
    """The paper's Fig. 1(a): the iteration's fixed point on a line is the line."""
    x = grid()
    line = 2000.0 - 30.0 * x
    assert np.allclose(sonneveld_visser_baseline(x, line), line, atol=1e-6)


def test_a_convex_tail_is_returned_unchanged():
    """The direct-beam tail of an oriented mount must not be eroded.

    Where the background is convex the mean of two neighbours is never below the
    point itself, so the rule cannot fire.  This is the property that makes the
    method safe at low angle, and it is exact rather than approximate.
    """
    x = grid()
    tail = 5.0e4 / x**1.5
    assert np.allclose(sonneveld_visser_baseline(x, tail), tail, atol=1e-9)


def test_the_curvature_allowance_is_the_second_difference():
    """``c`` bounds the downward curvature kept: ``d2p/dx2 >= -2c/h^2``.

    Verified by putting a parabola of known curvature through the sampler and
    finding where the retention switches: just above the value the relation
    predicts the parabola survives to 0.01 % of its range, just below it does
    not.
    """
    x = grid()
    h = 0.2
    second_derivative = -4.0
    parabola = 3000.0 + 0.5 * second_derivative * (x - 0.5 * (x[0] + x[-1])) ** 2
    span = float(np.ptp(parabola[:: int(round(h / STEP))]))
    needed = -second_derivative * h**2 / 2.0

    kept = sonneveld_visser_baseline(x, parabola, sampling=h, curvature=1.1 * needed / span)
    eroded = sonneveld_visser_baseline(x, parabola, sampling=h, curvature=0.5 * needed / span)
    assert np.max(np.abs(kept - parabola)) < 1e-4 * np.ptp(parabola)
    assert np.max(parabola - eroded) > 20.0 * np.max(np.abs(kept - parabola))


# --- what must not ---------------------------------------------------------


@pytest.mark.parametrize("fwhm", [0.1, 0.2, 0.4])
def test_a_clay_basal_reflection_is_removed(fwhm):
    x = grid()
    y = 1000.0 + gaussian(x, 20.0, 5000.0, fwhm)
    baseline = sonneveld_visser_baseline(x, y)
    assert baseline.max() - 1000.0 < 0.01 * 5000.0


def test_the_reach_follows_the_square_root_of_the_passes():
    """Not one sample per pass: the rule is a diffusion step, so reach ~ sqrt(n).

    Getting this wrong by assuming a linear reach overstates what 30 passes
    remove by a factor of five, which is why :func:`sonneveld_visser_iterations`
    inverts the measured law instead.
    """
    x = grid()

    def half_width(passes):
        low, high = 0.02, 30.0
        for _ in range(40):
            middle = 0.5 * (low + high)
            y = 1000.0 + gaussian(x, 20.0, 5000.0, middle)
            baseline = sonneveld_visser_baseline(
                x, y, sampling=0.2, iterations=passes, curvature=0.0
            )
            if (baseline.max() - 1000.0) / 5000.0 < 0.5:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    # Quadrupling the passes doubles the reach; it does not quadruple it.
    assert half_width(120) / half_width(30) == pytest.approx(2.0, abs=0.1)
    assert half_width(30) / half_width(0 + 5) == pytest.approx(np.sqrt(6.0), abs=0.15)


@pytest.mark.parametrize("window", [1.0, 2.0, 4.0, 8.0])
def test_the_requested_window_is_the_width_half_removed(window):
    x = grid()
    y = 1000.0 + gaussian(x, 20.0, 5000.0, window)
    baseline = sonneveld_visser_baseline(x, y, sampling=0.2, window=window, curvature=0.0)
    assert (baseline.max() - 1000.0) / 5000.0 == pytest.approx(0.5, abs=0.06)


def test_iterations_grow_as_the_square_of_the_window():
    assert sonneveld_visser_iterations(4.0, 0.2) == pytest.approx(
        4 * sonneveld_visser_iterations(2.0, 0.2), rel=0.1
    )


# --- the parameters, and which of them matter -----------------------------


def test_the_curvature_allowance_is_inert_on_a_counts_scale():
    """The paper's ``c``, the literal 0.02 and zero agree on real magnitudes.

    Worth an assertion rather than a remark: the paper introduces ``c`` as the
    thing that lets a curved background through, so a reader expects it to
    matter, and anyone porting the method is tempted to spend time on it.  On a
    pattern of 10^4 counts the sampling and the pass count decide the answer and
    ``c`` does not.
    """
    x = grid()
    y = 8000.0 + gaussian(x, 22.0, 400.0, 14.0)
    span = float(np.ptp(y[::12]))
    paper = sonneveld_visser_baseline(x, y, curvature=SONNEVELD_VISSER_CURVATURE)
    literal = sonneveld_visser_baseline(x, y, curvature=0.02 / span)
    none = sonneveld_visser_baseline(x, y, curvature=0.0)
    assert np.max(np.abs(paper - literal)) < 1e-3 * span
    assert np.max(np.abs(paper - none)) < 1e-3 * span


def test_the_order_of_travel_stops_mattering_once_the_erosion_settles():
    x = grid()
    y = 4.0e4 / x**1.6 + 600.0
    for centre, height, fwhm in ((6.2, 9000, 0.18), (12.4, 12000, 0.15), (26.7, 15000, 0.12)):
        y = y + gaussian(x, centre, height, fwhm)

    def difference(passes):
        a = sonneveld_visser_baseline(x, y, sampling=0.2, iterations=passes, sequential=True)
        b = sonneveld_visser_baseline(x, y, sampling=0.2, iterations=passes, sequential=False)
        return float(np.max(np.abs(a - b)))

    assert difference(1) > 100.0
    assert difference(83) == 0.0


def test_the_sampling_must_be_positive():
    x = grid()
    with pytest.raises(ValueError):
        sonneveld_visser_baseline(x, np.ones_like(x), sampling=0.0)
    with pytest.raises(ValueError):
        sonneveld_visser_baseline(x, np.ones_like(x), curvature=-1.0)


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError):
        sonneveld_visser_baseline(grid(), np.ones(5))


def test_the_baseline_never_exceeds_the_measurement():
    rng = np.random.default_rng(7)
    x = grid()
    y = 500.0 + rng.normal(0.0, 30.0, x.size) + gaussian(x, 12.4, 9000.0, 0.15)
    assert np.all(sonneveld_visser_baseline(x, y) <= y + 1e-9)


# --- against the other estimator ------------------------------------------


def test_it_sits_above_the_peak_stripped_baseline():
    """Why SNIP stays the default: Sonneveld-Visser leaves more in the baseline.

    Leaving more in the baseline means taking more out of the signal.  Over the
    39 real patterns available the difference runs from -0.18 % of the intensity
    range to +36 %, so it is one-sided for practical purposes without being
    exactly so.  On a sharp-peaked pattern the two agree closely; on a diffuse
    one the excess is large, and what is lost is broad basal intensity.
    """
    x = grid()
    diffuse = 2000.0 + gaussian(x, 20.0, 8000.0, 9.0) + gaussian(x, 12.4, 4000.0, 0.2)
    sv = sonneveld_visser_baseline(x, diffuse, window=4.0)
    snip = snip_baseline(x, diffuse, window=4.0)
    assert np.min(sv - snip) > -0.005 * np.ptp(diffuse)
    assert np.max(sv - snip) > 0.02 * np.ptp(diffuse)

    sharp = 1000.0 + sum(gaussian(x, c, 9000.0, 0.15) for c in (8.8, 12.4, 19.9, 26.7))
    sv = sonneveld_visser_baseline(x, sharp, window=4.0)
    snip = snip_baseline(x, sharp, window=4.0)
    assert np.max(np.abs(sv - snip)) < 0.03 * np.ptp(sharp)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_both_estimators_drive_the_fitted_and_the_non_parametric_background(estimator):
    x = grid()
    y = 4.0e4 / x**1.6 + 600.0 + gaussian(x, 8.8, 9000.0, 0.2)
    fit = BackgroundModel(chebyshev_degree=4, inverse=True).fit(x, y, estimator=estimator)
    assert np.all(np.isfinite(fit(x)))
    stripped = StrippedBackground.fit(x, y, estimator=estimator)
    assert stripped.baseline.shape == x.shape
    assert ESTIMATOR_LABELS[estimator] in stripped.model


def test_an_unknown_estimator_is_named_in_the_error():
    x = grid()
    with pytest.raises(ValueError, match="sonneveld-visser"):
        baseline_estimate(x, np.ones_like(x), estimator="polynomial")


# --- the noise level, Sec. 3.2 --------------------------------------------


def test_the_noise_level_recovers_a_known_sigma():
    rng = np.random.default_rng(3)
    difference = rng.normal(0.0, 12.0, 4000)
    level = noise_level(difference)
    assert level.sigma == pytest.approx(12.0, rel=0.06)
    assert level.mean == pytest.approx(0.0, abs=1.0)
    assert level.converged


def test_peaks_do_not_inflate_the_noise_level():
    """The point of the clipping: sigma must describe the noise, not the peaks."""
    rng = np.random.default_rng(11)
    difference = rng.normal(0.0, 12.0, 4000)
    with_peaks = difference.copy()
    with_peaks[::200] += 4000.0
    naive = float(np.std(with_peaks))
    level = noise_level(with_peaks)
    assert naive > 10.0 * level.sigma
    assert level.sigma == pytest.approx(12.0, rel=0.1)
    assert level.rejected >= with_peaks[::200].size


def test_the_clipping_is_one_sided():
    """A symmetric clip would measure a sigma about a third too small.

    The contaminating population lies only above the background, so rejecting
    below the mean throws away noise the estimate needs.  This fixes the
    behaviour against that mistake rather than merely documenting it.
    """
    rng = np.random.default_rng(5)
    difference = rng.normal(0.0, 10.0, 20000)
    level = noise_level(difference)
    symmetric = float(np.std(difference[np.abs(difference) <= 3.0 * 10.0]))
    assert level.sigma == pytest.approx(10.0, rel=0.05)
    assert level.sigma > symmetric * 0.99


def test_the_noise_range_is_six_sigma():
    rng = np.random.default_rng(13)
    level = noise_level(rng.normal(0.0, 5.0, 3000))
    assert level.range == pytest.approx(6.0 * level.sigma)
    assert level.threshold == pytest.approx(level.mean + 3.0 * level.sigma)


def test_a_subsample_gives_the_same_level_as_all_the_data():
    """The paper's N ~ 500 was a concession to a 1975 computer, not a requirement."""
    rng = np.random.default_rng(17)
    difference = rng.normal(0.0, 20.0, 9000)
    assert noise_level(difference, sample=500).sigma == pytest.approx(
        noise_level(difference).sigma, rel=0.12
    )


def test_a_pattern_that_is_all_peak_still_returns_a_level():
    level = noise_level(np.arange(100.0) ** 2)
    assert level.sigma > 0.0
    assert level.converged


def test_too_little_data_is_refused():
    with pytest.raises(ValueError):
        noise_level(np.array([1.0]))
    with pytest.raises(ValueError):
        noise_level(np.zeros(10), sample=1)
