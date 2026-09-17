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
    SONNEVELD_VISSER_BENDING,
    SONNEVELD_VISSER_CURVATURE,
    SONNEVELD_VISSER_FLOOR,
    SONNEVELD_VISSER_GRANULARITY,
    BackgroundModel,
    StrippedBackground,
    baseline_estimate,
    noise_level,
    peak_groups,
    snip_baseline,
    sonneveld_visser_baseline,
    sonneveld_visser_granularity,
    sonneveld_visser_reach,
    suggest_sonneveld_visser,
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

    kept = sonneveld_visser_baseline(x, parabola, granularity=int(round(h / STEP)), bending=1.1 * needed / (span * SONNEVELD_VISSER_CURVATURE))
    eroded = sonneveld_visser_baseline(x, parabola, granularity=int(round(h / STEP)), bending=0.5 * needed / (span * SONNEVELD_VISSER_CURVATURE))
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
                x, y, granularity=12, iterations=passes, bending=0.0
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
    baseline = sonneveld_visser_baseline(x, y, granularity=12, window=window, bending=0.0)
    assert (baseline.max() - 1000.0) / 5000.0 == pytest.approx(0.5, abs=0.06)


def test_the_granularity_asked_for_a_width_scales_with_it():
    """With the passes fixed, reach is linear in granularity, so this is too."""
    assert sonneveld_visser_granularity(4.0, STEP) == pytest.approx(
        2 * sonneveld_visser_granularity(2.0, STEP), rel=0.1
    )


def test_the_reach_in_degrees_is_reported_for_a_granularity():
    """The number a user needs: granularity is points, reach is degrees.

    HighScore's recommended granularity of 20 reaches 2.4 deg on the 0.01 deg
    film scan the method was written for, and 4.0 deg on this laboratory's
    0.0167 deg step - which is, by coincidence worth noting, the stripping width
    ClayQuant had already settled on independently.
    """
    assert sonneveld_visser_reach(20, 0.01, 30) == pytest.approx(2.4, abs=0.1)
    assert sonneveld_visser_reach(20, STEP, 30) == pytest.approx(4.0, abs=0.1)
    # And it is what the estimator actually does, not just arithmetic.
    x = grid()
    width = sonneveld_visser_reach(20, STEP, 30)
    y = 1000.0 + gaussian(x, 20.0, 5000.0, width)
    baseline = sonneveld_visser_baseline(x, y, granularity=20, bending=0.0)
    assert (baseline.max() - 1000.0) / 5000.0 == pytest.approx(0.5, abs=0.08)


# --- the parameters, and which of them matter -----------------------------


def test_granularity_dominates_the_bending_factor():
    """Which of the two parameters to reach for, settled by measurement.

    The paper introduces ``c`` as the thing that lets a curved background
    through and HighScore puts it on a slider, so a reader expects it to be the
    control that matters.  It is not: over the 39 real patterns available, the
    bending factor's whole span moves the baseline by at most 0.70 % of the
    intensity range while granularity from 10 to 40 moves it by 30 %.  Asserted
    here on a synthetic standing in for them, as an order-of-magnitude claim
    rather than the exact figures, so that the conclusion is held in place
    without the test depending on data that is not in the repository.
    """
    x = grid()
    # A diffuse mount, which is where the contrast lives: on a pattern of sharp
    # peaks on a smooth background neither parameter has much to do, and the
    # ratio falls to about three.  The real disagreement between the two comes
    # from broad scattering, so the stand-in has to have some - a hectorite or
    # an opal-CT-bearing bentonite rather than a well-crystallised chlorite.
    y = (4.0e4 / x**1.6 + 400.0
         + gaussian(x, 6.5, 30000.0, 5.0)
         + gaussian(x, 22.0, 6000.0, 14.0)
         + gaussian(x, 19.9, 9000.0, 0.2)
         + gaussian(x, 26.7, 12000.0, 0.15))
    span = float(np.ptp(y))

    from_bending = np.max(np.abs(
        sonneveld_visser_baseline(x, y, granularity=20, bending=4.0)
        - sonneveld_visser_baseline(x, y, granularity=20, bending=0.0)
    ))
    from_granularity = np.max(np.abs(
        sonneveld_visser_baseline(x, y, granularity=40)
        - sonneveld_visser_baseline(x, y, granularity=10)
    ))
    assert from_bending < 0.03 * span
    assert from_granularity > 10.0 * from_bending


def test_the_order_of_travel_stops_mattering_once_the_erosion_settles():
    x = grid()
    y = 4.0e4 / x**1.6 + 600.0
    for centre, height, fwhm in ((6.2, 9000, 0.18), (12.4, 12000, 0.15), (26.7, 15000, 0.12)):
        y = y + gaussian(x, centre, height, fwhm)

    def difference(passes):
        a = sonneveld_visser_baseline(x, y, granularity=12, iterations=passes, sequential=True)
        b = sonneveld_visser_baseline(x, y, granularity=12, iterations=passes, sequential=False)
        return float(np.max(np.abs(a - b)))

    assert difference(1) > 100.0
    assert difference(83) == 0.0


def test_the_parameters_are_checked():
    x = grid()
    with pytest.raises(ValueError):
        sonneveld_visser_baseline(x, np.ones_like(x), bending=-1.0)


def test_granularity_zero_means_every_point():
    """HighScore's granularity slider starts at 0, and 0 can only mean this.

    Accepted rather than refused so that a setting transfers between the two
    programs without the reader having to know that one of them counts from 1.
    """
    x = grid()
    y = 4.0e4 / x**1.6 + 600.0 + gaussian(x, 12.4, 9000.0, 0.15)
    assert np.allclose(
        sonneveld_visser_baseline(x, y, granularity=0),
        sonneveld_visser_baseline(x, y, granularity=1),
    )


def test_the_bending_factor_bites_only_at_the_top_of_its_range():
    """Why the slider needed to run to 100, and what an earlier note got wrong.

    I reported the bending factor as inert on the strength of a 0-to-4 span,
    which was the span I had given it.  Over HighScore's actual 0 to 100 it
    moves a baseline by several per cent of the intensity range - on real
    patterns 2.3 to 12.4 % - so it is weak at the bottom rather than inert, and
    a reader who only ever saw 0 to 4 would conclude the parameter did nothing.
    """
    x = grid()
    y = (4.0e4 / x**1.6 + 400.0 + gaussian(x, 6.5, 30000.0, 5.0)
         + gaussian(x, 22.0, 6000.0, 14.0) + gaussian(x, 26.7, 12000.0, 0.15))
    span = float(np.ptp(y))
    zero = sonneveld_visser_baseline(x, y, granularity=20, bending=0.0)

    def moved(bending):
        other = sonneveld_visser_baseline(x, y, granularity=20, bending=bending)
        return float(np.max(np.abs(other - zero))) / span

    assert moved(4.0) < 0.02, "the bottom of the range barely moves it"
    assert moved(100.0) > 3.0 * moved(4.0), "the top of the range does"


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


# --- as a background component in its own right ---------------------------


def test_it_is_a_background_in_itself_rather_than_something_fitted():
    """HighScore's arrangement, and the method's own: the curve is the background.

    Fitting a polynomial to a Sonneveld-Visser estimate is possible - the
    estimator dispatch still allows it, and Sec. A.19 uses it for comparison -
    but it is not what the method is for, and it was the wrong thing to put in
    front of an operator.
    """
    x = grid()
    y = 4.0e4 / x**1.6 + 600.0 + gaussian(x, 8.8, 9000.0, 0.2)
    background = StrippedBackground.fit(x, y, granularity=20, bending=1.0)
    assert background.estimator == "sonneveld-visser"
    assert background.n_terms == 0, "nothing is fitted, so there are no free terms"
    assert "granularity 20" in background.model and "bending 1" in background.model
    # It behaves as any other background does.
    assert np.all(background(x) <= y + 1e-9)
    assert np.all(background.subtract(x, y) >= 0.0)


def test_granularity_and_bending_are_the_parameters_highscore_exposes():
    """Named and defaulted to match, so a setting means the same in both.

    Granularity is the number of points between samples - the paper's every
    twentieth point, HighScore's number of intervals, recommended there between
    15 and 30.  Bending 1 is anchored on the value the paper itself used.
    """
    assert SONNEVELD_VISSER_GRANULARITY == 20
    assert 15 <= SONNEVELD_VISSER_GRANULARITY <= 30
    assert SONNEVELD_VISSER_BENDING == 1.0
    x = grid()
    y = 4.0e4 / x**1.6 + 600.0 + gaussian(x, 12.4, 9000.0, 0.15)
    # The defaults are what the bare call uses.
    assert np.allclose(
        sonneveld_visser_baseline(x, y),
        sonneveld_visser_baseline(x, y, granularity=SONNEVELD_VISSER_GRANULARITY,
                                  bending=SONNEVELD_VISSER_BENDING),
    )


def test_granularity_is_counted_in_points_not_degrees():
    """Which is why the reach has to be reported: it moves with the step size."""
    fine = np.arange(3.0, 40.0, 0.005)
    coarse = np.arange(3.0, 40.0, 0.02)
    reaches = [sonneveld_visser_reach(20, float(np.mean(np.diff(g))), 30)
               for g in (fine, coarse)]
    assert reaches[1] == pytest.approx(4.0 * reaches[0], rel=0.05)


# --- reading the parameters off a measurement -----------------------------


def noisy(x, y, seed=0):
    rng = np.random.default_rng(seed)
    return y + rng.normal(0.0, np.sqrt(np.clip(y, 1.0, None)) * 0.5)


def test_peak_groups_are_the_stretches_above_the_noise():
    """The paper's Sec. 3.3, and the measurement the pre-screen is built on."""
    x = grid()
    y = gaussian(x, 12.0, 5000.0, 0.30) + gaussian(x, 25.0, 3000.0, 1.20)
    groups = peak_groups(x, noisy(x, y), threshold=200.0)
    assert len(groups) == 2
    first, second = sorted(groups, key=lambda group: group.centre)
    assert first.centre == pytest.approx(12.0, abs=0.1)
    assert first.width == pytest.approx(0.30, abs=0.06)
    assert second.centre == pytest.approx(25.0, abs=0.1)
    assert second.width == pytest.approx(1.20, abs=0.15)
    assert all(group.points >= 3 for group in groups)


def test_a_single_point_over_the_line_is_not_a_reflection():
    x = grid()
    difference = np.zeros_like(x)
    difference[500] = 10_000.0
    assert peak_groups(x, difference, threshold=100.0) == []


def test_the_noise_level_survives_broad_contamination():
    """The defect this fixes made the pre-screen blind to what it is for.

    Sonneveld & Visser take sigma over everything, peaks included, then reject
    above mu + 3 sigma.  On a pattern carrying a broad feature - an amorphous
    hump, a smectite band - the feature is a fifth of the points, sigma comes
    out inflated, the first threshold lands above every point, nothing is
    rejected, and sigma not moving reads as convergence.  The level returned is
    then the spread of the feature, and a feature that is not detected cannot
    set a granularity.  Seeding from the median and the MAD engages the paper's
    own loop instead of stalling it.

    It is left somewhat conservative rather than made exact: with a broad
    feature present its flanks stay within three sigma of the median over a long
    stretch and survive the clipping, so the level comes back about a third high
    - 24 against 18 here, where the unseeded version gave 2900.  A threshold
    slightly too high is slightly less sensitive, which is the safe direction
    for deciding what counts as a reflection.
    """
    x = grid()
    truth = 18.0
    rng = np.random.default_rng(4)
    difference = rng.normal(0.0, truth, x.size) + gaussian(x, 20.0, 9000.0, 8.0)
    level = noise_level(difference)
    assert truth <= level.sigma < 1.5 * truth
    # And the feature is then found, which is the point of measuring the noise.
    groups = peak_groups(x, difference, level.threshold)
    assert max(group.width for group in groups) == pytest.approx(8.0, rel=0.25)


def test_the_suggestion_clears_the_widest_reflection():
    x = grid()
    for width, expected in ((0.2, 4.0), (2.0, 4.0), (4.0, 8.0)):
        y = (4.0e4 / x**1.6 + 400.0 + gaussian(x, 20.0, 9000.0, width)
             + gaussian(x, 26.7, 12000.0, 0.15))
        suggestion = suggest_sonneveld_visser(x, noisy(x, y))
        assert suggestion.widest_reflection == pytest.approx(width, rel=0.2)
        assert suggestion.reach == pytest.approx(expected, rel=0.15)
        # Twice the width, which is what the reach law says keeps all of it.
        assert suggestion.reach >= 2.0 * suggestion.widest_reflection - 0.3


def test_a_pattern_of_sharp_peaks_falls_back_to_the_floor():
    """Not evidence that nothing broad is present, so it errs the safe way."""
    x = grid()
    y = 4.0e4 / x**1.6 + 400.0 + sum(
        gaussian(x, centre, 9000.0, 0.15) for centre in (8.8, 12.4, 19.9, 26.7)
    )
    suggestion = suggest_sonneveld_visser(x, noisy(x, y))
    assert suggestion.floored is True
    assert suggestion.reach == pytest.approx(SONNEVELD_VISSER_FLOOR, rel=0.1)
    assert "floor" in suggestion.note


def test_a_feature_too_broad_to_judge_is_capped_and_says_so():
    x = grid()
    y = 4.0e4 / x**1.6 + 400.0 + gaussian(x, 20.0, 9000.0, 8.0)
    suggestion = suggest_sonneveld_visser(x, noisy(x, y))
    assert suggestion.capped is True
    assert suggestion.reach <= 0.25 * float(x[-1] - x[0]) + 0.5
    assert "yours" in suggestion.note


def test_the_suggested_bending_is_zero_and_the_note_says_why():
    """Measured, not chosen: all eight real mounts fitted best at 0."""
    x = grid()
    y = 4.0e4 / x**1.6 + 400.0 + gaussian(x, 12.4, 9000.0, 0.2)
    suggestion = suggest_sonneveld_visser(x, noisy(x, y))
    assert suggestion.bending == 0.0
    assert "eight real mounts" in suggestion.note


def test_the_suggestion_is_usable_as_it_stands():
    x = grid()
    y = 4.0e4 / x**1.6 + 400.0 + gaussian(x, 8.8, 9000.0, 0.5)
    suggestion = suggest_sonneveld_visser(x, noisy(x, y))
    background = StrippedBackground.fit(
        x, y, granularity=suggestion.granularity, bending=suggestion.bending
    )
    assert np.all(background(x) <= y + 1e-9)
    # The reflection it was told to keep is still there afterwards.
    kept = background.subtract(x, y)
    assert kept[np.argmin(np.abs(x - 8.8))] > 0.9 * 9000.0


def test_too_little_data_to_read_a_suggestion_from_is_refused():
    with pytest.raises(ValueError):
        suggest_sonneveld_visser(np.arange(5.0), np.ones(5))
    with pytest.raises(ValueError):
        suggest_sonneveld_visser(grid(), np.ones(10))
