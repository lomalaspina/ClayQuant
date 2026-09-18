"""The Clayfit background models.

The point of these tests is fidelity: the five models taken from Clayfit have
to give the same curve here as there, so where an algorithm is short enough to
transcribe it is transcribed literally from Clayfit's own source and the two are
compared point for point.  Where it is not - the percentile model - the numbers
checked are the ones Clayfit's own test suite checks.

The second thing they establish is that the two polynomial models are *not* the
same, which is what distinguishes this implementation from the one it replaced.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from clayquant.background import (
    CLAYFIT_ANCHOR_RADIUS,
    CLAYFIT_MODELS,
    CLAYFIT_ORDER,
    CLAYFIT_ORDER_LIMITS,
    QPA_MODEL_NAME,
    ClayfitBackground,
    als_baseline_2d,
    background_anchors,
    chebyshev_baseline,
    clayfit_background,
    exponential_baseline,
    inverse_x_baseline,
    polynomial_baseline,
    qpa_percentile_baseline,
    rolling_ball,
    smooth_by_degrees,
)


def clay_pattern(points: int = 1400, step: float = 0.02, seed: int = 3):
    """A pattern with the two features that decide a background on a clay mount.

    A convex direct-beam tail at the start, which a background must keep, and
    sharp reflections standing on a flat continuum, which it must not.
    """
    x = 3.0 + step * np.arange(points)
    background = 120.0 + 9000.0 / x**1.6
    peaks = sum(
        height * np.exp(-0.5 * ((x - centre) / 0.07) ** 2)
        for centre, height in ((6.2, 4000.0), (8.8, 2500.0), (12.4, 6000.0),
                               (17.7, 900.0), (20.86, 3000.0), (26.64, 1800.0))
    )
    rng = np.random.default_rng(seed)
    total = background + peaks
    return x, total + rng.normal(0.0, np.sqrt(total)), background


# --------------------------------------------------------------------------
# The rolling ball
# --------------------------------------------------------------------------

def clayfit_rolling_ball(array: np.ndarray, radius: float) -> np.ndarray:
    """Clayfit's ``rolling_ball``, transcribed from ``utils.py``.

    Kept deliberately literal - the nested loops, the nearest-point validity
    test, the ``1e-9`` and ``1e-6`` tolerances - so that agreement with
    :func:`clayquant.background.rolling_ball` is evidence about the algorithm
    and not about a shared idea of it.
    """

    def lower_center(p1, p2, r, arr):
        (x1, y1), (x2, y2) = p1, p2
        q = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if q >= 2 * r - 1e-9 or q < 1e-9:
            return None
        y3 = (y1 + y2) / 2
        x3 = (x1 + x2) / 2
        offset = math.sqrt(r**2 - (q / 2) ** 2)
        dx = (y1 - y2) / q
        dy = (x2 - x1) / q
        cx1, cy1 = x3 + offset * dx, y3 + offset * dy
        cx2, cy2 = x3 - offset * dx, y3 - offset * dy

        def center_valid(cx, cy):
            index = int(np.argmin(np.abs(arr[:, 0] - cx)))
            return cy <= arr[index, 1]

        valid1, valid2 = center_valid(cx1, cy1), center_valid(cx2, cy2)
        if valid1 and valid2:
            return (cx1, cy1) if cy1 < cy2 else (cx2, cy2)
        if valid1:
            return cx1, cy1
        if valid2:
            return cx2, cy2
        return None

    def all_points_outside(center, r, arr, exclude_i, exclude_j):
        cx, cy = center
        for k in range(arr.shape[0]):
            if k in (exclude_i, exclude_j):
                continue
            dx = arr[k, 0] - cx
            dy = arr[k, 1] - cy
            if math.sqrt(dx * dx + dy * dy) < r - 1e-6:
                return False
        return True

    x_min, x_max = np.min(array[:, 0]), np.max(array[:, 0])
    y_min, y_max = np.min(array[:, 1]), np.max(array[:, 1])
    scale_x, scale_y = x_max - x_min, y_max - y_min
    scaled = np.empty_like(array)
    scaled[:, 0] = (array[:, 0] - x_min) / scale_x
    scaled[:, 1] = (array[:, 1] - y_min) / scale_y

    count = scaled.shape[0]
    found_x, found_y = [], []
    for i in range(count - 1):
        p1 = scaled[i]
        for j in range(i + 1, count):
            p2 = scaled[j]
            center = lower_center(p1, p2, radius, scaled)
            if center is None:
                continue
            if all_points_outside(center, radius, scaled, i, j):
                found_x.extend([p1[0], p2[0]])
                found_y.extend([p1[1], p2[1]])

    points = np.zeros((len(found_x), 2))
    points[:, 0] = np.array(found_x) * scale_x + x_min
    points[:, 1] = np.array(found_y) * scale_y + y_min
    return np.unique(points, axis=0)


@pytest.mark.parametrize("radius", [0.2, 0.5, 0.9])
def test_rolling_ball_agrees_with_clayfits_own_loops(radius):
    """Point for point, on the pattern shape the method has to handle."""
    x, y, _ = clay_pattern(points=400)
    x, y = x[::5], y[::5]
    mine = np.column_stack((x[rolling_ball(x, y, radius=radius)],
                            y[rolling_ball(x, y, radius=radius)]))
    theirs = clayfit_rolling_ball(np.column_stack((x, y)), radius)
    np.testing.assert_allclose(np.sort(mine, axis=0), np.sort(theirs, axis=0))


def test_the_ball_touches_the_envelope_and_not_the_peaks():
    x, y, truth = clay_pattern()
    anchor_x, anchor_y = background_anchors(x, y)
    assert 8 <= anchor_x.size <= 60
    # Every anchor is within a few counts of the true background, rather than up
    # on a reflection: the peaks here stand 900 to 6000 counts above it.
    interpolated = np.interp(anchor_x, x, truth)
    assert np.all(anchor_y - interpolated < 6.0 * np.sqrt(interpolated))
    # And they span the pattern, so a polynomial through them is interpolating
    # rather than extrapolating over most of the range.
    assert anchor_x[0] < x[0] + 2.0
    assert anchor_x[-1] > x[-1] - 2.0


def test_the_ball_keeps_the_low_angle_tail():
    """The one failure that destroys the measurement rather than the peaks."""
    x, y, truth = clay_pattern()
    anchor_x, anchor_y = background_anchors(x, y)
    start = anchor_x < x[0] + 1.5
    assert start.sum() >= 2
    # The tail falls from about 1050 to 700 counts over the first degree and a
    # half; an envelope that cut under it would hold the anchors near 120.
    assert anchor_y[start].max() > 600.0


def test_a_smaller_ball_finds_more_points():
    x, y, _ = clay_pattern()
    wide, _ = background_anchors(x, y, radius=0.9)
    narrow, _ = background_anchors(x, y, radius=0.1)
    assert narrow.size > wide.size


def test_a_flat_pattern_has_no_envelope_to_find():
    x = np.linspace(3.0, 40.0, 50)
    contacts = rolling_ball(x, np.full_like(x, 500.0))
    assert contacts.size == x.size


def test_the_ball_rejects_a_nonpositive_radius():
    x, y, _ = clay_pattern(points=60)
    with pytest.raises(ValueError, match="radius"):
        rolling_ball(x, y, radius=0.0)


def test_anchors_are_cached_on_the_pattern():
    x, y, _ = clay_pattern(points=300)
    first = background_anchors(x, y)
    second = background_anchors(x.copy(), y.copy())
    np.testing.assert_array_equal(first[0], second[0])
    # A different stride is a different question and not answered from cache.
    other = background_anchors(x, y, stride=3)
    assert other[0].size != first[0].size or not np.array_equal(other[0], first[0])


# --------------------------------------------------------------------------
# Polynomial, Chebyshev, exponential
# --------------------------------------------------------------------------

def test_the_polynomial_is_polyfit_on_the_anchors_in_raw_two_theta():
    x, y, _ = clay_pattern()
    anchor_x, anchor_y = background_anchors(x, y)
    expected = np.polyval(np.polyfit(anchor_x, anchor_y, 6), x)
    np.testing.assert_allclose(polynomial_baseline(x, anchor_x, anchor_y, order=6), expected)


def test_the_polynomial_refuses_an_order_the_anchors_cannot_support():
    x, y, _ = clay_pattern(points=200)
    anchor_x, anchor_y = background_anchors(x, y)
    with pytest.raises(ValueError, match="anchor points"):
        polynomial_baseline(x, anchor_x, anchor_y, order=anchor_x.size + 1)


def test_the_chebyshev_amplitude_is_the_answer_curve_fit_converges_to():
    """The closed form is used for speed; it has to be the same number."""
    from scipy.optimize import curve_fit

    x, y, _ = clay_pattern()
    anchor_x, anchor_y = background_anchors(x, y)
    order = 6
    shape = np.polynomial.chebyshev.Chebyshev.fit(x, y / x, order)
    amplitude, _ = curve_fit(lambda angle, a: a * shape(angle), anchor_x, anchor_y)
    np.testing.assert_allclose(
        chebyshev_baseline(x, y, anchor_x, anchor_y, order=order),
        (lambda angle, a: a * shape(angle))(x, *amplitude),
        rtol=1e-6,
    )


def test_the_chebyshev_shape_is_fitted_to_the_whole_pattern():
    """Including the peaks - which is why it is not the polynomial.

    Remove the reflections and the shape changes; a model fitted only to the
    anchor points could not notice, because the anchors are unchanged.
    """
    x, y, truth = clay_pattern()
    anchor_x, anchor_y = background_anchors(x, y)
    with_peaks = chebyshev_baseline(x, y, anchor_x, anchor_y, order=6)
    without = chebyshev_baseline(x, truth, anchor_x, anchor_y, order=6)
    assert np.abs(with_peaks - without).max() > 20.0


def test_polynomial_and_chebyshev_are_not_the_same_curve():
    """The claim an earlier version of this program got wrong.

    Under an unconstrained least-squares fit to one target the two bases span
    the same space and give the same curve.  These are not that: only the
    polynomial is fitted to the anchor points, and only the Chebyshev model sees
    the peaks.  They differ by hundreds of counts at every order.
    """
    x, y, _ = clay_pattern()
    for order in range(*(CLAYFIT_ORDER_LIMITS[0], CLAYFIT_ORDER_LIMITS[1] + 1)):
        polynomial = clayfit_background(x, y, kind="polynomial", order=order)(x)
        chebyshev = clayfit_background(x, y, kind="chebyshev", order=order)(x)
        assert np.abs(polynomial - chebyshev).max() > 50.0, order


def test_the_order_moves_both_curves():
    """The other half of it: the order is not an inert control."""
    x, y, _ = clay_pattern()
    for kind in ("polynomial", "chebyshev"):
        curves = [clayfit_background(x, y, kind=kind, order=order)(x)
                  for order in range(4, 13)]
        moves = [np.abs(later - earlier).max()
                 for earlier, later in zip(curves, curves[1:])]
        assert max(moves) > 10.0, kind
        assert sum(move > 1.0 for move in moves) >= 4, kind


def test_the_exponential_recovers_a_known_tail():
    x = np.linspace(3.0, 40.0, 120)
    truth = 1800.0 * np.exp(-0.25 * x) + 140.0
    recovered = exponential_baseline(x, x, truth)
    np.testing.assert_allclose(recovered, truth, rtol=1e-4)


def test_the_exponential_falls_back_on_the_polynomial_and_says_so():
    """Clayfit's own fallback, reported rather than silently substituted."""
    x = np.linspace(3.0, 40.0, 400)
    # An anchor set no decaying exponential can describe: rising, then flat.
    y = np.where(x < 20.0, 100.0 + 40.0 * x, 900.0)
    fit = clayfit_background(x, y, kind="exponential", order=4)
    if fit.note:
        assert "polynomial" in fit.note
        assert fit.order == 4
    else:
        # It converged after all, which is allowed; then it is an exponential.
        assert fit.kind == "exponential"


# --------------------------------------------------------------------------
# Asymmetric least squares
# --------------------------------------------------------------------------

def clayfit_baseline_als_2d(array, lam=1, p=0.001, niter=100):
    """Clayfit's ``baseline_als_2d``, transcribed from ``utils.py``."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    x = array[:, 0]
    y = array[:, 1]
    length = len(y)
    dx = np.diff(x)
    matrix = sparse.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2)).toarray()
    matrix = sparse.csr_matrix(matrix / np.mean(dx) ** 2)
    weights = np.ones(length)
    baseline = y
    for _ in range(niter):
        diagonal = sparse.spdiags(weights, 0, length, length)
        system = diagonal + lam * matrix @ matrix.T
        baseline = spsolve(system, weights * y)
        weights = p * (y > baseline) + (1 - p) * (y < baseline)
    return baseline


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_als_agrees_with_clayfits_own_implementation():
    x, y, _ = clay_pattern(points=500)
    np.testing.assert_allclose(
        als_baseline_2d(x, y),
        clayfit_baseline_als_2d(np.column_stack((x, y))),
        rtol=1e-6, atol=1e-6,
    )


def test_als_stays_under_the_data_and_cuts_the_low_angle_tail():
    """Its known behaviour on an oriented mount, and worth having on record."""
    x, y, truth = clay_pattern()
    baseline = als_baseline_2d(x, y)
    assert np.mean(baseline <= y) > 0.95
    # It asks for the smoothest curve under the data, and the direct-beam tail
    # is not smooth on that scale, so it passes below it.
    assert baseline[0] < 0.75 * truth[0]


@pytest.mark.parametrize("kwargs", [{"smoothness": 0.0}, {"asymmetry": 0.0},
                                    {"iterations": 0}])
def test_als_rejects_impossible_parameters(kwargs):
    x, y, _ = clay_pattern(points=60)
    with pytest.raises(ValueError):
        als_baseline_2d(x, y, **kwargs)


# --------------------------------------------------------------------------
# The percentile model
# --------------------------------------------------------------------------

def test_the_percentile_model_reproduces_clayfits_defaults_exactly():
    """The numbers Clayfit's own test suite asserts, on its own grid."""
    from scipy.ndimage import percentile_filter
    from scipy.signal import savgol_filter

    x = np.arange(3.0, 40.0, 0.02)
    y = (120.0 + 900.0 / x + 0.06 * (x - 20.0) ** 2
         + 900.0 * np.exp(-0.5 * ((x - 6.0) / 0.05) ** 2)
         + 1500.0 * np.exp(-0.5 * ((x - 20.8) / 0.06) ** 2))

    expected_baseline = percentile_filter(y, percentile=15, size=75, mode="nearest")
    expected_baseline = savgol_filter(expected_baseline, 15, 2)
    expected_corrected = np.clip(y - expected_baseline, 0.0, None)
    expected_smoothed = savgol_filter(expected_corrected, 5, 2)

    result = qpa_percentile_baseline(x, y)
    assert result.percentile_points == 75
    assert result.baseline_smooth_points == 15
    assert result.final_smooth_points == 5
    np.testing.assert_allclose(result.baseline, expected_baseline)
    np.testing.assert_allclose(result.corrected, expected_corrected)
    np.testing.assert_allclose(result.smoothed, expected_smoothed)


def test_the_percentile_windows_convert_to_clayfits_point_counts():
    x = np.arange(3.0, 40.0, 0.02)
    y = 200.0 + 1000.0 / x
    edited = qpa_percentile_baseline(
        x, y, window=0.70, baseline_smooth=0.16, baseline_order=3,
        final_smooth=0.16, final_order=3,
    )
    assert (edited.percentile_points, edited.baseline_smooth_points,
            edited.final_smooth_points) == (35, 9, 9)


def test_zero_windows_switch_the_smoothing_off():
    x = np.arange(3.0, 40.0, 0.02)
    y = 200.0 + 1000.0 / x
    result = qpa_percentile_baseline(x, y, baseline_smooth=0.0, final_smooth=0.0)
    assert result.baseline_smooth_points == 0
    assert result.final_smooth_points == 0
    np.testing.assert_allclose(result.smoothed, result.corrected)


@pytest.mark.parametrize("kwargs", [{"window": 0.0}, {"percentile": 101.0},
                                    {"final_order": -1}])
def test_the_percentile_model_rejects_impossible_parameters(kwargs):
    x = np.arange(3.0, 12.0, 0.02)
    y = 200.0 + 1000.0 / x
    with pytest.raises(ValueError):
        qpa_percentile_baseline(x, y, **kwargs)


def test_smoothing_by_degrees_needs_an_odd_window_wider_than_the_order():
    values = np.linspace(0.0, 1.0, 101)
    smoothed, points = smooth_by_degrees(values, 0.02, 0.10, 2)
    assert points == 5
    np.testing.assert_allclose(smoothed, values, atol=1e-9)
    with pytest.raises(ValueError, match="order"):
        smooth_by_degrees(values, 0.02, 0.10, -1)


# --------------------------------------------------------------------------
# The A/2theta term
# --------------------------------------------------------------------------

def test_the_inverse_term_is_an_amplitude_over_the_angle():
    np.testing.assert_allclose(inverse_x_baseline(np.array([2.0, 4.0, 8.0]), 400.0),
                               [200.0, 100.0, 50.0])


def test_the_inverse_term_rejects_a_nonpositive_angle_or_a_negative_amplitude():
    with pytest.raises(ValueError, match="positive"):
        inverse_x_baseline(np.array([0.0, 1.0]), 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        inverse_x_baseline(np.array([1.0, 2.0]), -1.0)


@pytest.mark.parametrize("kind", CLAYFIT_MODELS)
def test_the_inverse_term_accumulates_on_every_model(kind):
    x, y, _ = clay_pattern(points=500)
    plain = clayfit_background(x, y, kind=kind)(x)
    with_term = clayfit_background(x, y, kind=kind, inverse_x_amplitude=400.0)(x)
    np.testing.assert_allclose(with_term - plain, 400.0 / x, rtol=1e-9, atol=1e-9)


def test_the_inverse_amplitude_is_not_counted_as_a_fitted_term():
    x, y, _ = clay_pattern(points=500)
    without = clayfit_background(x, y, kind="polynomial", order=6)
    with_term = clayfit_background(x, y, kind="polynomial", order=6,
                                   inverse_x_amplitude=400.0)
    assert without.n_terms == with_term.n_terms == 7


# --------------------------------------------------------------------------
# The dispatcher and the fitted object
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", CLAYFIT_MODELS)
def test_every_model_returns_a_usable_background(kind):
    x, y, _ = clay_pattern(points=600)
    fit = clayfit_background(x, y, kind=kind)
    assert isinstance(fit, ClayfitBackground)
    assert fit.baseline.shape == x.shape
    assert np.all(np.isfinite(fit.baseline))
    # The interface the rest of the program uses.
    np.testing.assert_allclose(fit(x), fit.baseline)
    subtracted = fit.subtract(x, y)
    assert subtracted.shape == x.shape
    assert np.all(subtracted >= 0.0)
    assert fit.model
    assert fit.components


def test_the_models_disagree_with_one_another():
    """Which is why all six are drawn and the operator chooses."""
    x, y, _ = clay_pattern()
    curves = {kind: clayfit_background(x, y, kind=kind)(x) for kind in CLAYFIT_MODELS}
    at_start = {kind: curve[0] for kind, curve in curves.items()}
    assert max(at_start.values()) - min(at_start.values()) > 200.0


def test_only_the_percentile_model_smooths_the_corrected_pattern():
    x, y, _ = clay_pattern()
    for kind in CLAYFIT_MODELS:
        fit = clayfit_background(x, y, kind=kind)
        plain = np.clip(y - fit(x), 0.0, None)
        smoothed = fit.subtract(x, y)
        if kind == QPA_MODEL_NAME:
            assert not np.allclose(smoothed, plain)
            # Smoothing must not move a reflection, only its noise.
            assert abs(float(np.trapezoid(smoothed, x) - np.trapezoid(plain, x))) \
                < 0.01 * float(np.trapezoid(plain, x))
        else:
            np.testing.assert_allclose(smoothed, plain)


def test_the_number_of_fitted_terms_is_what_each_model_actually_fits():
    x, y, _ = clay_pattern(points=600)
    terms = {kind: clayfit_background(x, y, kind=kind, order=8).n_terms
             for kind in CLAYFIT_MODELS}
    assert terms["polynomial"] == 9
    # One amplitude, however high the order: the shape is fitted to the pattern.
    assert terms["chebyshev"] == 1
    assert terms["exponential"] == 3
    assert terms["als"] == terms[QPA_MODEL_NAME] == terms["sonneveld-visser"] == 0


@pytest.mark.parametrize("order", [0, 3, 13, 20])
def test_the_order_is_held_to_clayfits_range(order):
    x, y, _ = clay_pattern(points=300)
    with pytest.raises(ValueError, match="4 to 12"):
        clayfit_background(x, y, kind="polynomial", order=order)
    # The models with no order are not refused for one they do not use.
    clayfit_background(x, y, kind="als", order=order)


def test_an_unknown_model_is_named_in_the_error():
    x, y, _ = clay_pattern(points=60)
    with pytest.raises(ValueError, match="unknown background model"):
        clayfit_background(x, y, kind="peak-stripped")


def test_the_default_is_clayfits_default():
    x, y, _ = clay_pattern(points=300)
    assert clayfit_background(x, y).kind == "exponential"
    assert CLAYFIT_ORDER == 6
    assert CLAYFIT_ORDER_LIMITS == (4, 12)
    assert CLAYFIT_ANCHOR_RADIUS == 0.5


def test_the_r_squared_is_measured_against_the_anchors():
    x, y, _ = clay_pattern()
    polynomial = clayfit_background(x, y, kind="polynomial", order=6)
    assert 0.9 < polynomial.r_squared(x) <= 1.0
    # The Chebyshev model fits one amplitude to them and passes further away.
    chebyshev = clayfit_background(x, y, kind="chebyshev", order=6)
    assert chebyshev.r_squared(x) < polynomial.r_squared(x)
    # Nothing fitted, nothing to report - rather than a 1.0 that reads as
    # success.
    assert math.isnan(clayfit_background(x, y, kind="als").r_squared(x))


def test_a_mismatched_pattern_is_refused_rather_than_broadcast():
    x, y, _ = clay_pattern(points=100)
    with pytest.raises(ValueError, match="same shape"):
        clayfit_background(x, y[:-1])


# --------------------------------------------------------------------------
# The Background tab's controls
# --------------------------------------------------------------------------

def test_the_controls_describe_the_settings_the_models_take():
    gui = pytest.importorskip("clayquant.gui.app")
    settings = gui.background_settings_from_controls(
        "polynomial", 8, ["on"], 500.0, 0.5, 5,
        1.5, 0.30, 2, 0.08, 2, 20, 0.0,
    )
    assert settings["kind"] == "polynomial"
    assert settings["order"] == 8
    assert settings["inverse_x_amplitude"] == 500.0
    # Unticked means no term at all, not an amplitude of whatever is in the box.
    off = gui.background_settings_from_controls(
        "polynomial", 8, [], 500.0, 0.5, 5, 1.5, 0.30, 2, 0.08, 2, 20, 0.0,
    )
    assert off["inverse_x_amplitude"] == 0.0
    # And they are the arguments clayfit_background takes, not a subset of them.
    x, y, _ = clay_pattern(points=400)
    clayfit_background(x, y, **settings)


def test_the_tab_fits_what_the_controls_say():
    gui = pytest.importorskip("clayquant.gui.app")
    from clayquant.pattern import Pattern

    x, y, _ = clay_pattern(points=600)
    pattern = Pattern(two_theta=x, intensity=y, name="synthetic")
    for kind in CLAYFIT_MODELS:
        settings = gui.background_settings_from_controls(
            kind, 6, [], 500.0, 0.5, 5, 1.5, 0.30, 2, 0.08, 2, 20, 0.0,
        )
        fit = gui.fit_background_from_controls(pattern, **settings)
        assert fit.kind == kind


def test_the_status_line_names_the_model_and_the_number_to_judge_it_by():
    gui = pytest.importorskip("clayquant.gui.app")
    from clayquant.pattern import Pattern

    x, y, _ = clay_pattern()
    pattern = Pattern(two_theta=x, intensity=y, name="synthetic")
    fit = clayfit_background(x, y, kind="polynomial", order=6)
    report = gui.background_report(pattern, fit, fit(x))
    assert "Polynomial (order 6" in report
    assert "anchor points" in report
    assert "Between the peaks" in report
    assert "Rwp is not" in report
    # And for a model that fits nothing it does not invent an agreement.
    erosion = clayfit_background(x, y, kind="sonneveld-visser")
    assert "Nothing is fitted" in gui.background_report(pattern, erosion, erosion(x))
