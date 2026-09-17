"""Taking the peak width model from the measurement instead of from a default.

The width sets the calculated peak height, so a width model that is wrong by ten
per cent leaves the fit ten per cent short at every strong peak at once.  These
tests pin the two things that made the naive versions of this unusable: a width
must not be read off an overlapped pair, and the model must not be dominated by
the broad peaks of one phase when what is wanted is the instrument.
"""

import numpy as np
import pytest

from clayquant.profile import (
    PeakShape,
    fit_peak_shape,
    measure_peak_widths,
    pseudo_voigt,
)

WAVELENGTH = 1.540596
STEP = 0.0167


def grid(low=3.0, high=40.0):
    return np.arange(low, high, STEP)


def peaks_at(x, centres, widths, heights=None):
    heights = heights or [1.0] * len(centres)
    y = np.zeros_like(x)
    for centre, width, height in zip(centres, widths, heights):
        profile = pseudo_voigt(x - centre, width, 0.5)
        y += height * profile / profile.max()
    return y


def test_a_single_peak_gives_back_its_own_width():
    x = grid()
    y = peaks_at(x, [20.0], [0.12])
    positions, widths = measure_peak_widths(x, y)
    assert positions.size == 1
    assert widths[0] == pytest.approx(0.12, abs=0.01)


def test_two_separated_peaks_are_measured_as_two():
    x = grid()
    y = peaks_at(x, [20.0, 21.0], [0.12, 0.12])
    positions, widths = measure_peak_widths(x, y)
    assert positions.size == 2
    assert all(width == pytest.approx(0.12, abs=0.02) for width in widths)


def test_a_pair_closer_than_the_separation_gives_at_most_one_width():
    # "Isolated" means the strongest thing within `separation` degrees, so a
    # neighbour 0.30 deg away disqualifies one of the pair even though the two
    # are resolved.  That is deliberate: the flank the width is read from is the
    # neighbour's flank as much as its own.
    x = grid()
    y = peaks_at(x, [20.0, 20.30], [0.12, 0.12])
    positions, widths = measure_peak_widths(x, y)
    assert positions.size <= 1
    positions, widths = measure_peak_widths(x, y, separation=0.15)
    assert positions.size == 2


def test_a_merged_pair_reads_too_broad_and_the_envelope_is_what_saves_it():
    # 0.10 deg apart with 0.12 deg widths there is no minimum and no rise on
    # either flank: the pair is one maximum and nothing in the profile can say
    # it is two, so its width comes out near the width of the pair.  This is why
    # fit_peak_shape takes the narrowest peaks rather than all of them - the
    # unresolvable case is admitted and then outvoted.
    x = grid()
    merged = peaks_at(x, [20.0, 20.10], [0.12, 0.12])
    _, widths = measure_peak_widths(x, merged)
    assert max(widths) > 0.15

    y = merged + peaks_at(x, [10.0, 15.0, 26.0, 31.0], [0.12] * 4)
    found = fit_peak_shape(x, y, WAVELENGTH,
                           default=PeakShape(u=0.0, v=0.0, w=0.12**2, eta=0.5))
    assert float(found.shape.fwhm(np.array([26.0]), WAVELENGTH)[0]) == pytest.approx(
        0.12, abs=0.02
    )


def test_a_shoulder_does_not_widen_its_neighbour():
    x = grid()
    y = peaks_at(x, [20.0, 20.45], [0.12, 0.12], [1.0, 0.35])
    positions, widths = measure_peak_widths(x, y)
    for position, width in zip(positions, widths):
        if abs(position - 20.0) < 0.05:
            assert width == pytest.approx(0.12, abs=0.02)


def test_a_one_point_spike_is_not_a_peak():
    x = grid()
    y = peaks_at(x, [20.0], [0.12])
    y[int(round((30.0 - x[0]) / STEP))] = 5.0
    positions, _ = measure_peak_widths(x, y)
    assert not any(abs(position - 30.0) < 0.05 for position in positions)


def test_noise_on_a_flank_does_not_stop_the_walk_to_half_maximum():
    rng = np.random.default_rng(7)
    x = grid()
    y = peaks_at(x, [12.0, 20.0, 26.0, 31.0], [0.12] * 4)
    y = y + rng.normal(0.0, 0.004, x.size)
    positions, widths = measure_peak_widths(x, np.clip(y, 0.0, None))
    assert positions.size >= 4
    assert np.median(widths) == pytest.approx(0.12, abs=0.02)


def test_the_width_model_is_scaled_to_the_measured_widths():
    x = grid()
    y = peaks_at(x, [10.0, 15.0, 20.0, 26.0, 31.0], [0.14] * 5)
    default = PeakShape(u=0.0, v=0.0, w=0.07**2, eta=0.5)
    found = fit_peak_shape(x, y, WAVELENGTH, default=default)
    assert float(found.shape.fwhm(np.array([20.0]), WAVELENGTH)[0]) == pytest.approx(
        0.14, abs=0.01
    )
    assert "scaled by" in found.note


def test_the_broad_peaks_of_one_phase_do_not_set_the_instrument_width():
    # Three sharp reflections of an accompanying mineral and four broad clay
    # basal peaks.  The instrumental width is the sharp one; the clay broadening
    # belongs to the library's own thickness and CSDS axes, and counting it here
    # too would apply it twice.
    x = grid()
    y = peaks_at(x, [20.9, 26.6, 36.5], [0.09] * 3, [1.0, 1.0, 0.5])
    y = y + peaks_at(x, [6.2, 12.5, 18.8, 25.2], [0.30] * 4, [0.6] * 4)
    found = fit_peak_shape(x, y, WAVELENGTH,
                           default=PeakShape(u=0.0, v=0.0, w=0.10**2, eta=0.5))
    width = float(found.shape.fwhm(np.array([26.6]), WAVELENGTH)[0])
    assert width == pytest.approx(0.09, abs=0.02)


def test_too_few_peaks_keeps_the_default_and_says_so():
    x = grid()
    y = peaks_at(x, [20.0], [0.12])
    default = PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6)
    found = fit_peak_shape(x, y, WAVELENGTH, default=default)
    assert found.shape == default
    assert "fewer than" in found.note
    assert np.isnan(found.residual)


def test_a_flat_pattern_keeps_the_default_rather_than_failing():
    x = grid()
    default = PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.6)
    found = fit_peak_shape(x, np.zeros_like(x), WAVELENGTH, default=default)
    assert found.shape == default


def test_the_angular_trend_of_the_default_is_kept_not_refitted():
    # Only the size of the width is estimated; a default that widens with angle
    # still widens with angle afterwards, in the same proportion.
    x = grid()
    y = peaks_at(x, [10.0, 15.0, 20.0, 26.0, 31.0], [0.14] * 5)
    default = PeakShape(u=0.02, v=-0.005, w=0.01, eta=0.5)
    found = fit_peak_shape(x, y, WAVELENGTH, default=default)
    angles = np.array([8.0, 20.0, 35.0])
    before = default.fwhm(angles, WAVELENGTH)
    after = found.shape.fwhm(angles, WAVELENGTH)
    assert np.allclose(after / before, (after / before)[0], rtol=1e-6)


def test_a_size_broadened_default_is_scaled_through_its_size():
    # size_ab enters the width as an inverse, so stretching the width means
    # shrinking the size; scaling u, v and w alone would leave the size term
    # untouched and the scaling incomplete.
    x = grid()
    y = peaks_at(x, [10.0, 15.0, 20.0, 26.0, 31.0], [0.20] * 5)
    default = PeakShape(u=0.0, v=0.0, w=0.001, eta=0.5, size_ab=400.0)
    found = fit_peak_shape(x, y, WAVELENGTH, default=default)
    assert found.shape.size_ab is not None
    assert found.shape.size_ab < 400.0
    assert float(found.shape.fwhm(np.array([20.0]), WAVELENGTH)[0]) == pytest.approx(
        0.20, abs=0.02
    )


def test_a_lorentzian_pattern_is_recognised_as_one():
    x = grid()
    y = np.zeros_like(x)
    for centre in (10.0, 15.0, 20.0, 26.0, 31.0):
        profile = pseudo_voigt(x - centre, 0.14, 1.0)
        y += profile / profile.max()
    found = fit_peak_shape(x, y, WAVELENGTH,
                           default=PeakShape(u=0.0, v=0.0, w=0.14**2, eta=0.3))
    assert found.shape.eta >= 0.9


def test_a_gaussian_pattern_is_not_called_lorentzian():
    x = grid()
    y = np.zeros_like(x)
    for centre in (10.0, 15.0, 20.0, 26.0, 31.0):
        profile = pseudo_voigt(x - centre, 0.14, 0.0)
        y += profile / profile.max()
    found = fit_peak_shape(x, y, WAVELENGTH,
                           default=PeakShape(u=0.0, v=0.0, w=0.14**2, eta=0.9),
                           etas=(0.0, 0.3, 0.6, 0.9))
    assert found.shape.eta <= 0.3


def test_the_residual_reports_how_well_the_one_scale_fits():
    x = grid()
    y = peaks_at(x, [10.0, 15.0, 20.0, 26.0, 31.0], [0.14] * 5)
    found = fit_peak_shape(x, y, WAVELENGTH,
                           default=PeakShape(u=0.0, v=0.0, w=0.07**2, eta=0.5))
    assert found.residual < 0.02
