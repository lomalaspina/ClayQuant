"""The width anchor must fit the quartz 101, not whatever is tallest near it.

On a mica-bearing specimen the illite/mica 003 at 3.31 A sits 0.25 deg above the
quartz 101 and can stand taller than it.  The search window has to be wide - 0.45
deg where no zero error has been applied - so that line is inside it, and taking
the tallest point in the window hands the doublet fit a mica reflection.  The
mica 003 is twice as broad as the quartz 101, and a library built at twice the
true width is a library whose every strong peak comes out half as tall as it
should be with the right area underneath, which reads as a set of missing phases
and is not one.  On one real metabasite separate the mica stood at 6149 counts
against the quartz 101's 5958, and the width came back 0.101 deg instead of
0.046.
"""

import numpy as np
import pytest

from clayquant.calibration import QUARTZ_101_D, reference_two_theta
from clayquant.profile import (
    KALPHA2_INTENSITY_RATIO,
    QUARTZ_CENTRE_TOLERANCE,
    kalpha2_two_theta,
    pseudo_voigt,
    quartz_line_width,
)

WAVELENGTH = 1.540596
QUARTZ_101 = reference_two_theta(QUARTZ_101_D, WAVELENGTH)


def a_scan(lines):
    """``lines`` is a sequence of (centre, height, fwhm) with the doublet on each."""
    x = np.arange(24.0, 29.0, 0.0167)
    y = np.zeros_like(x)
    for centre, height, width in lines:
        profile = pseudo_voigt(x - centre, width, 0.4)
        peak = height * profile / profile.max()
        partner = pseudo_voigt(x - kalpha2_two_theta(centre), width, 0.4)
        peak = peak + KALPHA2_INTENSITY_RATIO * height * partner / partner.max()
        y += peak
    return x, y


def test_the_taller_neighbour_does_not_capture_the_anchor():
    # the quartz 101, narrow, with a broader and *taller* mica 003 0.27 deg above
    x, y = a_scan([(QUARTZ_101, 5958.0, 0.050), (QUARTZ_101 + 0.27, 6149.0, 0.110)])
    line = quartz_line_width(x, y, reference=QUARTZ_101, shift=0.0)
    assert line is not None
    assert line.centre == pytest.approx(QUARTZ_101, abs=0.05)
    # and so the width is quartz's, not the mica's
    assert line.fwhm < 0.08


def test_the_quartz_is_still_found_when_it_is_the_taller_one():
    x, y = a_scan([(QUARTZ_101, 6000.0, 0.050), (QUARTZ_101 + 0.27, 1200.0, 0.110)])
    line = quartz_line_width(x, y, reference=QUARTZ_101, shift=0.0)
    assert line is not None
    assert line.centre == pytest.approx(QUARTZ_101, abs=0.05)


def test_a_line_too_far_from_nominal_is_refused_rather_than_returned():
    # nothing at the quartz position at all, only a mineral well outside tolerance
    offset = 2.5 * QUARTZ_CENTRE_TOLERANCE
    x, y = a_scan([(QUARTZ_101 + offset, 6000.0, 0.110)])
    assert quartz_line_width(x, y, reference=QUARTZ_101, shift=0.0) is None


def test_a_real_zero_error_inside_the_tolerance_still_passes():
    offset = 0.5 * QUARTZ_CENTRE_TOLERANCE
    x, y = a_scan([(QUARTZ_101 + offset, 6000.0, 0.060)])
    line = quartz_line_width(x, y, reference=QUARTZ_101, shift=0.0)
    assert line is not None
    assert line.centre == pytest.approx(QUARTZ_101 + offset, abs=0.05)
