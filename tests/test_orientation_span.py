"""The orientation axis spans textures a clay mount can have, and stops there.

March-Dollase ``r`` is the width of an orientation distribution, not a free
shape parameter, and a library that offers values below what a settled film
achieves is offering the fit a texture no specimen has.  It will take it: a
basal series is enhanced as ``r ** -3``, so the bottom of the axis is where
low-angle intensity is cheapest in mass, and the mass is what comes out as a
weight percent.

Measured over twelve clay separates on a background that passes the pedestal
guard, with the axis reaching 0.1: nine of twelve put a clay on 0.1, and four
specimens that are clay separates came back at 4.5, 29.7, 33.0 and 39.9 per cent
clay by weight - the first of them against 95 per cent quartz.
"""

import numpy as np
import pytest

from clayquant.library import PREFERRED_ORIENTATIONS

# Moore & Reynolds describe an oriented mount by a Gaussian sigma*: about 12 deg
# for a good settled or smear mount, about 6 for an exceptionally good one.
BEST_REAL_MOUNT_SIGMA = 6.0
GAUSSIAN_FWHM = 2.355


def half_width(r: float) -> float:
    """FWHM in degrees of ``P(alpha; r) = (r^2 cos^2 a + sin^2 a / r)^(-3/2)``."""
    if r >= 1.0:
        return 180.0
    # P(a)/P(0) = [cos^2 a + sin^2 a / r^3] ^ (-3/2); half maximum at 2^(2/3).
    sine_squared = (2.0 ** (2.0 / 3.0) - 1.0) / (1.0 / r ** 3 - 1.0)
    if sine_squared >= 1.0:
        # Above about r = 0.85 the distribution never falls to half its pole
        # value anywhere in the quadrant: it is as good as random.
        return 180.0
    return 2.0 * float(np.degrees(np.arcsin(np.sqrt(sine_squared))))


def test_the_axis_does_not_reach_below_what_a_clay_film_can_be():
    """Nothing in the library may describe a mount better oriented than real ones."""
    sharpest = min(PREFERRED_ORIENTATIONS)
    sigma = half_width(sharpest) / GAUSSIAN_FWHM
    assert sigma >= BEST_REAL_MOUNT_SIGMA - 0.5, (
        f"the axis reaches r = {sharpest}, an orientation distribution "
        f"{half_width(sharpest):.1f} deg wide (sigma* {sigma:.1f} deg). An "
        f"exceptionally well oriented clay film is about {BEST_REAL_MOUNT_SIGMA} "
        f"deg, so this describes a mount nobody can prepare - and a fit will use "
        f"it, because r**-3 makes it the cheapest intensity in the library."
    )


def test_the_axis_still_reaches_a_random_powder():
    """The other end is physical too, and has to stay: it is the end of what exists."""
    assert max(PREFERRED_ORIENTATIONS) == pytest.approx(1.0)


def test_the_axis_covers_an_ordinary_mount():
    """A good settled mount is sigma* about 12 deg, and must be inside the span."""
    ordinary = [r for r in PREFERRED_ORIENTATIONS
                if 8.0 <= half_width(r) / GAUSSIAN_FWHM <= 20.0]
    assert len(ordinary) >= 2, (
        "the axis has to hold more than one value in the range an ordinary mount "
        f"occupies; it holds {ordinary}"
    )


def test_the_half_width_formula_is_the_march_dollase_one():
    """Pins the mapping the docstring's table was computed from."""
    assert half_width(0.1) == pytest.approx(2.78, abs=0.05)
    assert half_width(0.3) == pytest.approx(14.67, abs=0.05)
    assert half_width(0.5) == pytest.approx(33.68, abs=0.05)
