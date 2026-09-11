"""Geometric intensity corrections: Lorentz-polarization and preferred orientation.

Lorentz-polarization
--------------------
For a randomly oriented (Debye-Scherrer) powder in a diffractometer,

    LP_powder(theta) = (1 + cos^2(2 theta)) / (2 sin^2(theta) cos(theta))

while a one-dimensionally periodic, perfectly oriented specimen would call for
the single-crystal form

    LP_crystal(theta) = (1 + cos^2(2 theta)) / (2 sin(2 theta)) .

Which one applies to an *oriented clay aggregate* on a glass slide is a genuine
question in the clay literature.  Bradley (1954) and Schoen (1962) found the
random-powder factor to work, MacEwan, Amil & Brown (1961) argued for something
between the two, and Reynolds (1965, Tables 2 and 4) showed for both an
ethylene-glycol montmorillonite complex and a dry Na-montmorillonite that the
random-powder factor is required to reproduce observed basal intensities; the
single-crystal factor gave large deviations.  The default here is therefore
``"powder"``, which is also the convention in which the published
one-dimensional clay layer models were refined.  Note that the two conventions
differ, after normalising at 001, by a factor ``l`` across a basal series, so
mixing them corrupts relative basal intensities.

Preferred orientation
---------------------
The March-Dollase model (March 1932; Dollase 1986) describes a platy texture by
one parameter ``r``:

    P(alpha; r) = (r^2 cos^2(alpha) + sin^2(alpha) / r)^(-3/2)

where ``alpha`` is the angle between the reflection vector and the texture axis
(here ``c*``, the platelet normal, which in Bragg-Brentano reflection geometry
coincides with the specimen normal).  ``r = 1`` is a random powder; ``r < 1``
describes platelets lying flat, enhancing 00l and suppressing hk0.  The
function conserves total scattered intensity: its average over the sphere is 1.
"""

from __future__ import annotations

import numpy as np

__all__ = ["lorentz_polarization", "march_dollase", "LP_MODES"]

LP_MODES = ("powder", "crystal", "none")


def lorentz_polarization(
    two_theta: np.ndarray,
    mode: str = "powder",
    monochromator_two_theta: float | None = None,
) -> np.ndarray:
    """Lorentz-polarization factor for ``two_theta`` in degrees.

    Parameters
    ----------
    mode:
        ``"powder"`` for the random-powder (Debye-Scherrer) Lorentz factor,
        which is the form validated for oriented clay aggregates by Reynolds
        (1965); ``"crystal"`` for the single-crystal form; ``"none"`` to apply
        the polarization factor only.
    monochromator_two_theta:
        If given, the polarization factor becomes
        ``(1 + cos^2(2 theta) cos^2(2 theta_M)) / (1 + cos^2(2 theta_M))``,
        appropriate for a crystal-monochromated beam.
    """
    if mode not in LP_MODES:
        raise ValueError(f"unknown Lorentz-polarization mode {mode!r}; expected one of {LP_MODES}")
    tt = np.radians(np.asarray(two_theta, dtype=float))
    theta = tt / 2.0

    if monochromator_two_theta is None:
        polarization = 1.0 + np.cos(tt) ** 2
    else:
        cos2m = np.cos(np.radians(monochromator_two_theta)) ** 2
        polarization = (1.0 + np.cos(tt) ** 2 * cos2m) / (1.0 + cos2m) * 2.0

    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "powder":
            lorentz = 1.0 / (2.0 * np.sin(theta) ** 2 * np.cos(theta))
        elif mode == "crystal":
            lorentz = 1.0 / (2.0 * np.sin(tt))
        else:
            lorentz = np.full(tt.shape, 0.5)
        result = polarization * lorentz
    return np.where(np.isfinite(result), result, 0.0)


def march_dollase(alpha: np.ndarray, r: float) -> np.ndarray:
    """March-Dollase preferred-orientation factor.

    Parameters
    ----------
    alpha:
        Angle in radians between each reflection vector and the texture axis.
    r:
        March-Dollase parameter; ``1`` is a random powder and values below 1
        describe platelets lying in the specimen plane (00l enhanced).
    """
    if r <= 0.0:
        raise ValueError("the March-Dollase parameter must be positive")
    alpha = np.asarray(alpha, dtype=float)
    return (r**2 * np.cos(alpha) ** 2 + np.sin(alpha) ** 2 / r) ** -1.5
