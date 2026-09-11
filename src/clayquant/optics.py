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

Beam overflow at low angles
---------------------------
The Lorentz factor diverges as ``1/sin^2(theta)``, but a real measurement does
not, because at low angles the irradiated length of a flat specimen,

    L_irr(theta) = R * gamma / sin(theta)

(``R`` the goniometer radius, ``gamma`` the equatorial divergence in radians),
grows beyond the specimen itself and part of the beam misses it.  Only the
fraction that lands on the specimen is diffracted, so the intensity carries

    phi(theta) = min(1, L_specimen * sin(theta) / (R * gamma))

This matters precisely where clay basal reflections live.  Reynolds (1965)
avoided the correction by changing beam slits with angle so that the beam never
exceeded his specimen (following Klug & Alexander 1954); with fixed slits it has
to be applied, otherwise every calculated pattern carries an unphysical
low-angle ramp.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["lorentz_polarization", "march_dollase", "Divergence", "LP_MODES"]

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


@dataclass(frozen=True)
class Divergence:
    """Beam overflow correction for a flat specimen in Bragg-Brentano geometry.

    Attributes
    ----------
    specimen_length:
        Length of the specimen along the beam in mm.
    goniometer_radius:
        Goniometer radius in mm (280 mm on a Bruker D8, 240 mm on a D2).
    divergence:
        Equatorial divergence of the incident beam in degrees, i.e. the
        divergence slit setting.
    """

    specimen_length: float = 20.0
    goniometer_radius: float = 280.0
    divergence: float = 0.5

    def __post_init__(self) -> None:
        if self.specimen_length <= 0 or self.goniometer_radius <= 0 or self.divergence <= 0:
            raise ValueError("specimen length, goniometer radius and divergence must be positive")

    @property
    def full_illumination_two_theta(self) -> float:
        """2theta in degrees above which the beam is fully intercepted."""
        ratio = self.goniometer_radius * np.radians(self.divergence) / self.specimen_length
        if ratio >= 1.0:
            return 180.0
        return float(np.degrees(2.0 * np.arcsin(ratio)))

    def factor(self, two_theta: np.ndarray) -> np.ndarray:
        """The fraction of the beam intercepted by the specimen."""
        theta = np.radians(np.asarray(two_theta, dtype=float)) / 2.0
        irradiated = self.goniometer_radius * np.radians(self.divergence)
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = self.specimen_length * np.sin(theta) / irradiated
        return np.clip(np.nan_to_num(fraction), 0.0, 1.0)


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
