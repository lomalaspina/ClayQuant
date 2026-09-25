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

import math
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
    shape: str = "rectangular"
    """``"rectangular"`` or ``"round"``.

    A round mount is not a rectangular one of the same length.  The beam lights
    a strip along the surface, and on a disc the strip's corners run off the
    edge before its middle does, so intensity is lost sooner than the length
    alone says.  The difference is small - about 3 per cent at the low-angle
    end of a clay scan - but the mounts this program was written for are discs,
    and a correction one knows the sign of is not worth leaving out.
    """

    beam_width: float = 10.0
    """Axial width of the beam at the specimen, in mm; the mask setting.

    Only a round mount uses it, and barely: widening it from 5 to 15 mm moves
    the low-angle factor by 6 per cent, because what limits the strip there is
    its length and not its width.
    """

    def __post_init__(self) -> None:
        if self.specimen_length <= 0 or self.goniometer_radius <= 0 or self.divergence <= 0:
            raise ValueError("specimen length, goniometer radius and divergence must be positive")
        if self.shape not in ("rectangular", "round"):
            raise ValueError(
                f"a specimen is 'rectangular' or 'round', not {self.shape!r}"
            )
        if self.beam_width <= 0:
            raise ValueError("the beam width must be positive")

    @property
    def full_illumination_two_theta(self) -> float:
        """2theta in degrees above which the beam is fully intercepted."""
        length = self.specimen_length
        if self.shape == "round":
            # The strip has to fit inside the disc, corners and all.
            radius = self.specimen_length / 2.0
            half = self.beam_width / 2.0
            if half >= radius:
                return 180.0
            length = 2.0 * math.sqrt(radius**2 - half**2)
        ratio = self.goniometer_radius * np.radians(self.divergence) / length
        if ratio >= 1.0:
            return 180.0
        return float(np.degrees(2.0 * np.arcsin(ratio)))

    def factor(self, two_theta: np.ndarray) -> np.ndarray:
        """The fraction of the beam intercepted by the specimen."""
        theta = np.radians(np.asarray(two_theta, dtype=float)) / 2.0
        irradiated = self.goniometer_radius * np.radians(self.divergence)
        if self.shape == "round":
            with np.errstate(divide="ignore", invalid="ignore"):
                strip = irradiated / np.sin(theta)
            return _disc_overlap(np.nan_to_num(strip, posinf=1e9),
                                 self.beam_width, self.specimen_length / 2.0)
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


def _disc_overlap(length: np.ndarray, width: float, radius: float) -> np.ndarray:
    """Fraction of a centred ``length x width`` strip that lies on a disc.

    The strip is the beam's footprint, centred on the mount; the disc is the
    mount.  Integrating the disc's half-height over the strip's half-length,

        area / 4 = b x1                           where the rectangle limits,
                 + [x sqrt(R^2 - x^2)/2 + R^2/2 asin(x/R)]   where the disc does

    with the crossover at ``x = sqrt(R^2 - b^2)``.  Exact, and cheap enough to
    evaluate per point of a scan.
    """
    length = np.atleast_1d(np.asarray(length, dtype=float))
    half_width = min(width, 2.0 * radius) / 2.0
    half_length = np.minimum(length / 2.0, radius)
    crossover = math.sqrt(max(radius**2 - half_width**2, 0.0))
    flat = np.minimum(half_length, crossover)

    def curved(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, radius)
        return 0.5 * x * np.sqrt(np.maximum(radius**2 - x**2, 0.0)) + \
            0.5 * radius**2 * np.arcsin(np.clip(x / radius, -1.0, 1.0))

    quarter = half_width * flat + np.where(
        half_length > crossover, curved(half_length) - curved(flat), 0.0)
    area = 4.0 * quarter
    beam = length * width
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(beam > 0.0, area / beam, 0.0)
    return np.clip(np.nan_to_num(fraction), 0.0, 1.0)
