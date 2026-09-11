"""Peak shapes and instrumental broadening.

Reflection widths follow the usual Caglioti form for the instrumental
contribution,

    FWHM_inst(theta)^2 = U tan^2(theta) + V tan(theta) + W    [deg^2]

optionally combined with a Scherrer size broadening.  Clay crystallites are
strongly anisotropic - thin along ``c*`` and wider in the ``ab`` plane - so the
effective coherent domain size is interpolated between the two directions as

    1 / L(alpha) = cos^2(alpha) / L_c + sin^2(alpha) / L_ab

with ``alpha`` the angle between the reflection vector and ``c*``.  For basal
reflections of a mixed-layer stack the thickness broadening is already contained
in the interference function of :mod:`clayquant.mixed_layer`, so ``size_c``
should be left at ``None`` there to avoid counting it twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["PeakShape", "pseudo_voigt", "accumulate_peaks", "convolve_variable_fwhm"]

_GAUSS_NORM = 2.0 * math.sqrt(math.log(2.0) / math.pi)
_LORENTZ_NORM = 2.0 / math.pi
_SCHERRER = 0.9


def pseudo_voigt(x: np.ndarray, fwhm: np.ndarray | float, eta: float) -> np.ndarray:
    """Normalised pseudo-Voigt profile evaluated at offsets ``x``.

    ``eta`` is the Lorentzian fraction: 0 gives a Gaussian, 1 a Lorentzian.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0, 1]")
    fwhm = np.asarray(fwhm, dtype=float)
    gauss = (_GAUSS_NORM / fwhm) * np.exp(-4.0 * math.log(2.0) * (x / fwhm) ** 2)
    lorentz = (_LORENTZ_NORM / fwhm) / (1.0 + 4.0 * (x / fwhm) ** 2)
    return eta * lorentz + (1.0 - eta) * gauss


@dataclass
class PeakShape:
    """Reflection width model.

    Attributes
    ----------
    u, v, w:
        Caglioti coefficients of the instrumental FWHM in deg^2.
    eta:
        Lorentzian fraction of the pseudo-Voigt profile.
    size_c, size_ab:
        Coherent domain sizes in A along ``c*`` and in the ``ab`` plane.
        ``None`` disables the corresponding size broadening.
    """

    u: float = 0.0
    v: float = 0.0
    w: float = 0.01
    eta: float = 0.5
    size_c: float | None = None
    size_ab: float | None = None

    def fwhm(
        self,
        two_theta: np.ndarray,
        wavelength: float,
        alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        """Total FWHM in degrees at the given ``two_theta`` values."""
        two_theta = np.asarray(two_theta, dtype=float)
        theta = np.radians(two_theta) / 2.0
        tan_theta = np.tan(theta)
        instrumental = self.u * tan_theta**2 + self.v * tan_theta + self.w
        variance = np.clip(instrumental, 1e-8, None)

        if self.size_c is not None or self.size_ab is not None:
            size_c = self.size_c if self.size_c is not None else np.inf
            size_ab = self.size_ab if self.size_ab is not None else np.inf
            if alpha is None:
                inverse_size = 1.0 / size_c
            else:
                alpha = np.asarray(alpha, dtype=float)
                inverse_size = np.cos(alpha) ** 2 / size_c + np.sin(alpha) ** 2 / size_ab
            with np.errstate(divide="ignore", invalid="ignore"):
                size_fwhm = np.degrees(
                    _SCHERRER * wavelength * inverse_size / np.maximum(np.cos(theta), 1e-6)
                )
            variance = variance + np.nan_to_num(size_fwhm) ** 2
        return np.sqrt(variance)


def accumulate_peaks(
    grid: np.ndarray,
    centers: np.ndarray,
    intensities: np.ndarray,
    fwhms: np.ndarray,
    eta: float,
    cutoff: float = 12.0,
) -> np.ndarray:
    """Sum peak profiles onto ``grid``.

    Each peak is evaluated only within ``cutoff`` times its FWHM, which keeps
    the cost proportional to the number of reflections rather than to their
    product with the grid size.
    """
    grid = np.asarray(grid, dtype=float)
    total = np.zeros_like(grid)
    if len(centers) == 0:
        return total
    step = float(np.mean(np.diff(grid))) if len(grid) > 1 else 1.0
    for center, intensity, fwhm in zip(centers, intensities, fwhms):
        if intensity == 0.0:
            continue
        half_window = cutoff * fwhm
        start = int(np.searchsorted(grid, center - half_window))
        stop = int(np.searchsorted(grid, center + half_window))
        if stop <= start:
            # Peak falls between grid points: put it on the nearest one.
            nearest = int(np.clip(np.searchsorted(grid, center), 0, len(grid) - 1))
            total[nearest] += intensity / step
            continue
        window = grid[start:stop]
        total[start:stop] += intensity * pseudo_voigt(window - center, fwhm, eta)
    return total


def convolve_variable_fwhm(
    grid: np.ndarray,
    y: np.ndarray,
    fwhm: np.ndarray,
    eta: float,
    cutoff: float = 12.0,
) -> np.ndarray:
    """Convolve ``y`` with a pseudo-Voigt kernel whose width varies along ``grid``.

    Used to apply instrumental broadening to the continuous intensity curve of a
    mixed-layer stack, where the width changes with angle.
    """
    grid = np.asarray(grid, dtype=float)
    y = np.asarray(y, dtype=float)
    fwhm = np.broadcast_to(np.asarray(fwhm, dtype=float), grid.shape)
    if len(grid) < 2:
        return y.copy()
    step = float(np.mean(np.diff(grid)))
    result = np.zeros_like(y)
    for index, (position, value, width) in enumerate(zip(grid, y, fwhm)):
        if value == 0.0:
            continue
        span = max(1, int(cutoff * width / step))
        start = max(0, index - span)
        stop = min(len(grid), index + span + 1)
        kernel = pseudo_voigt(grid[start:stop] - position, width, eta)
        total = kernel.sum()
        if total > 0:
            result[start:stop] += value * kernel / total
    return result
