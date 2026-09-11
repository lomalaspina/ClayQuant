"""2theta zero-error calibration on the quartz 100 reflection.

Quartz ``100`` near 20.86 deg 2theta (d = 4.2551 A, from a = 4.9134 A) is the
usual internal standard for the zero error of an oriented clay mount: it is
strong, it is present in practically every sediment sample, and no clay basal
reflection falls on it.  The zero error is the offset that must be *subtracted*
from the measured angles to put that reflection at its true position.

Two routes are provided, matching how the calibration is done in practice:

* :func:`estimate_zero_error` fits the peak and returns the offset directly,
  snapped to a chosen step (0.01 deg by default, half the usual 0.02 deg
  measurement step).
* :func:`zero_error_profile` returns a figure of merit for every candidate
  offset on that step grid, which is what a GUI slider displays while the user
  walks the peak onto the reference line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pattern import Pattern

__all__ = [
    "QUARTZ_100_D",
    "QUARTZ_101_D",
    "reference_two_theta",
    "ZeroErrorResult",
    "estimate_zero_error",
    "zero_error_profile",
    "apply_zero_error",
    "peak_position",
]

QUARTZ_100_D = 4.25510
"""d(100) of quartz in A (a = 4.91344 A, d = a * sqrt(3) / 2)."""

QUARTZ_101_D = 3.34346
"""d(101) of quartz in A, the strongest quartz reflection."""

CU_KA1 = 1.540596


def reference_two_theta(d: float = QUARTZ_100_D, wavelength: float = CU_KA1) -> float:
    """Bragg angle in degrees for a d-spacing and wavelength."""
    argument = wavelength / (2.0 * d)
    if not -1.0 < argument < 1.0:
        raise ValueError("the reflection does not exist for this wavelength")
    return float(np.degrees(2.0 * np.arcsin(argument)))


def peak_position(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    center: float,
    window: float = 1.0,
    method: str = "centroid",
    threshold: float = 0.5,
) -> float:
    """Locate a peak near ``center`` within +/- ``window`` degrees.

    A straight baseline through the two window edges is removed first.

    Parameters
    ----------
    method:
        ``"centroid"`` uses the intensity-weighted centroid of the points above
        ``threshold`` times the window maximum, which is robust against
        asymmetry from the Ka doublet; ``"parabola"`` fits a parabola through
        the maximum and its two neighbours; ``"max"`` returns the grid point of
        the maximum.
    threshold:
        Fraction of the peak maximum used by the ``"centroid"`` method.
    """
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    inside = np.flatnonzero(np.abs(two_theta - center) <= window)
    if inside.size < 3:
        raise ValueError(
            f"fewer than three points within {window} deg of {center} deg; "
            f"widen the window or check the scan range"
        )
    x = two_theta[inside]
    y = intensity[inside]
    baseline = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
    y = y - baseline
    if y.max() <= 0:
        raise ValueError(f"no peak found near {center} deg 2theta")

    if method == "max":
        return float(x[np.argmax(y)])
    if method == "parabola":
        apex = int(np.argmax(y))
        if 0 < apex < len(y) - 1:
            left, middle, right = y[apex - 1], y[apex], y[apex + 1]
            denominator = left - 2.0 * middle + right
            if denominator != 0.0:
                step = x[apex] - x[apex - 1]
                return float(x[apex] - 0.5 * step * (right - left) / denominator)
        return float(x[apex])
    if method == "centroid":
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        keep = y >= threshold * y.max()
        return float(np.sum(x[keep] * y[keep]) / np.sum(y[keep]))
    raise ValueError("method must be 'centroid', 'parabola' or 'max'")


@dataclass
class ZeroErrorResult:
    """Outcome of a zero-error determination."""

    shift: float
    observed_two_theta: float
    reference_two_theta: float
    method: str
    height: float = 0.0
    noise: float = 1.0
    detected: bool = True
    note: str = ""

    @property
    def raw_shift(self) -> float:
        """The offset before snapping to the step grid."""
        return self.observed_two_theta - self.reference_two_theta

    @property
    def signal_to_noise(self) -> float:
        return self.height / self.noise if self.noise > 0 else 0.0


def estimate_zero_error(
    pattern: Pattern,
    reference: float | None = None,
    window: float = 0.30,
    step: float = 0.01,
    method: str = "centroid",
    d: float = QUARTZ_100_D,
    wavelength: float = CU_KA1,
    min_signal_to_noise: float = 4.0,
) -> ZeroErrorResult:
    """Determine the 2theta zero error from the quartz 100 reflection.

    Parameters
    ----------
    reference:
        True position of the calibration reflection in degrees.  Defaults to the
        Bragg angle of ``d`` at ``wavelength``.
    window:
        Half-width in degrees of the search window.  It is deliberately narrow.
        Quartz 100 at 4.255 A does not overlap any clay *basal* reflection, but
        it is only 0.4 deg from the kaolinite 020 band at 4.36 A, and in a clay
        separate quartz can be weak or absent while kaolinite is strong.  With a
        wide window the routine then locks onto kaolinite and reports a zero
        error of about -0.4 deg, which is wrong and would shift the whole
        pattern.
    step:
        Grid the result is snapped to, 0.01 deg by default - half the usual
        0.02 deg measurement step.
    min_signal_to_noise:
        Height above the local noise the calibration peak must reach.  Below it
        the result is returned with ``detected = False`` and a zero shift rather
        than a meaningless number: set the zero error by hand, or calibrate on
        another reflection.
    """
    target = reference_two_theta(d, wavelength) if reference is None else float(reference)
    two_theta = pattern.two_theta
    inside = np.flatnonzero(np.abs(two_theta - target) <= window)
    if inside.size < 3:
        return ZeroErrorResult(
            shift=0.0,
            observed_two_theta=float("nan"),
            reference_two_theta=target,
            method=method,
            detected=False,
            note=(
                f"the calibration reflection at {target:.2f} deg lies outside the scan "
                f"({two_theta[0]:.2f} to {two_theta[-1]:.2f} deg)"
            ),
        )

    local = pattern.intensity[inside].astype(float)
    baseline = np.interp(two_theta[inside], [two_theta[inside[0]], two_theta[inside[-1]]],
                         [local[0], local[-1]])
    corrected = local - baseline
    height = float(corrected.max())

    # Noise from the scatter of successive points just outside the window.
    flank = np.flatnonzero(
        (np.abs(two_theta - target) > window) & (np.abs(two_theta - target) <= 3.0 * window)
    )
    if flank.size > 4:
        differences = np.diff(pattern.intensity[flank].astype(float))
        noise = max(float(np.median(np.abs(differences))) / 0.9539, 1.0)
    else:
        noise = max(math.sqrt(max(float(np.median(local)), 1.0)), 1.0)

    if height < min_signal_to_noise * noise:
        return ZeroErrorResult(
            shift=0.0,
            observed_two_theta=float("nan"),
            reference_two_theta=target,
            method=method,
            height=height,
            noise=noise,
            detected=False,
            note=(
                f"no calibration peak found at {target:.2f} +/- {window:.2f} deg "
                f"(height {height:.0f} counts, noise {noise:.0f}, "
                f"S/N {height / noise:.1f} < {min_signal_to_noise:g}). "
                f"In a clay separate quartz may be absent; set the zero error manually "
                f"or use another reflection."
            ),
        )

    observed = peak_position(
        two_theta, pattern.intensity, center=target, window=window, method=method
    )
    shift = observed - target
    if step > 0:
        shift = round(shift / step) * step
    return ZeroErrorResult(
        shift=float(shift),
        observed_two_theta=observed,
        reference_two_theta=target,
        method=method,
        height=height,
        noise=noise,
        detected=True,
    )


def zero_error_profile(
    pattern: Pattern,
    reference: float | None = None,
    span: float = 0.5,
    step: float = 0.01,
    d: float = QUARTZ_100_D,
    wavelength: float = CU_KA1,
) -> tuple[np.ndarray, np.ndarray]:
    """Figure of merit for each candidate zero error on a ``step`` grid.

    For every trial offset the measured pattern is interpolated at the reference
    angle; the offset that maximises the result is the one that brings the
    calibration peak onto the reference line.  Returns ``(shifts, merit)``, with
    ``merit`` normalised to a maximum of 1, ready to plot under a slider.
    """
    target = reference_two_theta(d, wavelength) if reference is None else float(reference)
    count = int(round(2.0 * span / step)) + 1
    shifts = -span + step * np.arange(count)
    merit = np.interp(
        target + shifts, pattern.two_theta, pattern.intensity, left=0.0, right=0.0
    )
    peak = merit.max()
    return shifts, (merit / peak if peak > 0 else merit)


def apply_zero_error(pattern: Pattern, shift: float) -> Pattern:
    """Return a copy of ``pattern`` with the zero error removed."""
    corrected = Pattern(
        two_theta=pattern.two_theta - float(shift),
        intensity=pattern.intensity.copy(),
        name=pattern.name,
        metadata={**pattern.metadata, "zero_error": float(shift)},
    )
    return corrected
