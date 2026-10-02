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
    "DisplacementResult",
    "estimate_displacement",
    "apply_displacement",
    "basal_series_positions",
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


# --------------------------------------------------------------------------
# Specimen displacement
# --------------------------------------------------------------------------

DEFAULT_GONIOMETER_RADIUS = 240.0
"""Goniometer radius in mm used when a pattern's file does not record one."""

MAXIMUM_DISPLACEMENT = 1.0
"""Largest |s| in mm the solver will report, beyond which it refuses.

A mount further off the focusing circle than a millimetre is a mount that was
not pressed flat, and a number fitted to it is describing the preparation rather
than correcting it.
"""


@dataclass
class DisplacementResult:
    """Outcome of a specimen-displacement determination."""

    displacement: float
    """Distance in mm the specimen surface sits off the focusing circle.

    Positive means the surface stands proud of the circle, which moves every
    line to *lower* 2theta.  Read it as a parameterisation of the angular
    correction and not as a measurement of the mount: over a clay scan it is
    nearly degenerate with a zero error, and :attr:`equivalent_zero_error` gives
    the constant that fits the same data almost as well.
    """

    orders: tuple[float, ...]
    """The reflection positions, in degrees, the solve was made from."""

    spacing: float
    """The layer repeat in A the corrected orders agree on."""

    spread_before: float
    """Largest minus smallest d(001) implied by the orders, uncorrected, in A."""

    spread_after: float
    """The same spread once the displacement is removed."""

    radius: float = DEFAULT_GONIOMETER_RADIUS
    detected: bool = True
    note: str = ""

    @property
    def equivalent_zero_error(self) -> float:
        """The constant 2theta shift that does nearly the same job, in degrees.

        Negated to the sign convention of :func:`apply_zero_error`, which
        subtracts.  Offered because the two are not separable over a clay scan
        and a reader who thinks in zero errors should not have to convert.
        """
        if not self.orders:
            return 0.0
        shifts = displacement_shift(np.asarray(self.orders, dtype=float),
                                    self.displacement, self.radius)
        return -float(np.mean(shifts))


def displacement_shift(
    two_theta: np.ndarray | float,
    displacement: float,
    radius: float = DEFAULT_GONIOMETER_RADIUS,
) -> np.ndarray:
    """How far a displaced specimen moves each line, in degrees.

    A flat specimen whose surface sits a distance ``s`` off the focusing circle
    reflects from the wrong place, and the error is

        d(2theta) = -(2 s / R) cos(theta)

    in radians, with ``R`` the goniometer radius: largest at low angle, and
    vanishing towards 2theta = 180 deg.

    **Over a clay scan this is very nearly a constant.** From 6 to 38 deg,
    cos(theta) runs from 0.9985 to 0.9468, so 0.25 mm on a 240 mm goniometer
    shifts every line by between 0.1192 and 0.1130 deg - a variation of
    0.006 deg, a twentieth of a peak width, on a shift of 0.117.  A zero error
    of -0.117 deg therefore absorbs about 98 per cent of it, and the two
    corrections are not separable from a measurement over this range.  The
    distinction matters at high angle and on a scan that reaches it; here it
    does not, and :func:`estimate_displacement` says so rather than claiming a
    millimetre it cannot measure.
    """
    angles = np.asarray(two_theta, dtype=float)
    return np.degrees(2.0 * float(displacement) / float(radius)
                      * np.cos(np.radians(angles / 2.0)))


def apply_displacement(
    pattern: Pattern,
    displacement: float,
    radius: float | None = None,
) -> Pattern:
    """Return a copy of ``pattern`` with the specimen displacement removed.

    The angular axis is corrected, not resampled: every point keeps its counts
    and moves to where it would have been measured from a specimen on the
    focusing circle.  The step therefore stops being exactly constant, by about
    a thousandth of a degree across a clay scan, which every consumer here
    handles because they interpolate rather than index.
    """
    if radius is None:
        radius = float((pattern.metadata or {}).get(
            "goniometer_radius", DEFAULT_GONIOMETER_RADIUS))
    angles = np.asarray(pattern.two_theta, dtype=float)
    return Pattern(
        two_theta=angles + displacement_shift(angles, displacement, radius),
        intensity=pattern.intensity.copy(),
        name=pattern.name,
        metadata={**(pattern.metadata or {}),
                  "displacement": float(displacement),
                  "displacement_radius": float(radius)},
    )


def basal_series_positions(
    pattern: Pattern,
    first: tuple[float, float],
    orders: int = 3,
    window: float = 0.45,
    wavelength: float = CU_KA1,
    minimum_height: float = 0.0,
) -> tuple[float, ...]:
    """Measure the 00l positions of one basal series, as many orders as show.

    ``first`` is the 2theta window to look for the 001 in.  Each further order is
    then looked for where the measured 001 puts it, not where a nominal spacing
    does, so a series at an unexpected spacing is still followed.
    """
    angles = np.asarray(pattern.two_theta, dtype=float)
    counts = np.asarray(pattern.intensity, dtype=float)
    low, high = sorted(float(v) for v in first)
    inside = (angles >= low) & (angles <= high)
    if not np.any(inside):
        return ()
    centre = 0.5 * (low + high)
    try:
        found = peak_position(angles, counts, centre, window=0.5 * (high - low))
    except ValueError:
        return ()
    spacing = wavelength / (2.0 * np.sin(np.radians(found / 2.0)))
    out = [float(found)]
    for order in range(2, int(orders) + 1):
        sine = order * wavelength / (2.0 * spacing)
        if not 0.0 < sine < 1.0:
            break
        nominal = 2.0 * np.degrees(np.arcsin(sine))
        near = (angles >= nominal - window) & (angles <= nominal + window)
        if not np.any(near) or float(np.max(counts[near])) <= minimum_height:
            continue
        try:
            out.append(float(peak_position(angles, counts, nominal, window=window)))
        except ValueError:
            continue
    return tuple(out)


def estimate_displacement(
    pattern: Pattern,
    first: tuple[float, float] = (11.6, 13.0),
    orders: int = 3,
    radius: float | None = None,
    wavelength: float = CU_KA1,
    window: float = 0.45,
    limit: float = MAXIMUM_DISPLACEMENT,
) -> DisplacementResult:
    """Solve the specimen displacement from two or more orders of one series.

    A single reflection cannot separate an angular offset from a spacing - any
    position is explained by either - so this needs a series.  Two orders fix
    both at once: an offset of a tenth of a degree is a large error in d at the
    001 and a small one at the 003, while a change of spacing moves every order
    in proportion.  The tell in uncorrected data is that the apparent d(001)
    *falls* with order.  Three kaolin standards measured here gave 7.197, 7.161
    and 7.148 A from their first three orders.

    **What this does and does not measure.** It finds the correction that makes
    the orders of one series agree, and reports it in millimetres of specimen
    displacement because that is the physically right shape.  It is not a
    measurement of where the mount sat: over a 4 to 40 deg scan a displacement
    and a constant zero error are nearly degenerate (see
    :func:`displacement_shift`), so a 0.25 mm solution and a -0.117 deg zero
    error fit the same data to within 0.006 deg, and a specimen that really is
    displaced by 0.25 mm cannot be told from one sitting true on a goniometer
    0.117 deg out of zero.  Use the number to correct the axis, not to describe
    the preparation; and if a separate also has quartz, set the zero error from
    that first and let this take up what is left.

    What it is genuinely for is the case this program had no answer to: a
    specimen with no quartz, where :func:`estimate_zero_error` has no reference
    line to work from and the axis was simply left uncorrected.  A basal series
    is its own reference, because its orders have to be consistent.

    ``first`` is where to look for the 001 of the series to solve from; the
    default is the 7.15 A window, the strongest line most clay separates have.
    Returns a :class:`DisplacementResult` whose ``detected`` is False, with the
    reason in ``note``, where fewer than two orders could be measured or the
    solution falls outside ``limit``.
    """
    if radius is None:
        radius = float((pattern.metadata or {}).get(
            "goniometer_radius", DEFAULT_GONIOMETER_RADIUS))
    positions = basal_series_positions(
        pattern, first, orders=orders, window=window, wavelength=wavelength)
    if len(positions) < 2:
        return DisplacementResult(
            displacement=0.0, orders=positions, spacing=float("nan"),
            spread_before=float("nan"), spread_after=float("nan"), radius=radius,
            detected=False,
            note=("fewer than two orders of the series were measurable, and one "
                  "reflection cannot tell a displacement from a spacing"),
        )

    def spacings(shift: float) -> np.ndarray:
        angles = np.asarray(positions, dtype=float)
        moved = angles + displacement_shift(angles, shift, radius)
        implied = wavelength / (2.0 * np.sin(np.radians(moved / 2.0)))
        return implied * np.arange(1, len(positions) + 1)

    def spread(shift: float) -> float:
        values = spacings(shift)
        return float(np.max(values) - np.min(values))

    before = spread(0.0)
    # The spread is piecewise smooth and single-minimum in s over any sane
    # range, so a golden-section search on it is enough and needs no derivative.
    low, high = -float(limit), float(limit)
    golden = 0.5 * (np.sqrt(5.0) - 1.0)
    a, b = low, high
    c, d = b - golden * (b - a), a + golden * (b - a)
    for _ in range(200):
        if spread(c) < spread(d):
            b = d
        else:
            a = c
        c, d = b - golden * (b - a), a + golden * (b - a)
        if abs(b - a) < 1e-9:
            break
    solved = 0.5 * (a + b)
    after = spread(solved)
    if abs(solved) >= limit * 0.999:
        return DisplacementResult(
            displacement=0.0, orders=positions, spacing=float("nan"),
            spread_before=before, spread_after=before, radius=radius,
            detected=False,
            note=(f"the orders are best reconciled by a displacement of "
                  f"{solved:+.2f} mm, outside the {limit:g} mm this will report; "
                  f"the series is more likely two minerals than one displaced one"),
        )
    return DisplacementResult(
        displacement=float(solved),
        orders=positions,
        spacing=float(np.mean(spacings(solved))),
        spread_before=before,
        spread_after=after,
        radius=radius,
        detected=True,
        note=(f"{len(positions)} orders at "
              + ", ".join(f"{p:.3f}" for p in positions)
              + f" deg imply layer repeats spread over {before:.4f} A; a specimen "
              f"{solved:+.3f} mm off the focusing circle brings them to "
              f"{after:.4f} A at {float(np.mean(spacings(solved))):.4f} A"),
    )
